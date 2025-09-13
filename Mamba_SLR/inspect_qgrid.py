# inspect_qgrid.py
import os, json, argparse, math, random, re
import numpy as np
from collections import Counter, defaultdict

def human_bytes(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def norm_key(s: str) -> str:
    # lowercase, keep only a-z0-9
    return "".join(ch for ch in s.lower() if ch.isalnum())

def list_npy(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".npy"):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)

def sample_indices(T, max_samples=200):
    if T <= max_samples: return np.arange(T)
    # evenly spaced samples across time
    step = T / max_samples
    return np.array([int(i*step) for i in range(max_samples)])

def analyze_file(path, sample_time=True, max_time_samples=200):
    arr = np.load(path, mmap_mode="r")  # cheap to read metadata
    shape = tuple(arr.shape)
    dtype = str(arr.dtype)
    ndim  = arr.ndim
    info = {"path": path, "ndim": ndim, "shape": shape, "dtype": dtype}
    # estimate size
    try:
        itemsize = arr.dtype.itemsize
        info["nbytes"] = int(np.prod(shape) * itemsize)
    except Exception:
        info["nbytes"] = None

    # sample values
    uniques = None
    vmin = vmax = None
    try:
        if ndim == 3:   # (T,H,W)
            T = shape[0]
            idx = sample_indices(T, max_time_samples) if sample_time else np.arange(T)
            # slice only selected time steps (memmap reads them on demand)
            vals = arr[idx, ...]
        elif ndim == 4: # (T,C,H,W) or (C,T,H,W)
            # try to guess time axis: pick the axis >= 100 typically
            axes = list(shape)
            t_axis = 0 if shape[0] >= 100 else (1 if shape[1] >= 100 else 0)
            T = shape[t_axis]
            take = [slice(None)]*4
            idx = sample_indices(T, max_time_samples) if sample_time else np.arange(T)
            take[t_axis] = idx
            vals = arr[tuple(take)]
        else:
            vals = np.asarray(arr)  # small files
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        # only compute uniques on small sample to avoid huge cost
        flat = np.asarray(vals).ravel()
        if flat.size > 200000:
            flat = flat[:200000]
        uniques = np.unique(flat)
        if uniques.size > 20:  # cap reporting size
            uniques = uniques[:20]
        uniques = [int(u) if float(u).is_integer() else float(u) for u in uniques]
    except Exception as e:
        info["sample_error"] = repr(e)

    info["vmin"] = vmin
    info["vmax"] = vmax
    info["uniques_sample"] = uniques
    return info

def build_index(npy_paths):
    index = {}
    stems = {}
    for p in npy_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        key_variants = {
            stem,
            stem.lower(),
            norm_key(stem),
            norm_key(stem.replace("-", "_")),
            norm_key(stem.replace("_", "-")),
        }
        for k in key_variants:
            if k not in index:
                index[k] = p
        stems[stem] = p
    return index, stems

def load_meta(meta_dir, split):
    sp = {"val":"dev","dev":"dev","test":"test","train":"train"}.get(split, split)
    path = os.path.join(meta_dir, f"{sp}_info.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=True).tolist()
    if isinstance(obj, dict):
        obj = list(obj.values())
    return obj

def try_match_fileid(fileid, index):
    cands = [fileid, fileid.lower(), fileid.replace("-", "_"), fileid.replace("_", "-")]
    if "-" in fileid and fileid.rsplit("-", 1)[-1].isdigit():
        cands.append(fileid.rsplit("-", 1)[0])
    for c in cands:
        k = norm_key(c)
        if k in index:
            return index[k]
    # relaxed: startswith containment
    k0 = norm_key(fileid)
    for k, p in index.items():
        if k.startswith(k0) or k0.startswith(k):
            return p
    # last resort: substring
    for k, p in index.items():
        if k0 in k or k in k0:
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qgrid_prefix", required=True, help="Root dir with Qgrid .npy files")
    ap.add_argument("--meta_dir", default="data/phoenix2014", help="Directory with *_info.npy")
    ap.add_argument("--split", default="train", choices=["train","dev","val","test"])
    ap.add_argument("--sample", type=int, default=25, help="how many files to fully inspect")
    ap.add_argument("--out_report", default="qgrid_report.json")
    ap.add_argument("--out_index", default="qgrid_index.json")
    args = ap.parse_args()

    print(f"[info] scanning: {args.qgrid_prefix}")
    npy_paths = list_npy(args.qgrid_prefix)
    print(f"[info] found {len(npy_paths)} .npy files")

    # quick stats over shapes/dtypes without loading values
    shape_counts = Counter()
    ndim_counts  = Counter()
    dtype_counts = Counter()
    total_bytes  = 0
    for p in npy_paths[:min(2000, len(npy_paths))]:  # cheap pass
        arr = np.load(p, mmap_mode="r")
        shape_counts[tuple(arr.shape)] += 1
        ndim_counts[arr.ndim] += 1
        dtype_counts[str(arr.dtype)] += 1
        try:
            total_bytes += arr.size * arr.dtype.itemsize
        except Exception:
            pass

    print("[info] shape frequencies (first ~2000):")
    for shp, cnt in shape_counts.most_common(10):
        print(f"  {shp}: {cnt}")
    print("[info] ndim frequencies:", dict(ndim_counts))
    print("[info] dtype frequencies:", dict(dtype_counts))
    print(f"[info] approx total size (first ~2000 files): {human_bytes(total_bytes)}")

    # deep sample
    rng = random.Random(1234)
    sample_paths = rng.sample(npy_paths, k=min(args.sample, len(npy_paths)))
    per_file = []
    for p in sample_paths:
        info = analyze_file(p, sample_time=True, max_time_samples=200)
        per_file.append(info)
        print(f"  {os.path.basename(p)} -> shape={info['shape']} dtype={info['dtype']} "
              f"vmin={info['vmin']} vmax={info['vmax']} uniq≈{info['uniques_sample']}")

    # build index
    index, stems = build_index(npy_paths)
    with open(args.out_index, "w") as f:
        json.dump(index, f)
    print(f"[info] wrote index with {len(index)} keys → {args.out_index}")

    # optional meta matching
    meta = load_meta(args.meta_dir, args.split)
    unmatched = []
    matched = 0
    examples = []
    for it in meta[:5000]:  # cap for speed
        fid = it.get("fileid") or it.get("id") or it.get("name")
        if not fid: continue
        path = try_match_fileid(fid, index)
        if path is None:
            unmatched.append(fid)
            if len(examples) < 10:
                examples.append(fid)
        else:
            matched += 1

    print(f"[match] {matched} matched, {len(unmatched)} unmatched (first 10 unmatched): {examples}")

    # aggregate report
    report = {
        "root": os.path.abspath(args.qgrid_prefix),
        "total_files": len(npy_paths),
        "ndim_counts": dict(ndim_counts),
        "dtype_counts": dict(dtype_counts),
        "shape_counts_top10": [(list(map(int, shp)), cnt) for shp, cnt in shape_counts.most_common(10)],
        "sample_analysis": per_file,
        "matched_estimate": {"matched": matched, "unmatched": len(unmatched)},
        "unmatched_examples_first10": examples,
    }
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[info] wrote report → {args.out_report}")

    # print loader suggestion
    print("\n=== loader suggestion ===")
    print("* If most arrays are (T,H,W): treat as single-channel and unsqueeze C=1.")
    print("* If most arrays are (T,C,H,W): pass through as-is.")
    print("* If some arrays are (C,T,H,W): permute to (T,C,H,W).")
    print("* Values look like -1/0/1? Then dtype float32 is fine; keep as-is.")
    print("* Use per-sample adaptive_avg_pool1d to map T_qgrid → T_img before fusion/CTC.")
    print("=========================\n")

if __name__ == "__main__":
    main()
