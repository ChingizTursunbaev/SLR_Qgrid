# diag_data_wiring.py
import os, glob, json, argparse, random, re, pickle, numpy as np
from collections import Counter, defaultdict

def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

SLUG_RE = re.compile(r"[0-9]{2}[A-Za-z]+_[0-9]{4}_.+?_default-\d+")

def extract_slug_from_path(path_like: str):
    if not path_like: return None
    parts = [p for p in re.split(r"[\\/]+", path_like) if p]
    for part in reversed(parts):
        if "default-" in part:
            m = SLUG_RE.search(part)
            return m.group(0) if m else part
    for part in reversed(parts):
        if "default" in part.lower():
            return part
    return None

def build_qgrid_index(qgrid_prefix: str):
    index = {}
    for dirpath, _, files in os.walk(qgrid_prefix):
        for f in files:
            if not f.endswith(".npy"): continue
            path = os.path.join(dirpath, f)
            stem = os.path.splitext(f)[0]
            keys = {
                stem, stem.lower(), norm(stem),
                norm(stem.replace("-", "_")), norm(stem.replace("_", "-"))
            }
            for k in keys:
                index.setdefault(k, path)
    return index

def build_image_index(image_prefix: str):
    index = {}
    for dirpath, _, files in os.walk(image_prefix):
        if any(fn.lower().endswith((".png",".jpg",".jpeg")) for fn in files):
            stem = os.path.basename(dirpath)
            keys = {
                stem, stem.lower(), norm(stem),
                norm(stem.replace("-", "_")), norm(stem.replace("_","-"))
            }
            slug = extract_slug_from_path(dirpath)
            if slug:
                keys |= {
                    slug, slug.lower(), norm(slug),
                    norm(slug.replace("-", "_")), norm(slug.replace("_","-"))
                }
            for k in keys:
                index.setdefault(k, dirpath)
    return index

def resolve_qgrid(fileid, folder, qidx, qgrid_prefix):
    # direct candidates
    if fileid:
        for cand in [f"{fileid}.npy", f"{fileid}_qgrid.npy", f"{fileid}_Qgrid.npy"]:
            p = os.path.join(qgrid_prefix, cand)
            if os.path.exists(p): return p
    # index lookups
    def lookup(key):
        return qidx.get(norm(key))
    if fileid:
        for fid in [fileid, fileid.lower(), fileid.replace("-", "_"), fileid.replace("_","-")]:
            p = lookup(fid)
            if p: return p
        if "-" in fileid and fileid.rsplit("-",1)[-1].isdigit():
            p = lookup(fileid.rsplit("-",1)[0])
            if p: return p
    if folder:
        slug = extract_slug_from_path(folder)
        if slug:
            for fid in [slug, slug.lower(), slug.replace("-", "_"), slug.replace("_","-")]:
                p = lookup(fid)
                if p: return p
        # substring fallback
        nf = norm(folder)
        best = None; best_len = -1
        for k, p in qidx.items():
            if k in nf and len(k) > best_len:
                best, best_len = p, len(k)
        if best: return best
    return None

def resolve_images(slug, folder, iidx, image_prefix):
    def lookup(key):
        return iidx.get(norm(key))
    # 1) by slug
    if slug:
        for fid in [slug, slug.lower(), slug.replace("-", "_"), slug.replace("_","-")]:
            p = lookup(fid)
            if p: return p
    # 2) explicit folder
    if folder:
        if os.path.isabs(folder):
            if os.path.isdir(folder) and glob.glob(os.path.join(folder, "*.[pj][pn]g")):
                return folder
        rel = os.path.join(image_prefix, folder)
        if os.path.isdir(rel) and glob.glob(os.path.join(rel, "*.[pj][pn]g")):
            return rel
        slug2 = extract_slug_from_path(folder)
        if slug2:
            for fid in [slug2, slug2.lower(), slug2.replace("-", "_"), slug2.replace("_","-")]:
                p = lookup(fid)
                if p: return p
        # substring fallback
        nf = norm(folder)
        best = None; best_len = -1
        for k, p in iidx.items():
            if k in nf and len(k) > best_len:
                best, best_len = p, len(k)
        if best: return best
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_prefix", required=True)
    ap.add_argument("--qgrid_prefix", required=True)
    ap.add_argument("--kp_pkl", required=True)
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--split", default="train", choices=["train","dev","val","test"])
    ap.add_argument("--gloss_dict", required=False)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--out", default="diag_wiring_report.json")
    args = ap.parse_args()

    split_map = {"val":"dev","dev":"dev","test":"test","train":"train"}
    meta_path = os.path.join(args.meta_dir, f"{split_map.get(args.split,args.split)}_info.npy")
    items = np.load(meta_path, allow_pickle=True).tolist()
    if isinstance(items, dict): items = list(items.values())
    print(f"[meta] {meta_path} -> {len(items)} entries")

    with open(args.kp_pkl,"rb") as f:
        kp_db = pickle.load(f)
    kp_keys = set(kp_db.keys())
    print(f"[kp] keys: {len(kp_keys)} (showing 5): {list(kp_keys)[:5]}")

    print("[index] building qgrid index…")
    qidx = build_qgrid_index(args.qgrid_prefix)
    print(f"[index] qgrid keys: {len(qidx)}")

    print("[index] building image index… (this can take a minute)")
    iidx = build_image_index(args.image_prefix)
    print(f"[index] image keys: {len(iidx)}")

    rng = random.Random(123)
    sample_items = rng.sample(items, k=min(args.samples, len(items)))

    results = []
    failures = Counter()

    for it in sample_items:
        if isinstance(it, dict):
            fileid = it.get("fileid") or it.get("id") or it.get("name") or it.get("video_id") or ""
            folder = it.get("folder") or it.get("path") or it.get("frames_path") or it.get("video_path") or ""
            label  = it.get("label") or it.get("gloss") or it.get("gt") or ""
        else:
            fileid, folder, label = (it, it, "")
        qpath = resolve_qgrid(fileid, folder, qidx, args.qgrid_prefix)
        if not qpath:
            results.append({"fileid":fileid, "folder":folder, "error":"no_qgrid"})
            failures["no_qgrid"] += 1
            continue
        slug = os.path.splitext(os.path.basename(qpath))[0]
        has_kp = slug in kp_keys or fileid in kp_keys
        if not has_kp:
            results.append({"fileid":fileid, "folder":folder, "slug":slug, "qgrid":os.path.basename(qpath), "error":"no_keypoints"})
            failures["no_keypoints"] += 1
            continue
        img_dir = resolve_images(slug, folder, iidx, args.image_prefix)
        if not img_dir:
            results.append({"fileid":fileid, "folder":folder, "slug":slug, "qgrid":os.path.basename(qpath), "error":"no_frames"})
            failures["no_frames"] += 1
            continue
        # quick stats
        frames = sorted(glob.glob(os.path.join(img_dir, "*.[pj][pn]g")))
        results.append({
            "fileid":fileid,
            "folder":folder,
            "slug":slug,
            "qgrid":os.path.basename(qpath),
            "frames_dir": img_dir.replace(args.image_prefix+"/",""),
            "n_frames": len(frames),
            "kp_key_used": slug if slug in kp_keys else (fileid if fileid in kp_keys else None),
            "label_len": len(label.strip().split()) if isinstance(label,str) else 0
        })

    print("\n[summary] sample scan:")
    ok = sum(1 for r in results if "error" not in r)
    print(f"  OK: {ok}  |  Failures: {sum(failures.values())} -> {dict(failures)}")
    print("  First 5 results:")
    for r in results[:5]:
        print("   ", r)

    with open(args.out,"w") as f:
        json.dump({
            "meta_path": meta_path,
            "image_prefix": args.image_prefix,
            "qgrid_prefix": args.qgrid_prefix,
            "samples_checked": len(results),
            "failures": dict(failures),
            "results": results
        }, f, indent=2)
    print(f"[saved] {args.out}")
    print("\nIf you see 'no_qgrid' or 'no_frames', the slug/fileid isn’t matching filenames. "
          "We can either (a) use the dataset that derives slug from qgrid and indexes images (I gave you that), "
          "or (b) patch your meta to store the true slug per item.")
if __name__ == "__main__":
    main()
