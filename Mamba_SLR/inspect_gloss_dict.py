# inspect_gloss_dict.py
import os, sys, json
import numpy as np
from collections import Counter

# --------- edit if your path differs ----------
GLOSS_DICT_PATH = "data/phoenix2014/gloss_dict.npy"
OUT_NORMALIZED  = "data/phoenix2014/gloss_dict_normalized.npy"
# ---------------------------------------------

def to_int_maybe(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str) and x.isdigit():
        return int(x)
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        for y in (x.tolist() if isinstance(x, np.ndarray) else x):
            if isinstance(y, (int, np.integer)):
                return int(y)
            if isinstance(y, str) and y.isdigit():
                return int(y)
    return None

def to_str_maybe(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        y = x[0]
        if isinstance(y, str):
            return y
        if isinstance(y, bytes):
            return y.decode("utf-8", errors="ignore")
    return str(x)

def load_raw(path):
    arr = np.load(path, allow_pickle=True)
    if hasattr(arr, "shape"):
        print(f"[info] np.load: type={type(arr)}, shape={arr.shape}, dtype={getattr(arr,'dtype',None)}")
    if hasattr(arr, "item") and arr.shape == ():
        obj = arr.item()
        print("[info] Detected 0-d numpy object; converted via .item() ->", type(obj))
        return obj
    return arr

def summarize_mapping(obj, max_print=10):
    print("\n--- quick sample ---")
    if isinstance(obj, dict):
        print(f"dict size: {len(obj)}")
        k_types = Counter(type(k).__name__ for k in obj.keys())
        v_types = Counter(type(v).__name__ for v in obj.values())
        print("key types:", dict(k_types))
        print("val types:", dict(v_types))
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_print: break
            print(f"  [{i}] {repr(k)} -> {repr(v)}")
    elif isinstance(obj, (list, tuple, np.ndarray)):
        L = len(obj)
        print(f"sequence length: {L}")
        for i in range(min(max_print, L)):
            print(f"  [{i}] {repr(obj[i])}  (type={type(obj[i]).__name__})")
    else:
        print(f"object of type {type(obj)}")
    print("--------------------\n")

def normalize(obj):
    """
    Returns gloss2id (dict[str,int]), id2gloss (dict[int,str]).
    Handles:
      - dict{gloss->id or [id]}
      - dict{id->gloss or [gloss]}
      - list/array of pairs (gloss,id) or (id,gloss)
    """
    gloss2id, id2gloss = {}, {}

    if isinstance(obj, dict):
        # decide orientation from first item
        try:
            k0, v0 = next(iter(obj.items()))
        except StopIteration:
            raise ValueError("empty gloss dict")

        if isinstance(k0, (str, bytes)):
            # keys are gloss strings
            for k, v in obj.items():
                g = to_str_maybe(k)
                i = to_int_maybe(v)
                if i is None:
                    raise ValueError(f"cannot parse id from value {v} for gloss {k}")
                gloss2id[g] = i
        elif isinstance(k0, (int, np.integer)) or to_int_maybe(k0) is not None:
            # keys are ids
            for k, v in obj.items():
                i = to_int_maybe(k)
                g = to_str_maybe(v)
                if i is None or g is None:
                    raise ValueError(f"cannot parse mapping from ({k}:{v})")
                id2gloss[int(i)] = g
        else:
            # fallback: coerce
            for k, v in obj.items():
                g = to_str_maybe(k)
                i = to_int_maybe(v)
                if i is None:
                    raise ValueError(f"cannot parse id from value {v} for gloss {k}")
                gloss2id[g] = i

        if not gloss2id and id2gloss:
            gloss2id = {g: i for i, g in id2gloss.items()}
        if not id2gloss and gloss2id:
            id2gloss = {i: g for g, i in gloss2id.items()}

    elif isinstance(obj, (list, tuple, np.ndarray)):
        # expect sequence of 2-tuples
        tmp = {}
        seq = obj.tolist() if isinstance(obj, np.ndarray) else obj
        for item in seq:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            a, b = item
            if isinstance(a, (int, np.integer)) or to_int_maybe(a) is not None:
                i, g = to_int_maybe(a), to_str_maybe(b)
            else:
                i, g = to_int_maybe(b), to_str_maybe(a)
            if i is not None and g is not None:
                tmp[int(i)] = g
        if not tmp:
            raise ValueError("could not parse list/array format into pairs")
        id2gloss = tmp
        gloss2id = {g: i for i, g in id2gloss.items()}

    else:
        raise ValueError(f"unsupported gloss_dict type: {type(obj)}")

    # sanity & duplicate checks
    if not gloss2id or not id2gloss:
        raise ValueError("parsed empty gloss maps")

    # detect duplicate ids / glosses
    rev_counts = Counter(gloss2id.values())
    dup_ids = [i for i, c in rev_counts.items() if c > 1]
    if dup_ids:
        print(f"[warn] duplicate ids detected (multiple glosses share same id): {dup_ids[:10]} ...")

    return gloss2id, id2gloss

def main(path, out_path):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    raw = load_raw(path)
    summarize_mapping(raw)

    try:
        g2i, i2g = normalize(raw)
    except Exception as e:
        print(f"❌ normalization failed: {e}")
        sys.exit(2)

    print(f"[ok] normalized: gloss2id={len(g2i)} entries, id2gloss={len(i2g)} entries")

    # show a few samples
    print("\n--- samples gloss→id ---")
    for j, (g, i) in enumerate(list(g2i.items())[:10]):
        print(f"  {g!r} -> {i}")
    print("--- samples id→gloss ---")
    for j, (i, g) in enumerate(list(i2g.items())[:10]):
        print(f"  {i} -> {g!r}")
    print("------------------------\n")

    # save normalized (as dict gloss->id)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, g2i, allow_pickle=True)
    print(f"[saved] normalized gloss dict → {out_path}")

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else GLOSS_DICT_PATH
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_NORMALIZED
    main(p, out)
