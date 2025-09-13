# slr/datasets/multi_modal_datasets.py
import os
import glob
import json
import pickle
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

# ===================== Gloss dict loader (robust) =====================
def _to_int_maybe(x):
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, str) and x.isdigit(): return int(x)
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        seq = x.tolist() if isinstance(x, np.ndarray) else list(x)
        for y in seq:
            if isinstance(y, (int, np.integer)): return int(y)
            if isinstance(y, str) and y.isdigit(): return int(y)
    return None

def _to_str_maybe(x):
    if isinstance(x, str): return x
    if isinstance(x, bytes): return x.decode("utf-8", errors="ignore")
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        y = x[0]
        if isinstance(y, str): return y
        if isinstance(y, bytes): return y.decode("utf-8", errors="ignore")
    return str(x)

def load_gloss_maps(gloss_dict_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    raw = np.load(gloss_dict_path, allow_pickle=True)
    if hasattr(raw, "item") and raw.shape == ():
        obj = raw.item()
    else:
        obj = raw

    gloss2id: Dict[str, int] = {}
    id2gloss: Dict[int, str] = {}

    if isinstance(obj, dict):
        try:
            k0, v0 = next(iter(obj.items()))
        except StopIteration:
            raise ValueError("gloss_dict is empty")

        if isinstance(k0, (str, bytes)):
            for k, v in obj.items():
                g = _to_str_maybe(k)
                i = _to_int_maybe(v)
                if i is None: raise ValueError(f"Cannot parse id from {v} for {k}")
                gloss2id[g] = i
            id2gloss = {i: g for g, i in gloss2id.items()}
        elif isinstance(k0, (int, np.integer)) or _to_int_maybe(k0) is not None:
            for k, v in obj.items():
                i = _to_int_maybe(k); g = _to_str_maybe(v)
                if i is None or g is None: raise ValueError(f"Cannot parse ({k}:{v})")
                id2gloss[int(i)] = g
            gloss2id = {g: i for i, g in id2gloss.items()}
        else:
            for k, v in obj.items():
                g = _to_str_maybe(k); i = _to_int_maybe(v)
                if i is None: raise ValueError(f"Cannot parse id from {v} for {k}")
                gloss2id[g] = i
            id2gloss = {i: g for g, i in gloss2id.items()}

    elif isinstance(obj, (list, tuple, np.ndarray)):
        seq = obj.tolist() if isinstance(obj, np.ndarray) else obj
        tmp = {}
        for item in seq:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a, b = item
                if isinstance(a, (int, np.integer)) or _to_int_maybe(a) is not None:
                    i = _to_int_maybe(a); g = _to_str_maybe(b)
                else:
                    i = _to_int_maybe(b); g = _to_str_maybe(a)
                if i is not None and g is not None:
                    tmp[int(i)] = g
        if not tmp: raise ValueError("Could not parse list/array gloss_dict format")
        id2gloss = tmp
        gloss2id = {g: i for i, g in id2gloss.items()}
    else:
        raise ValueError(f"Unsupported gloss_dict type: {type(obj)}")

    if not gloss2id or not id2gloss:
        raise ValueError("Parsed empty gloss maps")
    return gloss2id, id2gloss

# ===================== Image clip transform =====================
class ClipToTensorResize:
    """list of RGB frames -> (3,T,H,W) float32 in [0,1]"""
    def __init__(self, size: int = 224):
        self.size = size

    def _to_tensor(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        # copy to avoid non-writable view warning
        arr = np.array(img, dtype=np.uint8, copy=True)  # H,W,3
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        return t

    def __call__(self, frames_list: List[Any]) -> torch.Tensor:
        if len(frames_list) == 0:
            return torch.zeros(3, 1, self.size, self.size, dtype=torch.float32)
        frames = [self._to_tensor(f) for f in frames_list]
        clip = torch.stack(frames, dim=1)  # (3,T,H,W)
        return clip

# ===================== Helpers / resolvers =====================
def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

SLUG_RE = re.compile(r"[0-9]{2}[A-Za-z]+_[0-9]{4}_.+?_default-\d+")

def extract_slug_from_path(path_like: str) -> Optional[str]:
    """Find Phoenix slug inside a path string."""
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

def slug_base(slug_with_default: str) -> str:
    """Drop trailing '_default-<n>' → base slug (without the default index)."""
    return re.sub(r"_default-\d+$", "", slug_with_default)

def clean_folder_pattern(folder: str) -> str:
    """Convert 'train/slug/1/*.png' or '../..../train/slug/1/*.png' into directory path '.../train/slug/1'."""
    if not folder: return ""
    folder = re.sub(r"/\*\.png$|/\*\.jpg$|/\*$", "", folder)
    return folder

# ===================== Dataset with PRE-RESOLVE & FILTER =====================
class MultiModalPhoenixDataset(Dataset):
    """
    - images: image_prefix/<split>/<slug>/1/*.png  → transformed to (3,T,H,W)
    - qgrid:  qgrid .npy shape (T,121,2) int8 in {-1,0,1} → (T,242) float32
    - keypoints: kp_db uses full paths: 'fullFrame-210x260px/<split>/<slug>/1/<slug_base>'
      → we build an index so we can fetch by (<split>, <slug>) or (<split>, <slug_base>)
    - labels: gloss ids (CTC targets)
    """
    def __init__(self,
                 image_prefix: str,
                 qgrid_prefix: str,
                 kp_path: str,
                 meta_dir_path: str,
                 gloss_dict_path: str,
                 split: str = "train",
                 transforms: Optional[ClipToTensorResize] = None):

        super().__init__()
        self.image_prefix = image_prefix
        self.qgrid_prefix = qgrid_prefix
        self.transforms   = transforms if transforms is not None else ClipToTensorResize(size=224)
        self.split_req    = split  # user-requested split (train/dev/test)

        # gloss maps
        self.gloss_dict, self.id_to_gloss = load_gloss_maps(gloss_dict_path)

        # keypoints DB
        with open(kp_path, "rb") as f:
            self.kp_db: Dict[str, Any] = pickle.load(f)

        # ---- build keypoints index ----
        self.kp_index: Dict[Tuple[str, str], str] = {}
        for k in self.kp_db.keys():
            parts = [p for p in re.split(r"[\\/]+", k) if p]
            sp = None
            slug_wd = None
            for p in parts:
                if p in ("train", "dev", "test"):
                    sp = p
                if "default-" in p:
                    slug_wd = p
            if sp is None or slug_wd is None:
                continue
            self.kp_index[(sp, slug_wd)] = k
            self.kp_index[(sp, slug_base(slug_wd))] = k  # also index base

        # meta
        split_name = {"val": "dev", "dev": "dev", "test": "test", "train": "train"}.get(split, split)
        meta_path = os.path.join(meta_dir_path, f"{split_name}_info.npy")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        items = np.load(meta_path, allow_pickle=True).tolist()
        if isinstance(items, dict):
            items = list(items.values())
        if not isinstance(items, list):
            raise ValueError(f"Unexpected meta format in {meta_path}")

        # Qgrid filename index
        self.qgrid_index = self._build_qgrid_index(self.qgrid_prefix)

        # Build & filter samples now
        raw_samples = []
        for it in items:
            if isinstance(it, dict):
                fileid = it.get("fileid") or it.get("id") or it.get("name") or it.get("video_id") or ""
                folder = it.get("folder") or it.get("path") or it.get("frames_path") or it.get("video_path") or ""
                label  = it.get("label") or it.get("gloss") or it.get("gt") or ""
                raw_samples.append({"fileid": str(fileid), "folder": str(folder), "label": str(label)})
            elif isinstance(it, str):
                raw_samples.append({"fileid": it, "folder": it, "label": ""})
            else:
                continue

        self.rows = self._materialize_rows(raw_samples)  # pre-resolve + filter
        if len(self.rows) == 0:
            raise RuntimeError(
                "After pre-resolution, zero usable rows remained. "
                "Your meta likely lacks usable slugs; please inspect data/phoenix2014/*_info.npy."
            )

    @staticmethod
    def _build_qgrid_index(qgrid_prefix: str) -> Dict[str, str]:
        idx = {}
        for dirpath, _, files in os.walk(qgrid_prefix):
            for f in files:
                if not f.endswith(".npy"): continue
                path = os.path.join(dirpath, f)
                stem = os.path.splitext(f)[0]
                for k in {stem, stem.lower(), _norm(stem),
                          _norm(stem.replace("-", "_")), _norm(stem.replace("_","-"))}:
                    idx.setdefault(k, path)
        return idx

    def _resolve_qgrid_path(self, fileid: str, folder: str) -> Optional[Tuple[str, str]]:
        # 1) direct
        if fileid:
            for cand in [f"{fileid}.npy", f"{fileid}_qgrid.npy", f"{fileid}_Qgrid.npy"]:
                p = os.path.join(self.qgrid_prefix, cand)
                if os.path.exists(p):
                    slug = os.path.splitext(os.path.basename(p))[0]
                    return p, slug

        # 2) index variants
        def lookup(key):
            return self.qgrid_index.get(_norm(key))
        if fileid:
            for fid in [fileid, fileid.lower(), fileid.replace("-", "_"), fileid.replace("_","-")]:
                p = lookup(fid)
                if p:
                    slug = os.path.splitext(os.path.basename(p))[0]
                    return p, slug
            if "-" in fileid and fileid.rsplit("-", 1)[-1].isdigit():
                p = lookup(fileid.rsplit("-", 1)[0])
                if p:
                    slug = os.path.splitext(os.path.basename(p))[0]
                    return p, slug

        # 3) slug from folder
        if folder:
            s = extract_slug_from_path(folder)
            if s:
                p = lookup(s)
                if p:
                    slug = os.path.splitext(os.path.basename(p))[0]
                    return p, slug

        return None

    def _get_split(self, folder: str) -> str:
        m = re.search(r"(train|dev|test)", folder)
        if m: return m.group(1)
        return {"val": "dev"}.get(self.split_req, self.split_req)

    def _resolve_frames_dir(self, split: str, slug_with_default: str, folder: str) -> Optional[str]:
        frames_dir = os.path.join(self.image_prefix, split, slug_with_default, "1")
        if os.path.isdir(frames_dir) and (glob.glob(os.path.join(frames_dir, "*.png")) or glob.glob(os.path.join(frames_dir, "*.jpg"))):
            return frames_dir

        folder_clean = clean_folder_pattern(folder)
        if folder_clean:
            if os.path.isabs(folder_clean):
                if os.path.isdir(folder_clean):
                    return folder_clean
            rel = os.path.join(self.image_prefix, folder_clean)
            if os.path.isdir(rel):
                return rel

        base = slug_base(slug_with_default)
        rel2 = os.path.join(self.image_prefix, split, base, "1")
        if os.path.isdir(rel2):
            return rel2
        return None

    def _resolve_kp_key(self, split: str, slug_with_default: str, fileid: str) -> Optional[str]:
        k = self.kp_index.get((split, slug_with_default))
        if k: return k
        k = self.kp_index.get((split, slug_base(slug_with_default)))
        if k: return k
        k = self.kp_index.get(({"val":"dev"}.get(self.split_req, self.split_req), slug_with_default))
        if k: return k
        if fileid:
            k = self.kp_index.get((split, fileid)) or self.kp_index.get((split, slug_base(fileid)))
            if k: return k
        sb = slug_base(slug_with_default)
        for (sp, slug_k), key in self.kp_index.items():
            if sp == split and (slug_k == sb or slug_k.endswith("/"+sb)):
                return key
        return None

    def _materialize_rows(self, raw_samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        dropped = {"no_qgrid":0, "no_frames":0, "no_keypoints":0}
        for it in raw_samples:
            fileid = it.get("fileid","")
            folder = it.get("folder","")
            label  = it.get("label","")

            q = self._resolve_qgrid_path(fileid, folder)
            if q is None:
                dropped["no_qgrid"] += 1
                continue
            qgrid_path, slug_wd = q
            split = self._get_split(folder)
            frames_dir = self._resolve_frames_dir(split, slug_wd, folder)
            if frames_dir is None:
                dropped["no_frames"] += 1
                continue
            kp_key = self._resolve_kp_key(split, slug_wd, fileid)
            if kp_key is None:
                dropped["no_keypoints"] += 1
                continue

            rows.append({
                "fileid": fileid,
                "slug": slug_wd,
                "split": split,
                "frames_dir": frames_dir,
                "qgrid_path": qgrid_path,
                "kp_key": kp_key,
                "label": label,
            })

        kept = len(rows)
        total = len(raw_samples)
        print(f"[MultiModalPhoenixDataset] kept {kept}/{total} ({kept/total*100:.1f}%), "
              f"dropped: {dropped}")
        return rows

        # -------- keypoints coercion: dict/list/torch → np.ndarray (T,242) --------
    @staticmethod
    def _np_from_any(x: Any) -> Optional[np.ndarray]:
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, list):
            try:
                return np.array(x)
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_kp_Tx242(kp_raw: Any) -> np.ndarray:
        """
        Returns np.ndarray of shape (T,242), selecting only (x,y) if a score channel exists.
        Accepts:
          - (T,121,2)               → flatten
          - (T,2,121)               → transpose + flatten
          - (T,121,3) / (T,3,121)   → drop last dim to (x,y) then flatten
          - anything with prod(...) == 242 or 363 (121*3): handle accordingly
          - dict/list/torch tensor containers (extract first ndarray-like payload)
        """
        # 1) pull a numpy array out of dict/list/tensor/array
        arr = MultiModalPhoenixDataset._np_from_any(kp_raw)
        if arr is None and isinstance(kp_raw, dict):
            # try common field names first
            for key in ("keypoints", "kp", "points", "data", "arr", "array", "pose", "pose_2d", "features", "feat"):
                if key in kp_raw:
                    arr = MultiModalPhoenixDataset._np_from_any(kp_raw[key])
                    if arr is not None:
                        break
            # else: first ndarray among values
            if arr is None:
                for v in kp_raw.values():
                    arr = MultiModalPhoenixDataset._np_from_any(v)
                    if arr is not None:
                        break
            # else: try stacking values
            if arr is None:
                try:
                    from operator import itemgetter
                    items = sorted(kp_raw.items(), key=lambda kv: kv[0])
                    mats = [MultiModalPhoenixDataset._np_from_any(v) for _, v in items]
                    mats = [m for m in mats if m is not None]
                    if mats:
                        arr = np.stack(mats, axis=0)
                except Exception:
                    arr = None

        if arr is None:
            raise TypeError(f"Could not extract numpy array from keypoints payload of type {type(kp_raw)}")

        # 2) normalize to (T,242), using only x,y if a third channel is present
        T = arr.shape[0] if arr.ndim >= 1 else 0

        # exact matches
        if arr.ndim == 2 and arr.shape[1] == 242:
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[1:] == (121, 2):
            return arr.reshape(T, 242).astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[1:] == (2, 121):
            return np.transpose(arr, (0, 2, 1)).reshape(T, 242).astype(np.float32, copy=False)

        # with score channel (drop it)
        if arr.ndim == 3 and arr.shape[1:] == (121, 3):
            arr2 = arr[:, :, :2]
            return arr2.reshape(T, 242).astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[1:] == (3, 121):
            arr2 = arr[:, :2, :]
            return np.transpose(arr2, (0, 2, 1)).reshape(T, 242).astype(np.float32, copy=False)

        # generic: coords-last dimension of size 2 or 3 with 121 points somewhere
        if arr.ndim >= 3 and arr.shape[-1] in (2, 3):
            # reshape to (T, Npoints, C)
            a = arr.reshape(T, -1, arr.shape[-1])
            if a.shape[1] == 121:
                a = a[:, :, :2]  # keep x,y
                return a.reshape(T, 242).astype(np.float32, copy=False)

        # fallback by total product: 121*2=242; 121*3=363 (drop the third channel)
        prod = int(np.prod(arr.shape[1:])) if arr.ndim >= 2 else 0
        if prod == 242:
            return arr.reshape(T, 242).astype(np.float32, copy=False)
        if prod == 363:
            a = arr.reshape(T, 121, 3)[:, :, :2]
            return a.reshape(T, 242).astype(np.float32, copy=False)

        raise ValueError(f"Keypoints array has unsupported shape {arr.shape}; cannot coerce to (T,242)")


    def __len__(self):
        return len(self.rows)

    def _load_frames(self, frames_dir: str) -> List[Image.Image]:
        pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        jpgs = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        files = pngs if len(pngs) > 0 else jpgs
        return [Image.open(p).convert("RGB") for p in files]

    def _encode_labels(self, label_str: str) -> torch.LongTensor:
        ids = []
        for g in label_str.strip().split():
            if g in self.gloss_dict:
                ids.append(int(self.gloss_dict[g]))
        if len(ids) == 0:
            ids = [0]  # avoid empty target for CTC
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        # images
        frames = self._load_frames(r["frames_dir"])
        images = self.transforms(frames)  # (3,T_img,H,W)

        # qgrid (T,121,2) -> (T,242)
        qgrid_np = np.load(r["qgrid_path"])
        if qgrid_np.ndim != 3 or qgrid_np.shape[1:] != (121, 2):
            T = qgrid_np.shape[0]
            if int(np.prod(qgrid_np.shape[1:])) != 242:
                raise ValueError(f"Unexpected Qgrid shape {qgrid_np.shape} for '{r['qgrid_path']}'")
            qgrid_np = qgrid_np.reshape(T, 121, 2)
        Tq = qgrid_np.shape[0]
        qgrid = torch.from_numpy(qgrid_np.astype(np.float32)).reshape(Tq, 242)  # (T,242)

        # keypoints robust extraction → (T_img,242)
        kp_raw = self.kp_db[r["kp_key"]]
        kp_np = self._coerce_kp_Tx242(kp_raw)
        kp = torch.from_numpy(kp_np.astype(np.float32))

        # labels
        labels = self._encode_labels(r["label"])
        return images, qgrid, kp, labels

# ===================== Collate: pad time =====================
def multi_modal_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return (torch.tensor([]),) * 7

    images, qgrids, keypoints, labels = zip(*batch)

    # images: (C,T,H,W) -> (T,C,H,W) so we can pad over T
    imgs_TCHW = [im.permute(1, 0, 2, 3) if im.dim() == 4 else im for im in images]  # (T,C,H,W)

    image_lengths = torch.LongTensor([x.shape[0] for x in imgs_TCHW])  # T_img per sample
    qgrid_lengths = torch.LongTensor([q.shape[0] for q in qgrids])     # T_q per sample
    label_lengths = torch.LongTensor([len(l) for l in labels])

    padded_images    = torch.nn.utils.rnn.pad_sequence(imgs_TCHW, batch_first=True, padding_value=0.0)  # (B,T,C,H,W)
    padded_qgrids    = torch.nn.utils.rnn.pad_sequence(qgrids,   batch_first=True, padding_value=0.0)  # (B,Tq,242)
    padded_keypoints = torch.nn.utils.rnn.pad_sequence(keypoints,batch_first=True, padding_value=0.0)  # (B,T,242)
    padded_labels    = torch.nn.utils.rnn.pad_sequence(labels,   batch_first=True, padding_value=0)    # (B,L)

    return padded_images, padded_qgrids, padded_keypoints, padded_labels, image_lengths, label_lengths, qgrid_lengths


















# # slr/datasets/multi_modal_datasets.py

# import os
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import json
# from PIL import Image
# import csv # Using csv for the metadata files
# import pickle # <--- ADD THIS LINE

# class MultiModalPhoenixDataset(Dataset):
#     # [MODIFIED] --- Added 'split' to the constructor ---
#     def __init__(self, image_prefix, qgrid_prefix, kp_path, meta_dir, gloss_dict_path, split='train', transforms=None):
#         self.image_prefix = image_prefix
#         self.qgrid_prefix = os.path.join(qgrid_prefix, split) # Path to the correct split folder
#         self.transforms = transforms
#         self.split = split

#         # Load gloss dictionary
#         self.gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        
#         # [MODIFIED] --- Load keypoints differently ---
#         # The keypoints .pkl file likely contains all splits, so we load the whole thing
#         all_keypoints_data = pickle.load(open(kp_path, 'rb'))
        
#         # Load metadata and build the list of samples for the correct split
#         # This now correctly handles the .csv format
#         meta_path = os.path.join(meta_dir, f'{split}.corpus.csv')
#         self.samples = []
#         self.keypoints_data = {}
#         with open(meta_path, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f, delimiter='|')
#             next(reader, None) # Skip header
#             for row in reader:
#                 video_id = row[0]
#                 self.samples.append({
#                     'id': video_id,
#                     'folder': os.path.join(split, video_id), # e.g., train/video-id
#                     'gloss': row[3] # Assuming gloss is the 4th column
#                 })
#                 # Find the matching keypoint data
#                 for complex_key, data in all_keypoints_data.items():
#                     if video_id in complex_key:
#                         self.keypoints_data[video_id] = data
#                         break
#         # ---

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         sample_info = self.samples[index]
#         video_id = sample_info['id']

#         # 1. [MODIFIED] --- Correctly construct the image folder path ---
#         image_folder = os.path.join(self.image_prefix, sample_info['folder'])
#         # ---
        
#         image_files = sorted(os.listdir(image_folder))
#         images = []
#         for img_file in image_files:
#             # Using try-except to handle potentially corrupted images
#             try:
#                 img = Image.open(os.path.join(image_folder, img_file)).convert('RGB')
#                 images.append(img)
#             except Exception as e:
#                 print(f"Warning: Could not load image {os.path.join(image_folder, img_file)}. Skipping. Error: {e}")
#                 continue
        
#         if not images:
#             # Handle case where a folder has no valid images
#             # Return a dummy sample or raise an error
#             return self.__getitem__((index + 1) % len(self))

#         if self.transforms:
#             images = self.transforms(images)

#         # 2. Load Qgrid data
#         qgrid_path = os.path.join(self.qgrid_prefix, f"{video_id}.npy")
#         qgrid = torch.from_numpy(np.load(qgrid_path)).float()

#         # 3. Load Keypoints
#         keypoints = torch.from_numpy(self.keypoints_data[video_id]).float()
#         keypoints = keypoints.reshape(keypoints.size(0), -1)

#         # 4. Load Labels
#         gloss_sequence = sample_info['gloss'].split()
#         labels = torch.LongTensor([self.gloss_dict.get(g, self.gloss_dict['<unk>']) for g in gloss_sequence])

#         return images, qgrid, keypoints, labels

# def multi_modal_collate_fn(batch):
#     # Separate the modalities
#     images, qgrids, keypoints, labels = zip(*batch)

#     # Pad sequences to the max length in the batch for each modality
#     # Note: `pad_sequence` expects (T, *) shape, so we use batch_first=True
#     padded_images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True, padding_value=0.0)
#     padded_qgrids = torch.nn.utils.rnn.pad_sequence(qgrids, batch_first=True, padding_value=0.0)
#     padded_keypoints = torch.nn.utils.rnn.pad_sequence(keypoints, batch_first=True, padding_value=0.0)
#     padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

#     # Get the original lengths for CTC loss
#     image_lengths = torch.LongTensor([len(img) for img in images])
#     label_lengths = torch.LongTensor([len(lbl) for lbl in labels])

#     return padded_images, padded_qgrids, padded_keypoints, padded_labels, image_lengths, label_lengths