# import numpy as np

# # Path to your training metadata file
# file_path = 'data/phoenix2014/train_info.npy' # Assuming you run from the Mamba_SLR root

# print(f"--- Deeply analyzing file: {file_path} ---")

# try:
#     # Load the .npy file and extract the dictionary
#     data = np.load(file_path, allow_pickle=True).item()
#     print(f"Successfully loaded data. It is a {type(data)}.")

#     valid_samples = []
#     invalid_items = []

#     # Iterate through all items in the dictionary
#     for key, value in data.items():
#         # Check if the value is a dictionary, which is what we expect for a sample
#         if isinstance(value, dict):
#             # Check if this dictionary has the keys we need ('id', 'folder', 'gloss')
#             if 'id' in value and 'folder' in value and 'gloss' in value:
#                 valid_samples.append(value)
#             else:
#                 # It's a dictionary, but missing essential keys
#                 invalid_items.append({key: value})
#         else:
#             # It's not a dictionary (like the 'prefix' string)
#             invalid_items.append({key: value})

#     print("\n--- Analysis Results ---")
#     print(f"Found {len(valid_samples)} valid-looking samples.")
#     print(f"Found {len(invalid_items)} other items that are NOT valid samples.")

#     if valid_samples:
#         print("\n--- Example of a VALID sample ---")
#         print(valid_samples[0])

#     if invalid_items:
#         print("\n--- Items that are NOT valid samples ---")
#         # Print the first 5 problematic items for review
#         for item in invalid_items[:5]:
#             print(item)

# except Exception as e:
#     print(f"\n--- An error occurred ---")
#     print(e)



# ---------------------------------------

# tiny_debug.py
import os
import torch
from torch.utils.data import DataLoader
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba

# --------- paths (use the ones you already confirmed) ----------
IMAGE_PREFIX = "/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px"
QGRID_PREFIX = "/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy"
KP_PATH      = "/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"
META_DIR     = "data/phoenix2014"
GLOSS_DICT   = "data/phoenix2014/gloss_dict_normalized.npy"

# --------- device ----------
assert torch.cuda.is_available(), "CUDA GPU is required for Triton fused LayerNorm in mamba_ssm."
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# --------- dataset / loader ----------
ds = MultiModalPhoenixDataset(
    image_prefix=IMAGE_PREFIX,
    qgrid_prefix=QGRID_PREFIX,
    kp_path=KP_PATH,
    meta_dir_path=META_DIR,
    gloss_dict_path=GLOSS_DICT,
    split="train",
    transforms=None
)
dl = DataLoader(
    ds,
    batch_size=2,
    shuffle=False,
    collate_fn=multi_modal_collate_fn,
    num_workers=2,
    pin_memory=True
)

# --------- model cfgs (example; adjust to your Model signature) ----------
img_cfg   = {"out_dim": 512}  # your slr/models/model.py will map this appropriately
qgrid_cfg = {"out_dim": 512}
kp_cfg    = {"input_dim": 242, "model_dim": 512}
fusion    = {"embed_dim": 512, "num_heads": 8, "dropout": 0.1}

# num_classes from gloss dict
num_classes = len(ds.gloss_dict) + 1  # +1 if your CTC blank is 0 and gloss ids start at 1

model = MultiModalMamba(img_cfg, qgrid_cfg, kp_cfg, num_classes=num_classes, fusion_cfg=fusion).to(device)
model.eval().to(device)

# --------- pull one batch and move to GPU ----------
batch = next(iter(dl))
images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch

print("images:", images.shape)
print("qgrids:", qgrids.shape)
print("keypoints:", keypoints.shape)
print("lens(img,qgrid):", image_lengths.tolist(), qgrid_lengths.tolist())

images        = images.to(device, non_blocking=True)
qgrids        = qgrids.to(device, non_blocking=True)
keypoints     = keypoints.to(device, non_blocking=True)
labels        = labels.to(device, non_blocking=True)
image_lengths = image_lengths.to(device, non_blocking=True)
label_lengths = label_lengths.to(device, non_blocking=True)
qgrid_lengths = qgrid_lengths.to(device, non_blocking=True)

with torch.no_grad():
    logits = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)  # (B, T_img, V)
print("OK logits:", logits.shape)

