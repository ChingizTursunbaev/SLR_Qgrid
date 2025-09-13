# slr/datasets/multi_modal_datasets.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pickle

class MultiModalPhoenixDataset(Dataset):
    def __init__(self, image_prefix, qgrid_prefix, kp_path, meta_dir_path, gloss_dict_path, split='train', transforms=None):
        self.image_prefix = image_prefix
        self.qgrid_prefix = os.path.join(qgrid_prefix, split)
        self.transforms = transforms
        self.split = split

        self.gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        self.unk_token_id = self.gloss_dict.get('<unk>', 1)
        
        all_keypoints_data = pickle.load(open(kp_path, 'rb'))

        info_file = os.path.join(meta_dir_path, f'{split}_info.npy')
        metadata = np.load(info_file, allow_pickle=True).item()
        
        # Filter the metadata to get only the sample dictionaries
        self.samples = [v for k, v in metadata.items() if isinstance(v, dict)]

        self.keypoints_data = {}
        for sample in self.samples:
            # --- THIS IS THE CORRECTED PART ---
            # Use the correct key 'fileid' which we found in the analysis
            video_id = sample['fileid']
            # --- END OF CORRECTION ---
            for complex_key, data in all_keypoints_data.items():
                if video_id in complex_key:
                    self.keypoints_data[video_id] = data
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        # --- THIS IS THE CORRECTED PART ---
        # Use the correct key 'fileid' here as well
        video_id = sample_info['fileid']
        # --- END OF CORRECTION ---

        base_folder = os.path.join(self.image_prefix, sample_info['folder'])
        
        images = []
        try:
            signer_subfolders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
            if not signer_subfolders:
                 return None

            image_folder = os.path.join(base_folder, sorted(signer_subfolders)[0])
            
            image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            for img_file in image_files:
                img_path = os.path.join(image_folder, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception:
                    continue
        
        except FileNotFoundError:
            return None
        
        if not images:
            return None

        if self.transforms:
            images = self.transforms(images)

        qgrid_path = os.path.join(self.qgrid_prefix, f"{video_id}.npy")
        qgrid = torch.from_numpy(np.load(qgrid_path)).float()

        if video_id not in self.keypoints_data:
            return None
        keypoints_array = self.keypoints_data[video_id]['keypoints']
        keypoints = torch.from_numpy(keypoints_array).float()
        
        keypoints = keypoints.reshape(keypoints.size(0), -1)

        # Use the 'label' key to get the gloss sequence
        gloss_sequence = sample_info['label'].split()
        labels = torch.LongTensor([self.gloss_dict.get(g, self.unk_token_id) for g in gloss_sequence])

        return images, qgrid, keypoints, labels

def multi_modal_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

    images, qgrids, keypoints, labels = zip(*batch)

    padded_images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True, padding_value=0.0)
    padded_qgrids = torch.nn.utils.rnn.pad_sequence(qgrids, batch_first=True, padding_value=0.0)
    padded_keypoints = torch.nn.utils.rnn.pad_sequence(keypoints, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    image_lengths = torch.LongTensor([len(img) for img in images])
    label_lengths = torch.LongTensor([len(lbl) for lbl in labels])

    return padded_images, padded_qgrids, padded_keypoints, padded_labels, image_lengths, label_lengths










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