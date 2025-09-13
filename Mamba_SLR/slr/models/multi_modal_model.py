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
        
        self.samples = [v for k, v in metadata.items() if isinstance(v, dict)]

        self.keypoints_data = {}
        for sample in self.samples:
            video_id = sample['fileid']
            for complex_key, data in all_keypoints_data.items():
                if video_id in complex_key:
                    self.keypoints_data[video_id] = data
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        video_id = sample_info['fileid']

        # --- THIS IS THE FINAL, CORRECTED PART ---
        # The 'folder' key contains a file pattern like ".../1/*.png".
        # We must remove the "/*.png" to get the actual directory path.
        folder_path_from_meta = sample_info['folder'].replace('/*.png', '')
        image_folder = os.path.join(self.image_prefix, folder_path_from_meta)
        # --- END OF CORRECTION ---
        
        images = []
        try:
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