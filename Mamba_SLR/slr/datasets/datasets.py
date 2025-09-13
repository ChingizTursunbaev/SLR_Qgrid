import os
import pickle
from functools import lru_cache
from glob import glob

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import functional as FF
except:
    import datasets.functional as FF
try:
    from .video_transforms import (
        Compose, Resize, CenterCrop, Normalize,
        create_random_augment, random_short_side_scale_jitter, 
        random_crop, random_resized_crop_with_shift, random_resized_crop,
        horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
    )
    from .volume_transforms import ClipToTensor
    from .pair_rand_augment import paired_random_augment
    from .pair_random_erasing import PairRandomErasing
except:
    from video_transforms import (
        Compose, Resize, CenterCrop, Normalize,
        create_random_augment, random_short_side_scale_jitter, 
        random_crop, random_resized_crop_with_shift, random_resized_crop,
        horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
    )
    from volume_transforms import ClipToTensor
    from pair_rand_augment import paired_random_augment
    from pair_random_erasing import PairRandomErasing


def _make_gaussian_kernel(kernel_size: int = 21, sigma: float = 6.):
    k  = cv2.getGaussianKernel(kernel_size, sigma)
    g  = (k @ k.T)
    g /= g.max()
    return g.astype(np.float32)

@lru_cache(maxsize=8)
def _kp_gaussian(kernel_size=21, sigma=6.):
    return _make_gaussian_kernel(kernel_size, sigma)

def keypoints_to_heatmap(
    kps: np.ndarray,
    out_size=(256, 256),
    kernel_size=21,
    sigma=6.,
    thr=0.,
    groups=('face', 'mouth', 'hands')
):
    H, W = out_size
    hmap = np.zeros((H, W), dtype=np.float32)

    IDX = {
        'body': list(range(0, 11)),
        'face': list(range(11, 59)),
        'mouth': list(range(59, 79)),
        'hands': list(range(79, 121)),
    }

    ker = _kp_gaussian(kernel_size, sigma)
    k2  = kernel_size // 2

    for grp in groups:
        for i in IDX[grp]:
            x, y, c = kps[i]
            if c < thr:
                continue
            x, y = int(round(x)), int(round(y))

            y0, y1 = max(0, y-k2), min(H, y+k2+1)
            x0, x1 = max(0, x-k2), min(W, x+k2+1)
            ky0, ky1 = k2-(y-y0), k2+(y1-y)
            kx0, kx1 = k2-(x-x0), k2+(x1-x)

            if y1 <= y0 or x1 <= x0 or ky1 <= ky0 or kx1 <= kx0:
                continue

            hmap[y0:y1, x0:x1] = np.maximum(
                hmap[y0:y1, x0:x1],
                ker[ky0:ky1, kx0:kx1]
            )
    return hmap

class Phoenix2014(Dataset):
    def __init__(
        self,
        prefix: str,
        gloss_dict: dict,
        input_size: int = 256,
        crop_size: int = 224,
        mode: str = "train",
        meta_dir_path: str = "",
        frame_interval: int = 1,
        temp_scale: float = 0.,
        use_heatmap: bool = False,
        kp_path: str = "",
        kernel_size: int = 193,
        sigma: float = 32.,
        args=None
    ):
        super().__init__()
        self.mode = mode
        self.prefix = prefix
        self.gloss_dict = gloss_dict
        self.input_size = input_size
        self.crop_size = crop_size
        self.frame_interval = frame_interval
        self.temp_scale = temp_scale
        self.use_heatmap = use_heatmap
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.args = args
        self.aa = args.aa
        self.rand_erase = False

        if self.mode == 'train':
            if self.args.reprob > 0:
                self.rand_erase = True

        self.inputs_list = np.load(os.path.join(meta_dir_path, f"{mode}_info.npy"), allow_pickle=True).item()

        if self.use_heatmap:
            with open(kp_path, "rb") as f:
                self.keypoints = pickle.load(f)

        if mode == 'train':
            pass
        elif mode in ('dev', 'test'):
            self.frame_transform = Compose([
                Resize(self.input_size, interpolation='bicubic'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[.45, .45, .45],std=[.225, .225, .225])
            ])
            self.heatmap_transform = Compose([
                Resize(self.input_size, interpolation='bicubic'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(channel_nb=1),
            ])
        else:
            raise ValueError
            
        print(f"[Phoenix-2014] {mode} split: {len(self)} samples")

    def __getitem__(self, idx: int):
        offset = int(torch.randint(0, self.frame_interval, (1,)))
        frames, labels, fi = self.read_video(idx, offset)

        if self.use_heatmap:
            kp_seq = self.load_kps(fi, offset) # (T, 121, 3)
            heatmaps = self.kps_to_heatmaps(kp_seq) # (T, H, W)
            heatmaps = [
                Image.fromarray((hmap.numpy() * 255).astype(np.uint8), mode='L')
                for hmap in heatmaps
            ]
        else:
            heatmaps = None
        
        if self.mode == 'train':
            frames, heatmaps = self._augment(frames, heatmaps, self.args)
        else:
            frames = self.frame_transform(frames)
            frames = frames.transpose(0, 1)
            if heatmaps is not None:
                heatmaps = self.heatmap_transform(heatmaps)
                heatmaps = heatmaps.transpose(0, 1)
        
        return frames, heatmaps, torch.LongTensor(labels), fi['label'], len(labels)
    
    def _augment(self, frames, heatmaps, args):
        aug_transform = paired_random_augment(
            config_str=self.aa,
            img_size=self.input_size,
            interpolation='bicubic'
        )
        frames, heatmaps = aug_transform(frames, heatmaps)
        frames = torch.stack([transforms.ToTensor()(frame) for frame in frames]) # (T, C, H, W)
        if heatmaps is not None:
            heatmaps = torch.stack([transforms.ToTensor()(heatmap) for heatmap in heatmaps]) # (T, 1, H, W)
        frames = frames.permute(0, 2, 3, 1) # (T, H, W, C)

        frames = tensor_normalize(frames, mean=[.45, .45, .45], std=[.225, .225, .225])
        frames = frames.permute(3, 0, 1, 2) # (C, T, H, W)

        buffer = torch.cat([frames, heatmaps.transpose(0, 1)], dim=0) if heatmaps is not None else frames
        scl, asp = (
            [0.8, 1.0],
            [0.75, 1.3333]
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=288,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        buffer = buffer.transpose(0, 1)
        if heatmaps is not None:
            frames, heatmaps = buffer[:, :3, :, :], buffer[:, -1, :, :]
        else:
            frames = buffer

        if self.rand_erase:
            erase_transform = PairRandomErasing(
                probability=args.reprob,
                max_count=args.recount,
                mode=args.remode,
                device="cpu",
                cube=True,
            )
            
            frames, heatmaps = erase_transform(frames.transpose(0, 1), heatmaps)
            frames = frames.transpose(0, 1)
            frames, heatmaps = erase_transform(frames, heatmaps)
        
        return frames, heatmaps
    
    def read_video(self, idx: int, offset: int):
        fi = self.inputs_list[idx]
        img_dir = os.path.join(
            self.prefix,
            "features/fullFrame-256x256px",
            fi["folder"].rstrip("/*.png")
        )
        img_paths = sorted(glob(f"{img_dir}/*.png"))
        img_paths = img_paths[offset::self.frame_interval]

        frames = [
            Image.fromarray(
                cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            ) for p in img_paths
        ]
        labels = [
            self.gloss_dict[gloss][0] for gloss in fi["label"].split()
            if gloss in self.gloss_dict
        ]
        return frames, labels, fi
    
    def kp_key(self, fi):
        folder = fi["folder"].split("/*.")[0]
        vid_no = fi["fileid"].split("_default-")[0]
        return f"fullFrame-210x260px/{folder}/{vid_no}"

    def load_kps(self, fi, offset):
        k = self.kp_key(fi)
        if k not in self.keypoints:
            raise KeyError(f"Cannot find the key: {k}")
        kp_seq = self.keypoints[k]["keypoints"]
        return kp_seq[offset::self.frame_interval]

    def kps_to_heatmaps(self, kp_seq: np.ndarray):
        hm = np.stack([
            keypoints_to_heatmap(
                k,
                out_size=(256, 256),
                kernel_size=self.kernel_size,
                sigma=self.sigma
            ) for k in kp_seq
        ], axis=0)
        return torch.from_numpy(hm)

    def __len__(self):
        return len(self.inputs_list) - 1

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def collate_fn(batch):
    videos, hmaps, labels, glosses, label_lengths = zip(*batch)
    video_lengths = [v.shape[0] for v in videos]
    videos = torch.cat(videos, dim=0) # (BT_sum, C, H ,W)
    frame_ids = torch.repeat_interleave(torch.arange(len(video_lengths)), torch.tensor(video_lengths)).to(torch.int32)

    if hmaps[0] is None:
        hmaps = None
    else:
        hmaps = torch.cat(hmaps, dim=0) # (BT_sum, H, W)
    
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.int32)
    
    return videos, hmaps, video_lengths, labels, glosses, label_lengths, frame_ids

def blockify_heatmap(hmap_224: torch.Tensor, cell: int = 16) -> torch.Tensor:
    """
    (224,224) 히트맵을 14×14로 평균 풀링했다가 최근접 업샘플.
    블록당 값이 일정해진 (224,224) 히트맵 반환.
    """
    if hmap_224.dim() == 2:
        hmap_224 = hmap_224.unsqueeze(0)  # (1,H,W)

    coarse = F.avg_pool2d(
        hmap_224.unsqueeze(0),           # (N=1,C=1,H,W)
        kernel_size=cell, stride=cell
    )
    block = F.interpolate(coarse, scale_factor=cell, mode="nearest")
    return block.squeeze()               # (224,224)

def save_block_overlay_grid(
    frames_224, heatmaps_224=None, path=None, alpha: float = 0.4, nrow: int = 4, cell: int = 16
):
    """
    frames_224   : (T,3,224,224)  값 범위 0~1
    heatmaps_224 : None | (T,224,224) | (T,1,224,224)  값 범위 0~1
    """
    import torchvision

    # frames를 리스트[(3,H,W), ...]로 통일
    if isinstance(frames_224, torch.Tensor):
        frame_list = [f.detach().cpu() for f in frames_224]     # 각 f: (3,224,224)
    else:
        frame_list = [torch.as_tensor(f).detach().cpu() for f in frames_224]

    # 히트맵이 없으면 프레임만 그리드로 저장
    if heatmaps_224 is None:
        grid = torchvision.utils.make_grid(frame_list, nrow=nrow)
        torchvision.utils.save_image(grid, path)
        print(f"시각화 결과(프레임 전용): {path}")
        return

    # heatmaps를 (T,224,224)로 정규화
    if isinstance(heatmaps_224, torch.Tensor):
        h = heatmaps_224.detach().cpu()
    else:
        h = torch.as_tensor(heatmaps_224)

    if h.dim() == 4 and h.size(1) == 1:   # (T,1,H,W) -> (T,H,W)
        h = h.squeeze(1)
    assert h.dim() == 3, "heatmaps_224는 (T,224,224) 또는 (T,1,224,224)이어야 합니다."

    overlays = []
    for f, hmap in zip(frame_list, h):
        # 블록화
        h_block = blockify_heatmap(hmap, cell=cell).clamp(0, 1)

        # 컬러맵 적용
        h_color = cv2.applyColorMap(
            (h_block.numpy() * 255).astype("uint8"),
            cv2.COLORMAP_JET
        )
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        h_color = torch.from_numpy(h_color).permute(2, 0, 1) / 255.0

        over = ((1 - alpha) * f + alpha * h_color).clamp_(0, 1)
        overlays.append(over)

    grid = torchvision.utils.make_grid(overlays, nrow=nrow)
    torchvision.utils.save_image(grid, path)
    print(f"시각화 결과(블록 오버레이): {path}")


def save_overlay_grid(
    frames_224, heatmaps_224=None, path=None, alpha: float = 0.4, nrow: int = 4
):
    """
    frames_224   : (T,3,224,224)  값 범위 0~1
    heatmaps_224 : None | (T,224,224) | (T,1,224,224)  값 범위 0~1
    """
    import torchvision

    # frames를 리스트[(3,H,W), ...]로 통일
    if isinstance(frames_224, torch.Tensor):
        frame_list = [f.detach().cpu() for f in frames_224]
    else:
        frame_list = [torch.as_tensor(f).detach().cpu() for f in frames_224]

    # 히트맵이 없으면 프레임만 그리드로 저장
    if heatmaps_224 is None:
        grid = torchvision.utils.make_grid(frame_list, nrow=nrow)
        torchvision.utils.save_image(grid, path)
        print(f"시각화 결과(프레임 전용): {path}")
        return

    # heatmaps를 (T,224,224)로 정규화
    if isinstance(heatmaps_224, torch.Tensor):
        h = heatmaps_224.detach().cpu()
    else:
        h = torch.as_tensor(heatmaps_224)

    if h.dim() == 4 and h.size(1) == 1:   # (T,1,H,W) -> (T,H,W)
        h = h.squeeze(1)
    assert h.dim() == 3, "heatmaps_224는 (T,224,224) 또는 (T,1,224,224)이어야 합니다."

    overlays = []
    for f, hmap in zip(frame_list, h):
        # 1) 히트맵 정규화
        h_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-6)
        h_norm = h_norm.clamp(0, 1)

        # 2) 컬러맵 적용
        h_color = cv2.applyColorMap(
            (h_norm.numpy() * 255).astype("uint8"),
            cv2.COLORMAP_JET
        )
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        h_color = torch.from_numpy(h_color).permute(2, 0, 1) / 255.0

        # 3) 합성
        over = ((1 - alpha) * f + alpha * h_color).clamp_(0, 1)
        overlays.append(over)

    grid = torchvision.utils.make_grid(overlays, nrow=nrow)
    torchvision.utils.save_image(grid, path)
    print(f"시각화 결과(오버레이): {path}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--aa', type=str, default='rand-m5-mstd0.25', metavar='NAME')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    prefix = "/mnt/data/phoenix-2014/phoenix-2014-multisigner"
    gloss_dict = np.load(
        "/home/kks/workspace/slr/data/phoenix2014/gloss_dict.npy",
        allow_pickle=True
    ).item()
    meta_dir_path = '/home/kks/workspace/slr/data/phoenix2014'

    ds = Phoenix2014(
        prefix         = prefix,
        gloss_dict     = gloss_dict,
        input_size     = 256,
        crop_size     = 224,
        mode           = "train",
        meta_dir_path=meta_dir_path,
        frame_interval = 1,
        temp_scale     = 0.,
        use_heatmap    = True,
        kernel_size    = 193,
        sigma          = 36.,
        kp_path    = '/mnt/data/phoenix-2014/phoenix-2014-keypoints.pkl',
        args=parser.parse_known_args()[0]
    )

    loader = DataLoader(
        ds,
        batch_size  = 2,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = collate_fn
    )

    save_root = "/home/kks/workspace/slr/slr/datasets"
    os.makedirs(save_root, exist_ok=True)

    mean = torch.tensor([.45, .45, .45]).view(1, 3, 1, 1)
    std  = torch.tensor([.225, .225, .225]).view(1, 3, 1, 1)

    for vids, hmaps, video_lengths, labels, glosses, label_lengths, frame_ids in loader:
        print("vids:", vids.shape, vids.min().item(), vids.max().item())
        if hmaps is None:
            print("hmaps: None")
        else:
            print("hmaps:", hmaps.shape, hmaps.min().item(), hmaps.max().item())

        print("video_lengths:", video_lengths)
        print("labels:", labels)
        print("glosses:", glosses)
        print("label_lengths:", label_lengths)
        print("frame_ids:", frame_ids)

        vids_seq = torch.split(vids, list(video_lengths), dim=0)  # List[(T,3,H,W)]
        if hmaps is None:
            hmaps_seq = None
        else:
            hmaps_seq = torch.split(hmaps, list(video_lengths), dim=0)  # List[(T,H,W)]

        vids_vis = (vids_seq[0] * std + mean)  # 0~1 복원
        if hmaps_seq is None:
            hms_vis = None
        else:
            hms_vis = hmaps_seq[0].float()

        save_block_overlay_grid(
            vids_vis, hms_vis,
            path=os.path.join(save_root, "overlay_block.png"),
            nrow=4, alpha=1.0, cell=16
        )
        save_overlay_grid(
            vids_vis,              # (T,3,224,224) 0~1
            hms_vis,               # None 또는 (T,224,224) 0~1
            path=os.path.join(save_root, "overlay.png"),
            alpha=0.45,
            nrow=4
        )
        break

