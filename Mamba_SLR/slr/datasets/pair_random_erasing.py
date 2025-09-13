import math
import random
import torch

try:
    from .random_erasing import _get_pixels
except Exception:
    from random_erasing import _get_pixels


class PairRandomErasing:
    def __init__(
        self,
        probability: float = 0.25,
        min_area: float = 0.02,
        max_area: float = 0.1,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
        mode: str = "pixel",
        max_count: int = 1,
        device: str = "cpu",
        cube: bool = True,          
    ):
        self.p = probability
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect or (1.0 / min_aspect)
        self.mode = mode
        self.max_count = max_count
        self.device = device
        self.cube = cube

        self.rand_color = (self.mode == "rand")
        self.per_pixel = (self.mode == "pixel")

    @staticmethod
    def _as_TCHW(frames):
        """(T,C,H,W) or (C,T,H,W) -> (T,C,H,W), and return whether transposed."""
        assert frames.dim() == 4
        if frames.shape[0] in (1, 3):
            return frames.permute(1, 0, 2, 3), True
        return frames, False

    @staticmethod
    def _restore_from_TCHW(frames_TCHW, transposed):
        return frames_TCHW.permute(1, 0, 2, 3) if transposed else frames_TCHW

    @staticmethod
    def _as_THW(heatmaps):
        """(T,H,W) or (T,1,H,W) or None -> (T,H,W), had_channel flag"""
        if heatmaps is None:
            return None, False
        if heatmaps.dim() == 4:
            assert heatmaps.shape[1] == 1
            return heatmaps[:, 0], True
        assert heatmaps.dim() == 3
        return heatmaps, False

    @staticmethod
    def _restore_from_THW(heatmaps_THW, had_channel):
        if heatmaps_THW is None:
            return None
        return heatmaps_THW.unsqueeze(1) if had_channel else heatmaps_THW

    def _sample_boxes(self, H, W):
        """random_erasing.py와 동일한 방식으로 max_count개 박스를 샘플."""
        area = H * W
        boxes = []
        log_aspect = (math.log(self.min_aspect), math.log(self.max_aspect))
        count = random.randint(1, self.max_count)
        for _ in range(count):
            for _ in range(100):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect = math.exp(random.uniform(*log_aspect))
                h = int(round(math.sqrt(target_area * aspect)))
                w = int(round(math.sqrt(target_area / aspect)))
                if 0 < h < H and 0 < w < W:
                    top = random.randint(0, H - h)
                    left = random.randint(0, W - w)
                    boxes.append((top, left, h, w))
                    break
        return boxes

    def __call__(self, frames, heatmaps=None):
        if random.random() > self.p:
            return frames, heatmaps

        # frames to (T,C,H,W)
        frames_TCHW, transposed = self._as_TCHW(frames)
        T, C, H, W = frames_TCHW.shape
        device = frames_TCHW.device

        # heatmaps to (T,H,W)
        heatmaps_THW, had_channel = self._as_THW(heatmaps)

        if self.cube:
            boxes = self._sample_boxes(H, W)
            for (top, left, h, w) in boxes:
                fill_one = _get_pixels(
                    self.per_pixel, self.rand_color, (C, h, w),
                    dtype=frames_TCHW.dtype, device=device
                )
                if self.per_pixel:
                    # (T,C,h,w)
                    fill = fill_one.unsqueeze(0).expand(T, -1, -1, -1)
                elif self.rand_color:
                    # (C,h,w) -> (T,C,h,w)
                    fill = fill_one.expand(C, h, w).unsqueeze(0).expand(T, -1, -1, -1)
                else:  # const(=0)
                    fill = torch.zeros((T, C, h, w), dtype=frames_TCHW.dtype, device=device)

                frames_TCHW[:, :, top:top + h, left:left + w] = fill

                if heatmaps_THW is not None:
                    heatmaps_THW[:, top:top + h, left:left + w] = 0.0

        else:
            for t in range(T):
                boxes = self._sample_boxes(H, W)
                for (top, left, h, w) in boxes:
                    fill_one = _get_pixels(
                        self.per_pixel, self.rand_color, (C, h, w),
                        dtype=frames_TCHW.dtype, device=device
                    )
                    if self.per_pixel:
                        frames_TCHW[t, :, top:top + h, left:left + w] = fill_one
                    elif self.rand_color:
                        frames_TCHW[t, :, top:top + h, left:left + w] = fill_one.expand(C, h, w)
                    else:
                        frames_TCHW[t, :, top:top + h, left:left + w] = 0.0

                    if heatmaps_THW is not None:
                        heatmaps_THW[t, top:top + h, left:left + w] = 0.0

        frames_out = self._restore_from_TCHW(frames_TCHW, transposed)
        heatmaps_out = self._restore_from_THW(heatmaps_THW, had_channel)
        return frames_out, heatmaps_out
