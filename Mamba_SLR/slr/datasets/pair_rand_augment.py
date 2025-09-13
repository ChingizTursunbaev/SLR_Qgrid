import numpy as np
import random
from PIL import Image

try:
    from .rand_augment import rand_augment_transform, NAME_TO_OP
except:
    from rand_augment import rand_augment_transform, NAME_TO_OP


_GEOM_NAMES = {
    "Rotate", "ShearX", "ShearY",
    "TranslateX", "TranslateY",
    "TranslateXRel", "TranslateYRel",
}
_GEOM_FUNCS = {NAME_TO_OP[n] for n in _GEOM_NAMES if n in NAME_TO_OP}

def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


def _to_rgb(imgs):
    return [im.convert("RGB") for im in imgs]

def _to_l(imgs):
    return [im.convert("L") for im in imgs]

def _apply_op_to_list(img_list, fn, level_args, kwargs):
    return [fn(img, *level_args, **kwargs) for img in img_list]

class PairedRandAug:
    def __init__(self, config_str, img_size, interpolation):
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(translate_const=int(img_size_min * 0.2), interpolation=_pil_interp(interpolation))
        self.ra = rand_augment_transform(config_str, aa_params)

    def __call__(self, frames_pil, heatmaps_pil):
        heatmaps_rgb = None
        if heatmaps_pil is not None:
            heatmaps_rgb = _to_rgb(heatmaps_pil)

        ops = np.random.choice(
            self.ra.ops,
            self.ra.num_layers,
            replace=self.ra.choice_weights is None,
            p=self.ra.choice_weights,
        )

        out_frames = frames_pil
        out_heatmaps_rgb = heatmaps_rgb

        for op in ops:
            magnitude = op.magnitude
            if getattr(op, "magnitude_std", 0) > 0:
                magnitude = random.gauss(magnitude, op.magnitude_std)

            if op.level_fn is not None:
                level_args = op.level_fn(magnitude, op.hparams)
            else:
                level_args = ()

            kwargs = op.kwargs.copy()

            out_frames = _apply_op_to_list(out_frames, op.aug_fn, level_args, kwargs)
            if out_heatmaps_rgb is not None and op.aug_fn in _GEOM_FUNCS:
                out_heatmaps_rgb = _apply_op_to_list(out_heatmaps_rgb, op.aug_fn, level_args, kwargs)

        if out_heatmaps_rgb is not None:
            out_heatmaps = _to_l(out_heatmaps_rgb)
        else:
            out_heatmaps = None

        return out_frames, out_heatmaps

def paired_random_augment(config_str: str, img_size: int, interpolation: str = "bilinear"):
    return PairedRandAug(config_str, img_size, interpolation)