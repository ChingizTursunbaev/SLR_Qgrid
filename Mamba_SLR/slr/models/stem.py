# slr/models/stem.py

import torch
import torch.nn as nn

class Stem(nn.Module):
    """
    A flexible Stem implementation for the Vision Transformer.
    This module performs patch embedding and works for any patch size.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # This single convolutional layer is the standard and flexible
        # way to perform patch embedding. It works for any patch size.
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            # You could also dynamically resize here if you prefer
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})."
            )
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, HW, C
        
        x = self.norm(x)
        return x

















# import torch
# import torch.nn as nn
# from timm.models.layers import to_2tuple, DropPath
# import timm


# class Block(nn.Module):
#     def __init__(self, dim, drop_path_rate, norm_cls, norm_epsilon=1e-5, **factory_kwargs):
#         super().__init__()
#         self.norm1 = norm_cls(dim, eps=norm_epsilon, **factory_kwargs)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, **factory_kwargs)
#         self.norm2 = norm_cls(dim, eps=norm_epsilon, **factory_kwargs)
#         self.pwconv1 = nn.Linear(dim, 4*dim, **factory_kwargs)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4*dim, dim, **factory_kwargs)
#         self.drop_path = DropPath(drop_path_rate)

#     def forward(self, x: torch.Tensor):
#         residual = x # (B, H, W, C)
#         x = self.norm1(x).permute(0, 3, 1, 2) # (B, C, H, W)
#         x = self.dwconv(x) # (B, C, H, W)
#         x = self.drop_path(x.permute(0, 2, 3, 1)) + residual # (B, H, W, C)

#         residual = x
#         x = self.norm2(x)
#         x = self.pwconv2(self.act(self.pwconv1(x)))
#         return self.drop_path(x) + residual
    
# class DownSample(nn.Module):
#     def __init__(self, in_chans, out_chans, norm_cls, norm_epsilon=1e-5, **factory_kwargs):
#         super().__init__()
#         self.norm = norm_cls(in_chans, eps=norm_epsilon, **factory_kwargs)
#         self.downsample = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1, **factory_kwargs)
    
#     def forward(self, x: torch.Tensor):
#         x = self.norm(x).permute(0, 3, 1, 2) # (B, H, W, C)
#         x = self.downsample(x)
#         return x.permute(0, 2, 3, 1) # (B, H, W, C)
    
# class Stem(nn.Module):
#     def __init__(
#             self, img_size=224, patch_size=16, in_chans=3, out_chans=512, drop_path_rate=0.,
#             norm_cls=nn.LayerNorm, norm_epsilon=1e-5, flatten=True, **factory_kwargs
#         ):
#         assert patch_size == 16
#         super().__init__()
#         self.img_size = to_2tuple(img_size)
#         self.patch_size = to_2tuple(patch_size)
#         self.grid_size = ((self.img_size[0] - self.patch_size[0]) // self.patch_size[0] + 1, (self.img_size[1] - self.patch_size[1]) // self.patch_size[1] + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]

#         self.stem = nn.Conv2d(in_chans, out_chans // 4, kernel_size=7, stride=4, padding=3, **factory_kwargs)
#         self.stage1 = nn.Sequential(*[Block(out_chans // 4, dpr[i], norm_cls, norm_epsilon, **factory_kwargs) for i in range(3)])
#         self.downsample1 = DownSample(out_chans // 4, out_chans // 2, norm_cls, norm_epsilon, **factory_kwargs)
#         self.stage2 = nn.Sequential(*[Block(out_chans // 2, dpr[3 + i], norm_cls, norm_epsilon, **factory_kwargs) for i in range(3)])
#         self.downsample2 = DownSample(out_chans // 2, out_chans, norm_cls, norm_epsilon, **factory_kwargs)

#     def forward(self, x: torch.Tensor):
#         x = self.stem(x).permute(0, 2, 3, 1) # (B, H, W, C)
#         x = self.stage1(x)
#         x = self.downsample1(x)
#         x = self.stage2(x)
#         x = self.downsample2(x)

#         return x.flatten(1, 2) if self.flatten else x