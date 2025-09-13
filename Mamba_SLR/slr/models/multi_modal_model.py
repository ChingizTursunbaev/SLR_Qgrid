# slr/models/multi_modal_model.py

import torch
import torch.nn as nn
from .model import Model # Reuse the existing Vision Transformer

class MultiModalMamba(nn.Module):
    def __init__(self, img_cfg, qgrid_cfg, kp_cfg, num_classes, fusion_cfg):
        super().__init__()

        self.image_encoder = Model(**img_cfg)
        self.qgrid_encoder = Model(**qgrid_cfg)

        self.keypoint_encoder = nn.Sequential(
            nn.Linear(kp_cfg['input_dim'], kp_cfg['model_dim']),
            nn.ReLU(),
            nn.LayerNorm(kp_cfg['model_dim'])
        )

        self.fusion_cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_cfg['embed_dim'],
            num_heads=fusion_cfg['num_heads'],
            dropout=fusion_cfg['dropout'],
            batch_first=True
        )
        
        fused_dim = self.image_encoder.d_model + fusion_cfg['embed_dim']
        self.ctc_head = nn.Linear(fused_dim, num_classes)

    def forward(self, images, qgrids, keypoints):
        """
        images: (B, T_img, C, H, W)
        qgrids: (B, T_qgrid, C, H, W)
        keypoints: (B, T_img, D)
        """
        B, T_img, C_img, H_img, W_img = images.shape
        images_reshaped = images.view(B * T_img, C_img, H_img, W_img)

        B, T_qgrid, C_qgrid, H_qgrid, W_qgrid = qgrids.shape
        qgrids_reshaped = qgrids.view(B * T_qgrid, C_qgrid, H_qgrid, W_qgrid)
        
        image_features = self.image_encoder(images_reshaped)
        qgrid_features = self.qgrid_encoder(qgrids_reshaped)
        
        image_features = image_features.view(B, T_img, -1)
        qgrid_features = qgrid_features.view(B, T_qgrid, -1)

        keypoint_features = self.keypoint_encoder(keypoints)

        low_freq_features = image_features + keypoint_features

        qgrid_context, _ = self.fusion_cross_attention(
            query=low_freq_features,
            key=qgrid_features,
            value=qgrid_features
        )

        final_fused_features = torch.cat([low_freq_features, qgrid_context], dim=-1)

        logits = self.ctc_head(final_fused_features)
        
        return logits.permute(1, 0, 2)