
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer import PositionEmbeddingSine, Transformer
from model.Backbone import Backbone
from utils.config import *

class ECALayer(nn.Module):
    """ECA-Net channel attention (fixed 1-D kernel size k)"""
    # NOTE: ECALayer returns the re-weighted features (x * w); CRAM multiplies by x again,
    # so the channel step is x * x * w. This is the variant evaluated in the paper.
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                              # x: [B, C, H, W]
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)          # [B, 1, C]
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)   # [B, C, 1, 1]
        return x * y.expand_as(x)


class DilatedSpatialAttention(nn.Module):
    """3x3 conv on [avg, max] channel descriptors, then a dilated kxk conv; returns the attention MAP (sigmoid)."""
    def __init__(self, kernel_size=7, dilation=3):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv_dilated = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=dilation * (kernel_size // 2),
                                      dilation=dilation, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv_dilated(self.conv1(torch.cat([avg_out, max_out], dim=1))))


class CRAM(nn.Module):
    def __init__(self, in_channels, eca_k_size=3, use_eca=True, use_dsa=False, dsa_kernel=7, dsa_dilation=3):
        super().__init__()
        self.channel_att = ECALayer(in_channels, k_size=eca_k_size) if use_eca else None
        self.spatial_att = DilatedSpatialAttention(dsa_kernel, dsa_dilation) if use_dsa else None

    def forward(self, x):
        if self.channel_att is not None:
            x = x * self.channel_att(x)
        if self.spatial_att is not None:
            x = x * self.spatial_att(x)
        return x


class FeatureEnhancementBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_eca=True, use_dsa=False):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (nn.Conv2d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())
        self.cram = CRAM(out_channels, use_eca=use_eca, use_dsa=use_dsa)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.residual(x) + self.skip(x))
        return self.cram(out)

class PooledNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q):
        # Input: (B, C, 1)
        q = q.squeeze(-1)       # (B, C)
        q = self.norm(q)        # (B, C)
        return q.unsqueeze(-1)  # (B, C, 1)

class MSR(nn.Module):
    def __init__(self, layers, num_classes=2, reduce_dim=256, use_eca=True, use_dsa=False):
        super().__init__()
        self.backbone = Backbone(f"resnet{layers}", train_backbone=False, return_interm_layers=True,
                                 dilation=[False, True, True])

        self.enhancement_cat = FeatureEnhancementBlock(reduce_dim, reduce_dim, use_eca, use_dsa)
        self.enhancement_3 = FeatureEnhancementBlock(reduce_dim, reduce_dim, use_eca, use_dsa)
        if ENCODER_CONFIG:
            self.embed_cat = nn.Embedding(reduce_dim, 1)
            self.embed_3 = nn.Embedding(reduce_dim, 1)
            self.pe_layer_cat = PositionEmbeddingSine(reduce_dim // 2, normalize=True)
            self.pe_layer_3 = PositionEmbeddingSine(reduce_dim // 2, normalize=True)

            tf_kwargs = dict(d_model=reduce_dim, dropout=0.1, nhead=4, dim_feedforward=reduce_dim // 4,
                                num_encoder_layers=1, num_decoder_layers=0, normalize_before=False,
                                return_intermediate_dec=False,return_encoder_output=ENCODER_CONFIG)
            self.transformer_cat = Transformer(**tf_kwargs)
            self.transformer_3 = Transformer(**tf_kwargs)
        if not ENCODER_CONFIG:
            self.norm_cat = PooledNorm(reduce_dim)
            self.norm_3 = PooledNorm(reduce_dim)

        def conv_red(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
                                 nn.BatchNorm2d(c_out), nn.ReLU())
        self.conv_red_1 = conv_red(512, reduce_dim // 2)
        self.conv_red_2 = conv_red(1024, reduce_dim // 2)
        self.conv_red_3 = conv_red(2048, reduce_dim)

        # self.cross_scale_attention = nn.MultiheadAttention(reduce_dim, 4, batch_first=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(reduce_dim * 2, reduce_dim), nn.BatchNorm1d(reduce_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(reduce_dim, reduce_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(reduce_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x, lung_mask=None):
        back_x = self.backbone(x)
    
        red_1 = self.conv_red_1(back_x["1"])
        red_2 = self.conv_red_2(back_x["2"])
        red_3 = self.conv_red_3(back_x["3"])
    
        red_cat = torch.cat((red_1, red_2), dim=1)
    
        # Optional feature-level masking before enhancement and attention
        if lung_mask is not None:
            m_cat = F.interpolate(
                lung_mask.float(),
                size=red_cat.shape[-2:],
                mode="nearest"
            )
            m_3 = F.interpolate(
                lung_mask.float(),
                size=red_3.shape[-2:],
                mode="nearest"
            )
    
            red_cat = red_cat * m_cat
            red_3 = red_3 * m_3
    
        enhanced_cat = self.enhancement_cat(red_cat) + red_cat
        enhanced_3 = self.enhancement_3(red_3) + red_3
        if ENCODER_CONFIG:
            masking = None
        
            # Branch 1: stage-2/3 features; branch 2: stage-4 features
            q_cat = self.avgpool(enhanced_cat).squeeze(-1)
            fg_cat = self.transformer_cat(
                enhanced_cat, masking, q_cat, self.embed_cat.weight,
                self.pe_layer_cat(enhanced_cat)
            )
        
            q_3 = self.avgpool(enhanced_3).squeeze(-1)
            fg_3 = self.transformer_3(
                enhanced_3, masking, q_3, self.embed_3.weight,
                self.pe_layer_3(enhanced_3)
            )
        
            out = torch.flatten(torch.cat((fg_cat, fg_3), dim=1), 1)
        
        if not ENCODER_CONFIG:
            q_cat = self.avgpool(enhanced_cat).flatten(1)   # (B, C)
            fg_cat = self.norm_cat(q_cat)
            q_3 = self.avgpool(enhanced_3).flatten(1)
            fg_3 = self.norm_3(q_3)
            out = torch.cat((fg_cat, fg_3), dim=1)          # (B, 2C)

        return self.classifier(out)


class FocalLossCE(nn.Module):
    """Focal loss no class weighting (== alpha 1)."""
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()