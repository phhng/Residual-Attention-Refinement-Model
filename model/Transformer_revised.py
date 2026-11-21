import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Revised inference path for the reported configuration.

    The original implementation used one encoder layer, but its output was
    discarded when return_encoder_output=False and num_decoder_layers=0.
    The prediction path therefore only normalizes the pooled query descriptors.
    This module preserves that behavior without instantiating or executing
    the unused encoder/decoder layers.

    The forward signature is retained for compatibility with MSR.
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        return_encoder_output=False,
    ):
        super().__init__()
        if return_encoder_output:
            raise ValueError(
                "The revised Transformer implements the pooled-descriptor "
                "prediction path only; return_encoder_output=True is unsupported."
            )
        if num_decoder_layers != 0:
            raise ValueError(
                "The revised Transformer is intended for the zero-layer decoder "
                "configuration used in the reported results."
            )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, src, mask, query_embed, query_pos, value_pos):
        # query_embed is the pooled descriptor supplied by MSR: [B, C, 1].
        # With zero decoder layers, the original decoder returned LayerNorm(tgt).
        # Match the original tensor layout: [B, C, 1] -> [1, B, C],
        # LayerNorm over C, then restore [B, C, 1].
        pooled = query_embed.squeeze(-1).unsqueeze(0)  # [1, B, C]
        return self.out_norm(pooled).squeeze(0).unsqueeze(-1)
