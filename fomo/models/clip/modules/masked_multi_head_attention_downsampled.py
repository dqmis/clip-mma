import torch
from torch import nn


class MaskedMultiheadAttentionDownsampled(nn.Module):
    def __init__(self, embed_dim=512, downsamling_dim=128, num_heads: int = 4) -> None:
        super(MaskedMultiheadAttentionDownsampled, self).__init__()

        self._num_heads = num_heads

        self.downsampler = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Linear(32, downsamling_dim),
        )
        self.mha = nn.MultiheadAttention(downsamling_dim, num_heads=self._num_heads)
        self._attn_mask: torch.Tensor = self._init_attn_mask(1, 1)
        self.upsampler = nn.Sequential(
            nn.Linear(downsamling_dim, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _init_attn_mask(num_prompts: int, num_images: int) -> torch.Tensor:
        num_total = num_prompts + num_images
        mask = torch.zeros((num_total, num_total))
        mask = mask - float("inf")

        for i in range(num_prompts):
            for j in range(num_prompts, num_total):
                mask[i, j] = 0

        for i in range(num_prompts, num_total):
            for j in range(num_prompts):
                mask[i, j] = 0

        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        if self._attn_mask.shape[0] != seq_len:
            self._attn_mask = self._init_attn_mask(seq_len - 1, 1)

        mask = self._attn_mask.clone().unsqueeze(0).repeat(batch_size * self._num_heads, 1, 1).to(self.device)

        output = self.downsampler(inputs)
        output, _ = self.mha.forward(output, output, output, attn_mask=mask)
        output = self.upsampler(output)
        return output
