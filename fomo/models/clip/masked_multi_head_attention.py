import torch
from torch import nn


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, num_heads: int = 4) -> None:
        super(MaskedMultiheadAttention, self).__init__()

        self._num_heads = num_heads

        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=self._num_heads)
        self._attn_mask: torch.Tensor = self._init_attn_mask(1, 1)
        self.linear = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 512),
            
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _init_attn_mask(num_prompts: int, num_images: int) -> torch.Tensor:
        num_total = num_prompts + num_images
        mask = torch.zeros((num_total, num_total))

        for i in range(num_prompts):
            for j in range(num_prompts, num_total):
                mask[i, j] = 1

        for i in range(num_prompts, num_total):
            for j in range(num_prompts):
                mask[i, j] = 1

        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        if self._attn_mask.shape[0] != seq_len:
            self._attn_mask = self._init_attn_mask(seq_len - 1, 1)

        mask = self._attn_mask.clone().unsqueeze(0).repeat(batch_size * self._num_heads, 1, 1).to(self.device)

        output, _ = self.mha.forward(inputs, inputs, inputs, attn_mask=mask)
        output = self.linear(output)
        return output
