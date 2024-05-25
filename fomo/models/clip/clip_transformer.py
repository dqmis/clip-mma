# import ClipBase
from typing import Self

import torch

from fomo.models.clip.clip_base import ClipBase
from fomo.models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)


class ClipTransformer(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(ClipTransformer, self).__init__(backbone, root=root)

        self.mmha = MaskedMultiheadAttention()

    @property
    def learnable_param_names(self) -> set[str]:
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["mmha"])

    def eval(self) -> Self:
        self._clip.eval()
        self.mmha.eval()
        return self

    def train_(self) -> Self:
        self._clip.train()
        self.mmha.train()
        return self

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self.mmha.to(torch.device("cpu"))
        self._clip.float()

    def to_mps(self) -> None:
        self._clip.to(torch.device("mps"))
        self.mmha.to(torch.device("mps"))

    def to_cuda(self) -> None:
        self.mmha.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed prompt features has to be present.")

        image_features = self.encode_images(images).to(torch.float32).unsqueeze(0)
        text_features = text_features.to(torch.float32).unsqueeze(1).expand(-1, image_features.shape[1], -1)

        num_classes = text_features.shape[0]

        input_seq = torch.cat([text_features, image_features], dim=0)
        tr_outputs = self.mmha.forward(input_seq)

        _image_features = (
            (image_features + tr_outputs[num_classes:])
            .permute(1, 0, 2)
            .transpose(1, 2)
            .squeeze()
            .unsqueeze(1)
        )

        _text_features = text_features.permute(1, 0, 2)

        logits_per_image: torch.Tensor = torch.bmm(_image_features, _text_features.transpose(1, 2)).squeeze(1)

        return logits_per_image
