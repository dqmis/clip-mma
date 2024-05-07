import torch
from torch import nn
from typing_extensions import Self,List, Union
from fomo.models.clip.clip_base import ClipBase


class ClipLinear(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        super(ClipLinear, self).__init__(backbone, root=root)

        self.image_linear = nn.Sequential(
            nn.Linear(self._clip.visual.output_dim, self._clip.visual.output_dim)
        )

    @property
    def learnable_param_names(self) -> set[str]:
        return set(["image_linear"])

    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self.image_linear.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        self.image_linear.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: Union[list[str], None] = None) -> torch.Tensor:
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)
        image_features = image_features.to(torch.float32)
        text_features = text_features.to(torch.float32)
        image_features = self.image_linear(image_features)

        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image
