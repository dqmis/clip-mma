import torch
from torch import nn

from fomo.models.clip.clip_base import ClipBase


class ClipWithAdapter(nn.Module):
    """
    Class to add an adapter that combines image and text embeddings from CLIP and then computes logits.
    """

    def __init__(self, adapter: nn.Module, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        super(ClipWithAdapter, self).__init__()
        self._clip = ClipBase(backbone, root)
        self._adapter = adapter

    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        text_features = self._clip.encode_text(prompts).float()
        image_features = self._clip.encode_images(images).float()

        logits_per_image: torch.Tensor = self._adapter(image_features, text_features)

        return logits_per_image
