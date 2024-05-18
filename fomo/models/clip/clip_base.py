from typing import Self

import torch
from clip import clip
from PIL.Image import Image
from torch import nn
from torchvision.transforms import Compose


class ClipBase(nn.Module):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        super(ClipBase, self).__init__()
        self._root = root
        self._clip, self._transforms = self._load_clip_to_cpu(backbone)

        self.logit_scale = self._clip.logit_scale.exp().detach()

        self._precomputed_prompt_features: torch.Tensor | None = None

    @property
    def learnable_param_names(self) -> set[str]:
        return set()

    @property
    def transforms(self) -> Compose:
        return self._transforms

    def eval(self) -> Self:
        self._clip.eval()
        return self

    def train_(self) -> Self:
        self._clip.train()
        return self

    def transform(self, images: list[Image]) -> torch.Tensor:
        output: torch.Tensor = self._transforms(images)
        return output

    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self._clip.float()

    def to_mps(self) -> None:
        self._clip.to(torch.device("mps"))

    def to_cuda(self) -> None:
        self._clip.to(torch.device("cuda"))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)

        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features: torch.Tensor = self._clip.encode_image(images.to(self.device))
            image_features /= image_features.norm(dim=1, keepdim=True)

        return image_features

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(self.device)

        with torch.no_grad():
            text_features: torch.Tensor = self._clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def precompute_prompt_features(self, prompts: list[str]) -> None:
        self._precomputed_prompt_features = self.encode_text(prompts)

    def _load_clip_to_cpu(self, backbone: str) -> tuple[nn.Module, Compose]:
        try:
            url = clip._MODELS[backbone]
        except KeyError:
            raise KeyError(f"Invalid backbone {backbone} selected.")

        model_path = clip._download(url, self._root)

        try:
            jit_model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model: nn.Module = clip.build_model(state_dict or jit_model.state_dict())
        return model, clip._transform(jit_model.input_resolution.item())
