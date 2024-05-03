from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from numpy.typing import NDArray
from tqdm import tqdm
from clip import clip
from PIL.Image import Image


class ClipBase(nn.Module):
    def __init__(
        self, backbone: str = "ViT-B/16", root: str = "./data", class_template: str | None = None
    ) -> None:
        super(ClipBase, self).__init__()
        self._root = root
        self._clip, self._transforms = self._load_clip_to_cpu(backbone)

        self.logit_scale = self._clip.logit_scale.exp().detach()
        self._precomputed_prompt_features: torch.Tensor | None = None

        self._class_propmts: list[str] | None = None

        self._class_template = class_template or "a photo of a {}"

    @property
    def transforms(self) -> Compose:
        return self._transforms

    def transform(self, images: list[Image]) -> torch.Tensor:
        output: torch.Tensor = self._transforms(images)
        return output

    def to_cpu(self) -> None:
        self._clip = self._clip.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        self._clip = self._clip.to(torch.device("cuda"))

    @property
    def current_device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_class_prompt(self, class_names: list[str]) -> list[str]:
        class_template = self._class_template
        return [class_template.format(class_name) for class_name in class_names]

    def reconfig_labels(self, labels: list[str]) -> None:
        prompts = self._build_class_prompt(labels)
        self.precompute_prompt_features(prompts)

    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        predictions = []
        targets = []
        for batch in tqdm(x):
            images, batch_targets = batch

            with torch.no_grad():
                logits_per_image = self.forward(images)
            probs = logits_per_image.softmax(dim=1)
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            targets.extend(batch_targets)
        return np.array(targets), np.array(predictions)

    def precompute_image_embeddings(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        image_embeddings = []
        targets = []
        for batch in tqdm(x):
            images, labels = batch

            with torch.no_grad():
                image_embeddings.extend(self.encode_images(images).numpy())
                labels.extend(targets)

        return np.array(image_embeddings), np.array(targets)

    def precompute_prompt_embeddings(self) -> None:
        prompts = self._build_class_prompt(self._class_propmts)

        return self.encode_text(prompts).numpy()

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images, grad=True)
        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image

    def encode_images(self, images: torch.Tensor, grad: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(grad):
            image_features: torch.Tensor = self._clip.encode_image(images.to(self.current_device))
            image_features /= image_features.norm(dim=1, keepdim=True)

        return image_features

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(self.current_device)

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
