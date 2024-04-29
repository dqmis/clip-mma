from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from fomo.models._base_model import BaseModel
from fomo.models.clip.clip_base import ClipBase


class ClipClassifier(BaseModel):
    def __init__(self, model_base: str, class_template: str | None = None) -> None:
        self._model = ClipBase(model_base)
        self._model.to_cpu()
        self._model.eval()

        self._class_propmts: list[str] | None = None

        self._class_template = class_template or "a photo of a {}"

    @property
    def batch_size(self) -> int:
        return 2

    @property
    def transforms(self) -> Compose:
        return self._model.transforms

    def reconfig_labels(self, labels: list[str]) -> None:
        prompts = self._build_class_prompt(labels)
        self._model.precompute_prompt_features(prompts)

    def _build_class_prompt(self, class_names: list[str]) -> list[str]:
        class_template = self._class_template
        return [class_template.format(class_name) for class_name in class_names]

    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        predictions = []
        targets = []
        for batch in tqdm(x):
            images, batch_targets = batch

            with torch.no_grad():
                logits_per_image = self._model.forward(images)
            probs = logits_per_image.softmax(dim=1)
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            targets.extend(batch_targets)
        return np.array(targets), np.array(predictions)
