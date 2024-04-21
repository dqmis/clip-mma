from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from fomo.models._base_model import BaseModel


class ClipClassifier(BaseModel):
    def __init__(self, model_base: str, class_template: str | None = None) -> None:
        self._model = CLIPModel.from_pretrained(model_base)
        self._processor = CLIPProcessor.from_pretrained(model_base)
        self._class_propmts: list[str] | None = None

        self._class_template = class_template or "a photo of a {}"

    @property
    def batch_size(self) -> int:
        return 32

    def reconfig_labels(self, labels: list[str]) -> None:
        self._class_propmts = self._build_class_prompt(labels)

    def _build_class_prompt(self, class_names: list[str]) -> list[str]:
        class_template = self._class_template
        return [class_template.format(class_name) for class_name in class_names]

    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        assert self._class_propmts

        predictions = []
        targets = []
        for batch in tqdm(x):
            images, batch_targets = batch

            inputs = self._processor(
                text=self._class_propmts, images=images, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            targets.extend(batch_targets)
        return np.array(targets), np.array(predictions)
