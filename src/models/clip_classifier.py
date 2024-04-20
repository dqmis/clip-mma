import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.models._base_model import BaseModel


class ClipClassifier(BaseModel):
    def __init__(self, model_base: str, class_names: list[str], class_template: str | None = None) -> None:
        self._model = CLIPModel.from_pretrained(model_base)
        self._processor = CLIPProcessor.from_pretrained(model_base)

        self._class_propmts = self._build_class_prompt(class_names, class_template)

    def _build_class_prompt(self, class_names: list[str], class_template: str | None) -> list[str]:
        class_template = class_template or "a photo of a {}"
        return [class_template.format(class_name) for class_name in class_names]

    def predict(self, x: DataLoader) -> torch.Tensor:

        predictions = []
        for image_paths in x:
            images = [Image.open(image_path) for image_path in image_paths]
            inputs = self._processor(
                text=self._class_propmts, images=images, return_tensors="pt", padding=True
            )
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            predictions.append(probs)
        predictions = torch.cat(predictions)
        predictions = predictions.argmax(dim=1)
        return torch.cat(predictions).cpu().numpy()
