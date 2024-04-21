from enum import Enum

from fomo.models._base_model import BaseModel
from fomo.models.clip_classifier import ClipClassifier


class Model(Enum):
    CLIP_VIT_BASE_PATCH32_PRETRAINED = ClipClassifier(model_base="openai/clip-vit-base-patch32")

    @classmethod
    def from_str(cls, name: str) -> BaseModel:
        return cls[name.upper()].value
