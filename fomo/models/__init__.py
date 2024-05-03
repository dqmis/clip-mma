from enum import Enum
from functools import partial

from fomo.models._base_model import BaseModel
from fomo.models.clip.clip_base import ClipBase


class Model(Enum):
    CLIP_VIT_BASE_PATCH16_PRETRAINED = partial(lambda: ClipBase)

    @classmethod
    def from_str(cls, name: str) -> BaseModel:
        model: BaseModel = cls[name.upper()].value()
        return model
