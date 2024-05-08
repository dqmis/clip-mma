from fomo.models.clip.clip_base import ClipBase
from fomo.models.clip.clip_linear import ClipLinear
from fomo.models.clip.clip_transformer import ClipTransformer

MODELS = {
    "clip_base": ClipBase,
    "clip_linear": ClipLinear,
    "clip_transformer": ClipTransformer,
}
