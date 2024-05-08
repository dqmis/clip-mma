from fomo.models.clip.clip_base import ClipBase
from fomo.models.clip.clip_linear import ClipLinear
from fomo.models.clip.clip_transformer import ClipTransformer
from fomo.models.clip.clip_mlp_head import CLIPMLPHead
from fomo.models.clip.clip_mm_mlp_adapter import CLIPMMMLPAdapter

MODELS = {
    "clip_base": ClipBase,
    "clip_linear": ClipLinear,
    "clip_transformer": ClipTransformer,
    "clip_mm_mlp": CLIPMLPHead,
    "clip_mm_mlp_adapter": CLIPMMMLPAdapter,
}
