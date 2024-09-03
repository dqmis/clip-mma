from fomo.models.clip.ablations.clip_mha_adapter import CLIPMhaAdapter
from fomo.models.clip.ablations.clip_mha_adapter_mlp_bottleneck import CLIPMhaAdapterMlpBottleneck
from fomo.models.clip.ablations.clip_transformer_adapter import CLIPTransformerAdapter
from fomo.models.clip.ablations.clip_transformer_adapter_mlp_bottleneck import (
    CLIPTransformerAdapterMlpBottleneck,
)
from fomo.models.clip.clip_adapter import CLIPAdapter
from fomo.models.clip.clip_base import ClipBase
from fomo.models.clip.clip_linear import ClipLinear
from fomo.models.clip.clip_mlp_head import CLIPMLPHead
from fomo.models.clip.clip_mm_mlp_adapter import CLIPMMMLPAdapter
from fomo.models.clip.clip_transformer import ClipTransformer
from fomo.models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from fomo.models.clip.clip_transformer_downscaled import ClipTransformerDownscaled
from fomo.models.clip.clip_transformer_w_hypernet import ClipTransformerWHypernet

MODELS = {
    "clip_base": ClipBase,
    "clip_linear": ClipLinear,
    "clip_transformer": ClipTransformer,
    "clip_transformer_downscaled": ClipTransformerDownscaled,
    "clip_mm_mlp": CLIPMLPHead,
    "clip_mm_mlp_adapter": CLIPMMMLPAdapter,
    "clip_transformer_w_hypernet": ClipTransformerWHypernet,
    "clip_transformer_adapter_text": CLIPTransformerAdapterText,
    "clip_adapter": CLIPAdapter,
    # Ablations
    "clip_transformer_adapter": CLIPTransformerAdapter,
    "clip_mha_adapter": CLIPMhaAdapter,
    "clip_transformer_adapter_mlp_bottleneck": CLIPTransformerAdapterMlpBottleneck,
    "clip_mha_adapter_mlp_bottleneck": CLIPMhaAdapterMlpBottleneck,
}
