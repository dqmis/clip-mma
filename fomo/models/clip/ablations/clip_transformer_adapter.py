from fomo.models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from fomo.models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPTransformerAdapter(CLIPTransformerAdapterText):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(CLIPTransformerAdapter, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(block_type="transformer")
