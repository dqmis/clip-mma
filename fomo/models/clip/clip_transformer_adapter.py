import torch
from torch import nn

from fomo.models.clip.clip_base import ClipBase
from fomo.models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)
from fomo.models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)



class CLIPTransformerAdapter(ClipBase):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(CLIPTransformerAdapter, self).__init__(backbone, root=root)

        self.image_encoder = self._clip.visual
        self.logit_scale = self._clip.logit_scale
        self.adapter = MaskedMultiheadAttentionDownsampled()

    @property
    def learnable_param_names(self) -> set[str]:
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["adapter"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self.adapter.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        self.adapter.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        # Change the forward method to include the visual_mlp
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images).to(torch.float32)  # [batch_size, rep_dim]
        text_features = text_features.to(torch.float32)  # [n_classes, rep_dim]

        num_classes = text_features.shape[0]

        adapter_input = torch.cat(
            [text_features.unsqueeze(1).expand(-1, image_features.shape[0], -1), image_features.unsqueeze(0)],
            dim=0,
        )

        adapter_output = self.adapter(adapter_input)

        adapter_image_features = adapter_output[num_classes:].squeeze(0)
        adapter_text_features = adapter_output[:num_classes].mean(dim=1)
        
        adapter_image_features = adapter_image_features / adapter_image_features.norm(dim=-1, keepdim=True)
        
        image_ratio = 0.2
        image_features = image_ratio * adapter_image_features + (1 - image_ratio) * image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image: torch.Tensor = logit_scale * image_features @ text_features.t()

        return logits_per_image
