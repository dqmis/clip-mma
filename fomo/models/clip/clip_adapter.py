import torch
from torch import nn

from fomo.models.clip.clip_base import ClipBase

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CLIPAdapter(ClipBase):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(CLIPAdapter, self).__init__(backbone, root=root)

        self.image_encoder = self._clip.visual
        self.logit_scale = self._clip.logit_scale
        self.adapter = Adapter(512, 4).to(torch.float32)

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

        image_features = self.encode_images(images)
        image_features = image_features.to(torch.float32)  # [batch_size, rep_dim]
        text_features = text_features.to(torch.float32)  # [n_classes, rep_dim]

        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_per_image: torch.Tensor = logit_scale * image_features @ text_features.t()

        return logits_per_image
