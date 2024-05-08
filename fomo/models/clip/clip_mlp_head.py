import torch
from torch import nn

from fomo.models.clip.clip_base import ClipBase


class CLIPMLPHead(ClipBase):
    """Clip with a multimodal adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(CLIPMLPHead, self).__init__(backbone, root=root)

        # add additional blocks to the model
        representation_dim = self._clip.visual.output_dim
        adapter_input_dim = representation_dim * 2
        hidden_size = 16

        self.combination_layer = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    @property
    def learnable_param_names(self) -> set[str]:
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["combination_layer"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self.combination_layer.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        self.combination_layer.to(torch.device("cuda"))
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

        image_features_exp = image_features.unsqueeze(1).repeat(
            1, text_features.shape[0], 1
        )  # [batch_size, n_classes, rep_dim]
        text_features_exp = text_features.unsqueeze(0).repeat(
            image_features.shape[0], 1, 1
        )  # [batch_size, n_classes, rep_dim]

        combined_features = torch.cat(
            (image_features_exp, text_features_exp), dim=2
        )  # [batch_size, n_classes, rep_dim]

        logits_per_image: torch.Tensor = self.combination_layer(combined_features).squeeze(
            -1
        )  # [batch_size, n_classes]

        return logits_per_image
