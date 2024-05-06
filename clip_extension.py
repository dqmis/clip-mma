from fomo.models.clip.clip_base import ClipBase
from torch import nn
import torch
from typing_extensions import Self,List, Union


class ClipExtension(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(ClipExtension, self).__init__(backbone, root=root)

        # add additional blocks to the model

        self.visual_mlp = nn.Sequential(
            nn.Linear(self._clip.visual.output_dim, 12),
            nn.Linear(12, self._clip.visual.output_dim)
        )

    @property
    def learnable_param_names(self) -> set[str]:
         # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["image_linear"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self.image_linear.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        self.image_linear.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: Union[list[str], None] = None) -> torch.Tensor:
        # Change the forward method to include the visual_mlp
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)

        image_features = self.image_linear(image_features)

        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image

if __name__ == "__main__":
    model = ClipExtension()

    #print learnable parameters
    print(model.learnable_param_names)