import torch
from torch import Tensor, nn

from fomo.models.clip.clip_base import ClipBase
from fomo.models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)


class HyperNet(nn.Module):
    def __init__(self, embed_dim: int = 512) -> None:
        super(HyperNet, self).__init__()

        self.embed_dim = embed_dim
        self.weight_shapes = [
            (embed_dim, 4),
            (4, embed_dim),
        ]  # Shape of the network parametrized by hypernetwork
        self.hidden_size = 16  # hypernetwork hidden size
        num_weights = sum(w[0] * w[1] for w in self.weight_shapes)

        self.fc1 = nn.Linear(embed_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_weights)

    def forward(self, x: Tensor) -> Tensor:
        weights = self.generate_weights(x)
        return self.apply_main_network(x, weights)

    def generate_weights(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        weights = self.fc2(x)
        return weights

    def apply_main_network(self, x: Tensor, weights: Tensor) -> Tensor:
        start = 0

        for shape in self.weight_shapes:
            layer_in_size, layer_out_size = shape[0], shape[1]
            end = start + layer_in_size * layer_out_size

            layer_weights = weights[:, :, start:end].view(
                weights.shape[0], weights.shape[1], layer_in_size, layer_out_size
            )

            x = torch.einsum("bce,bcej->bcj", x, layer_weights)
            x = torch.relu(x)

            start = end

        return x


class ClipTransformerWHypernet(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(ClipTransformerWHypernet, self).__init__(backbone, root=root)

        down_dim = 32

        self.mmha = MaskedMultiheadAttention(embed_dim=down_dim)

        # Downsampling from 512 to down_dim
        self.image_downsample = nn.Linear(512, down_dim)
        self.text_downsample = nn.Linear(512, down_dim)

        # Upsampling from down_dim to 512
        self.image_upsample = nn.Linear(down_dim, 512)
        self.text_upsample = nn.Linear(down_dim, 512)

        # Hypernetworks
        self.image_hypernetwork = HyperNet(embed_dim=down_dim)
        self.text_hypernetwork = HyperNet(embed_dim=down_dim)

    @property
    def learnable_param_names(self) -> set[str]:
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(
            [
                "mmha",
                "image_downsample",
                "text_downsample",
                "image_upsample",
                "text_upsample",
                "image_hypernetwork",
                "text_hypernetwork",
            ]
        )

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self._clip.float()
        self.mmha.to(torch.device("cpu"))
        self.image_downsample.to(torch.device("cpu"))
        self.text_downsample.to(torch.device("cpu"))
        self.image_upsample.to(torch.device("cpu"))
        self.text_upsample.to(torch.device("cpu"))
        self.image_hypernetwork.to(torch.device("cpu"))
        self.text_hypernetwork.to(torch.device("cpu"))

    def to_mps(self) -> None:
        self._clip.to(torch.device("mps"))
        self.mmha.to(torch.device("mps"))
        self.image_downsample.to(torch.device("mps"))
        self.text_downsample.to(torch.device("mps"))
        self.image_upsample.to(torch.device("mps"))
        self.text_upsample.to(torch.device("mps"))
        self.image_hypernetwork.to(torch.device("mps"))
        self.text_hypernetwork.to(torch.device("mps"))

    def to_cuda(self) -> None:
        self._clip.to(torch.device("cuda"))
        self.mmha.to(torch.device("cuda"))
        self.image_downsample.to(torch.device("cuda"))
        self.text_downsample.to(torch.device("cuda"))
        self.image_upsample.to(torch.device("cuda"))
        self.text_upsample.to(torch.device("cuda"))
        self.image_hypernetwork.to(torch.device("cuda"))
        self.text_hypernetwork.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        if prompts:
            text_features = self.encode_text(prompts)

        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features

        else:
            raise ValueError("At least one prompts or pre-computed prompt features has to be present.")

        batch_size = images.shape[0]
        num_classes = text_features.shape[0]

        image_features = self.encode_images(images)

        image_features = image_features.to(torch.float32).unsqueeze(0)

        text_features = text_features.to(torch.float32).unsqueeze(1).expand(-1, batch_size, -1)

        _image_features = self.image_downsample(image_features)
        _text_features = self.text_downsample(text_features)

        input_seq = torch.cat([_text_features, _image_features], dim=0)

        tr_outputs = self.mmha.forward(input_seq)

        input_seq = input_seq + tr_outputs

        _image_features = input_seq[num_classes:]  # [1, batch_size, embed_dim]
        _text_features = input_seq[:num_classes]  # [n_classes, batch_size, embed_dim]

        _image_features = _image_features.permute(1, 0, 2)  # [batch_size, 1, embed_dim]
        _text_features = _text_features.permute(1, 0, 2)  # [batch_size, n_classes, embed_dim]

        _image_features = self.image_hypernetwork(_image_features)
        # _text_features = self.text_hypernetwork(_text_features)

        _image_features = self.image_upsample(_image_features)
        _text_features = self.text_upsample(_text_features)

        _image_features = _image_features.transpose(1, 2)

        text_features = text_features.permute(1, 0, 2) + _text_features
        image_features = image_features.permute(1, 2, 0) + _image_features

        logits_per_image: torch.Tensor = torch.bmm(text_features, image_features).squeeze(2)

        return logits_per_image
