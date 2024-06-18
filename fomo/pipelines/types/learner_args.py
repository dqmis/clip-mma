import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LearnerArgs:
    use_wandb: bool = True
    epochs: int = 100
    patience: int = 5
    model_type: str = "clip_base"
    print_freq: int = 200
    save_freq: int = 200
    output_dir: str = "./output/"
    model_backbone: str = "ViT-B/16"
    dataset: str = "cifar10"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 4
    train_size: float | None = None
    train_eval_size: tuple[int, int] | None = None
    text_prompt_template: str = "a photo of {}."
    learning_rate: float = 0.01
    momentum: float = 0.5
    weight_decay: float = 1e-4
    warmup: int = 0
    info: str | None = None
    seed: int = 42
    train_subsample: str = "all"
    test_subsample: str = "all"
    gaussian_noise_std: float = 0.0
    evaluate_only: bool = False

    def __post_init__(self) -> None:
        self.run_id = f"{self.model_type}_{self.dataset}_{str(int(time.time()))}".replace("/", "").lower()
        self.output_dir = os.path.join(self.output_dir, self.run_id)

    def save_config(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            data = self.to_dict()
            json.dump(data, f, indent=4)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__
