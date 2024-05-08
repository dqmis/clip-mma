import argparse
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

try:
    from wandb import wandb
except ImportError:
    print("Wandb is not installed. Please install it to log the results.")

import logging

from fomo.pipelines.types.learner_args import LearnerArgs
from fomo.pipelines.utils.initializers import (
    initalize_dataloaders,
    initalize_datasets,
    intialize_model,
)
from fomo.pipelines.utils.utils import (
    AverageMeter,
    ProgressMeter,
    accuracy,
    build_label_prompts,
    cosine_lr,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use wandb for logging",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names",
    )
    return parser.parse_args()


class Learner:
    """Trainer for prompt-learning using CLIP."""

    def __init__(self, learner_args: LearnerArgs) -> None:
        self._lr_args = learner_args

        set_seed(self._lr_args.seed)
        self.best_acc1 = 0.0

        # Load clip image transformation
        self.model = intialize_model(
            self._lr_args.model_type, self._lr_args.model_backbone, self._lr_args.device
        )
        self.transforms = self.model.transforms

        (train_dataset, test_dataset), labels = initalize_datasets(self._lr_args.dataset, self.transforms)

        self.train_loader, self.val_loader, self.test_loader = initalize_dataloaders(
            train_dataset, test_dataset, self._lr_args
        )

        # precompute prompt features
        self.model.precompute_prompt_features(build_label_prompts(labels, self._lr_args.text_prompt_template))

        self._configure_trainable_params()

        # Define criterion and optimizer
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self._lr_args.learning_rate,
            momentum=self._lr_args.momentum,
            weight_decay=self._lr_args.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

        # Define scheduler
        self.scheduler = cosine_lr(
            self.optimizer,
            self._lr_args.learning_rate,
            self._lr_args.warmup,
            len(self.train_loader) * self._lr_args.epochs,
        )

    def _configure_trainable_params(self) -> None:
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for learnable_param_name in self.model.learnable_param_names:
                if learnable_param_name in name:
                    param.requires_grad = True

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")

        # Print number of parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of learnable paramms: {num_params}")

    def run(self) -> None:
        """Runs training for the specified number of epochs."""
        
        if self._lr_args.use_wandb:
            wandb.init(
                project="fomo",
                name=self._lr_args.run_id,
                config=self._lr_args.to_dict(),
                reinit=True,
            )

        epochs_since_improvement = 0

        for epoch in range(self._lr_args.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            acc1 = self.evaluate()

            if self._lr_args.use_wandb:
                wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": acc1})

            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_acc1": self.best_acc1,
                    "optimizer": self.optimizer.state_dict(),
                },
                self._lr_args.output_dir,
                is_best=is_best,
            )

            if is_best:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                print(f"There's no improvement for {epochs_since_improvement} epochs.")

                if epochs_since_improvement >= self._lr_args.patience:
                    print("The training halted by early stopping criterion.")
                    break

        # loading best model
        self.model.load_state_dict(torch.load(f"{self._lr_args.output_dir}/model_best.pth.tar")["state_dict"])
        # evaluate on test set
        self.evaluate("test")

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch),
        )

        # Switch to train mode
        self.model.train()

        num_batches_per_epoch = len(self.train_loader)

        end = time.time()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            # Measure data loading time
            data_time.update(time.time() - end)

            # Adjust learning rate
            step = num_batches_per_epoch * epoch + i
            self.scheduler(step)

            self.optimizer.zero_grad()
            images = images.to(self.model.device)
            targets = targets.to(self.model.device)
            outputs = self.model(images).squeeze(-1)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)

            # Measure accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self._lr_args.print_freq == 0:
                progress.display(i)

            if i % self._lr_args.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_acc1": self.best_acc1,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self._lr_args.output_dir,
                )

        return losses.avg, top1.avg

    def evaluate(self, split: str = "valid") -> float:
        """Evaluates the model on the given `split` set and returns average accuracy."""
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1_prompt = AverageMeter("Prompt Acc@1", ":6.2f")
        loader = self.val_loader if split == "valid" else self.test_loader
        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1_prompt],
            prefix="Validate: ",
        )

        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.to(self.model.device)
                target = target.to(self.model.device)

                output = self.model(images)
                loss = self.criterion(output, target)
                

                acc1 = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_prompt.update(acc1[0].item(), images.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self._lr_args.print_freq == 0:
                    progress.display(i)

            print(" * Prompt Acc@1 {top1_prompt.avg:.3f}".format(top1_prompt=top1_prompt))

        if self._lr_args.use_wandb:
            wandb.log({f"{split}_acc": top1_prompt.avg})

        return top1_prompt.avg
