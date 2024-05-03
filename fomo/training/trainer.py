import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model.to_cuda()
        else:
            self.model.to_cpu()

    def train(self, epochs: int) -> None:
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    logger.info(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}")
                    running_loss = 0.0

            self.validate()

    def validate(self) -> None:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for data in self.val_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                total_loss += self.compute_loss(outputs, labels).item()
                total_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        size = len(self.val_dataloader.dataset)
        avg_loss = total_loss / size
        avg_accuracy = total_correct / size
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    def compute_loss(self, outputs: Any, labels: Any) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, labels)
