from __future__ import annotations

from abc import ABC, abstractmethod
from torch import nn
from torch.utils.data import DataLoader

class TrainingCallback(ABC):
    @abstractmethod
    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        data: dict
    ) -> None:
        ...
