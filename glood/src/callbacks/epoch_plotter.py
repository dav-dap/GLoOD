from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from glood.src.callbacks.training_callback import TrainingCallback
import glood.src.plotting.plot_paths as pp
import glood.src.utils.env as env


class EpochPlotter(TrainingCallback):
    """
    Persist predicted and true snapshots for a fixed number of examples, once every ``epoch_interval`` epochs.
    The saved files can later be consumed by the plotting utilities under ``glood.src.plotting``.
    """

    def __init__(
        self,
        number_of_examples: int,
        epoch_interval: int,
    ) -> None:
        if number_of_examples <= 0:
            raise ValueError("EpochPlotter requires number_of_examples to be a positive integer.")
        if epoch_interval <= 0:
            raise ValueError("EpochPlotter requires epoch_interval to be a positive integer.")

        self.number_of_examples = number_of_examples
        self.epoch_interval = epoch_interval

    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        data: dict,
    ) -> None:
        if epoch % self.epoch_interval != 0 or epoch == 0:
            return

        device = next(model.parameters()).device
        with torch.no_grad():
            for loader_name, loader in zip(("train", "val"), (train_loader, val_loader)):
                if loader is None:
                    continue

                inputs, targets, matnums = self._collect_examples(loader, self.number_of_examples)
                preds = model(inputs.to(device, non_blocking=True)).cpu()
                targets_cpu = targets.cpu()

                self._persist_examples(loader_name, epoch, preds, targets_cpu, matnums=matnums)

        instantiate(data['cfg_plotters'])

    def _collect_examples(
        self,
        loader: DataLoader,
        num_examples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        inputs_parts: list[torch.Tensor] = []
        targets_parts: list[torch.Tensor] = []
        matnum_parts: list[torch.Tensor] = []
        collected = 0

        for batch in loader:
            batch_matnum: torch.Tensor | None = None
            if isinstance(batch, dict):
                if "inputs" not in batch:
                    raise TypeError("EpochPlotter expects dict batches to contain an 'inputs' key.")
                batch_inputs = batch["inputs"]
                batch_targets = batch.get("targets", batch_inputs)
                batch_matnum = batch.get("matnum")
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                batch_inputs, batch_targets = batch
            else:
                raise TypeError(
                    "EpochPlotter expects dataloader batches as (inputs, targets) or "
                    "a dict containing 'inputs' and optional 'targets'/'matnum'."
                )

            if not isinstance(batch_inputs, torch.Tensor) or not isinstance(batch_targets, torch.Tensor):
                raise TypeError("EpochPlotter requires both inputs and targets to be tensors.")
            if batch_matnum is not None and not isinstance(batch_matnum, torch.Tensor):
                raise TypeError("EpochPlotter requires 'matnum' to be a tensor when provided.")

            batch_size = batch_inputs.size(0)
            if batch_size == 0:
                continue

            remaining = num_examples - collected
            take = min(remaining, batch_size)
            if take <= 0:
                break

            inputs_parts.append(batch_inputs[:take].detach().cpu())
            targets_parts.append(batch_targets[:take].detach().cpu())
            if batch_matnum is not None:
                matnum_parts.append(batch_matnum[:take].detach().cpu())
            elif matnum_parts:
                raise ValueError("Inconsistent batches: some contain 'matnum' and others do not.")
            collected += take

            if collected >= num_examples:
                break

        if collected < num_examples:
            raise ValueError(
                f"Requested {num_examples} examples, but only {collected} were available from the dataloader."
            )

        inputs_batch = torch.cat(inputs_parts, dim=0)
        targets_batch = torch.cat(targets_parts, dim=0)
        matnum_batch = torch.cat(matnum_parts, dim=0) if matnum_parts else None
        return inputs_batch, targets_batch, matnum_batch

    def _persist_examples(
        self,
        loader_name: str,
        epoch: int,
        preds: torch.Tensor,
        targets: torch.Tensor,
        matnums: torch.Tensor | None = None,
    ) -> None:
        base_dir = env.out() / loader_name
        total_examples = preds.size(0)

        for example_idx, (pred, target) in enumerate(zip(preds, targets)):
            destination = pp.get_plot_data_path(base_dir, example_idx, epoch)
            self._prepare_destination(destination)

            np.save(destination / "pred.npy", pred.numpy())
            np.save(destination / "true.npy", target.numpy())
            if matnums is not None:
                np.save(destination / "matnum.npy", matnums[example_idx].numpy())
            print(
                f"EpochPlotter[{loader_name}] example {example_idx}: "
                f"pred shape={tuple(pred.shape)}, true shape={tuple(target.shape)}"
            )

        print(f"EpochPlotter[{loader_name}] saved {total_examples} example(s) for epoch {epoch}.")

    @staticmethod
    def _prepare_destination(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
