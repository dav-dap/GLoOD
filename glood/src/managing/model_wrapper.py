from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Union
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, IterableDataset
from hydra.utils import instantiate

from glood.src.callbacks.training_callback import TrainingCallback
import glood.src.plotting.plot_paths as pp
import glood.src.utils.env as env

Batch = Union[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    list[torch.Tensor],
    Dict[str, torch.Tensor],
]

MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ModelWrapper:
    """
    Lightweight training helper that wraps a PyTorch module and handles
    dataloader creation plus a minimal training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_cls: type[Optimizer],
        *,
        optimizer_kwargs: Dict[str, object] | None = None,
        scheduler_cls: type,
        scheduler_kwargs: Dict[str, object] | None = None,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        raw_optimizer_kwargs = dict(optimizer_kwargs or {})
        self.optimizer_kwargs = self._normalize_optimizer_kwargs(raw_optimizer_kwargs)
        self.optimizer: Optimizer | None = None
        if scheduler_cls is None:
            raise ValueError("ModelWrapper requires a scheduler_cls to be provided.")
        self.scheduler_cls = scheduler_cls
        raw_scheduler_kwargs = dict(scheduler_kwargs or {})
        self.scheduler_kwargs = self._normalize_optimizer_kwargs(raw_scheduler_kwargs)
        self.scheduler: object | None = None
        self._scheduler_requires_metric = False

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                "ModelWrapper only supports CUDA devices; training without a GPU is disallowed."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA device {self.device} but only {torch.cuda.device_count()} device(s) present."
            )

        self.model.to(self.device)

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                self.num_workers = int(slurm_cpus)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid SLURM_CPUS_PER_TASK value: {slurm_cpus!r}"
                ) from exc
        else:
            affinity_count: int | None = None
            try:
                affinity = os.sched_getaffinity(0)
            except AttributeError:
                affinity = None
            except OSError:
                affinity = None

            if affinity is not None:
                affinity_count = len(affinity)

            cpu_count = os.cpu_count()

            if affinity_count is not None and cpu_count is not None:
                self.num_workers = min(affinity_count, cpu_count)
            else:
                self.num_workers = affinity_count or cpu_count

            if self.num_workers is None:
                raise RuntimeError("Unable to determine CPU count via os.cpu_count().")

        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive; received {self.num_workers}.")

    def train(
        self,
        cfg_plotters: DictConfig,
        dataset: IterableDataset,
        *,
        epochs: int,
        batch_size: int,
        val_fraction: float | None = None,
        split_shuffle: bool = True,
        val_batch_size: int | None = None,
        gradient_clip_norm: float | None = None,
        callbacks: Iterable[TrainingCallback] | None = None,
        metrics: Iterable[MetricFn] | None = None,
    ) -> None:
        """
        Execute a training loop over the supplied iterable dataset and track per-epoch statistics.

        Notes
        -----
        Callback objects are responsible for consuming the per-epoch statistics that are
        calculated inside this method. No aggregated history is returned.
        """
        train_loader, val_loader = self._ensure_dataloader(
            dataset,
            batch_size=batch_size,
            val_fraction=val_fraction,
            split_shuffle=split_shuffle,
            val_batch_size=val_batch_size,
        )
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        self.scheduler = self._instantiate_scheduler(self.optimizer)
        self.model.train()

        metric_fns = tuple(metrics or ())
        metric_names = tuple(self._metric_name(metric_fn, idx) for idx, metric_fn in enumerate(metric_fns))
        callback_list = list(callbacks or [])

        for epoch_idx in range(epochs):
            train_loss, train_metrics_epoch = self._run_train_epoch(
                train_loader,
                metric_fns,
                metric_names,
                gradient_clip_norm=gradient_clip_norm,
            )

            val_loss, val_metrics_epoch = self._run_validation_epoch(
                val_loader,
                metric_fns,
                metric_names,
            )

            self._maybe_step_scheduler(train_loss, val_loss)

            if callback_list:
                model_was_training = self.model.training
                try:
                    self.model.eval()
                    for callback in callback_list:
                        callback(
                            epoch_idx,
                            self.model,
                            train_loader,
                            val_loader,
                            {
                                "train": {
                                    "loss": train_loss,
                                    "metrics": dict(train_metrics_epoch),
                                },
                                "validation": {
                                    "loss": val_loss,
                                    "metrics": dict(val_metrics_epoch),
                                },
                                "cfg_plotters": cfg_plotters,
                                "optimizer": self.optimizer,
                                "scheduler": self.scheduler,
                                "learning_rates": tuple(self._current_learning_rates()),
                                },
                        )
                finally:
                    self.model.train(model_was_training)

        return None

    def infer(
        self,
        cfg_plotters: DictConfig,
        dataset: IterableDataset,
        *,
        n_items: int,
    ) -> None:
        """
        Run inference on the dataset, persisting predictions/targets for the first n_items examples.

        If n_items <= 0, the entire dataset is processed.
        """
        if not isinstance(dataset, IterableDataset):
            raise TypeError("ModelWrapper expects an IterableDataset instance.")
        if not isinstance(n_items, int):
            raise TypeError("n_items must be an int.")

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.model.eval()
        total_examples = 0
        base_dir = env.out() / "infer"
        epoch_idx = 0

        with torch.no_grad():
            for batch in dataloader:
                if n_items > 0 and total_examples >= n_items:
                    break

                inputs, targets = self._unpack_batch(batch)
                matnum = self._extract_matnum(batch)
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs).detach().cpu()
                targets_cpu = targets.detach().cpu()
                matnum_cpu = matnum.detach().cpu() if matnum is not None else None

                batch_size = outputs.size(0)
                if n_items > 0:
                    remaining = n_items - total_examples
                    if remaining <= 0:
                        break
                    if remaining < batch_size:
                        outputs = outputs[:remaining]
                        targets_cpu = targets_cpu[:remaining]
                        if matnum_cpu is not None:
                            matnum_cpu = matnum_cpu[:remaining]
                        batch_size = remaining

                for idx in range(batch_size):
                    example_idx = total_examples + idx
                    destination = pp.get_plot_data_path(base_dir, example_idx, epoch_idx)
                    self._prepare_destination(destination)
                    np.save(destination / "pred.npy", outputs[idx].numpy())
                    np.save(destination / "true.npy", targets_cpu[idx].numpy())
                    if matnum_cpu is not None:
                        np.save(destination / "matnum.npy", matnum_cpu[idx].numpy())

                total_examples += batch_size

        if total_examples == 0:
            raise ValueError("Inference produced no examples.")

        instantiate(cfg_plotters)

        return None

    def _ensure_dataloader(
        self,
        dataset: IterableDataset,
        *,
        batch_size: int,
        val_fraction: float | None = None,
        split_shuffle: bool = True,
        val_batch_size: int | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:
        if not isinstance(dataset, IterableDataset):
            raise TypeError("ModelWrapper expects an IterableDataset instance.")

        if val_fraction is None:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.num_workers,
                shuffle=False,  # Shuffling is already handled inside the IterableDataset
            )
            return train_loader, None

        if not hasattr(dataset, "split"):
            raise TypeError("val_fraction requires the dataset to provide a split() method.")

        train_dataset, val_dataset = dataset.split(val_fraction, shuffle=split_shuffle)
        if hasattr(val_dataset, "shuffle"):
            val_dataset.shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Shuffling is already handled inside the IterableDataset
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return train_loader, val_loader

    def _unpack_batch(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, torch.Tensor):
            return batch, batch
        if isinstance(batch, dict):
            try:
                inputs = batch["inputs"]
                targets = batch.get("targets", inputs)
                return inputs, targets
            except KeyError as err:
                raise ValueError("Batch dict must contain an 'inputs' key.") from err
        if isinstance(batch, (list, tuple)):
            if len(batch) != 2:
                raise ValueError("Batch tuples/lists must contain exactly two elements: (inputs, targets).")
            inputs, targets = batch
            return inputs, targets
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    @staticmethod
    def _extract_matnum(batch: Batch) -> torch.Tensor | None:
        if isinstance(batch, dict):
            matnum = batch.get("matnum")
            if matnum is None:
                return None
            if not isinstance(matnum, torch.Tensor):
                raise TypeError("Batch dict 'matnum' must be a torch.Tensor when provided.")
            return matnum
        return None

    def _run_train_epoch(
        self,
        dataloader: DataLoader,
        metrics: Sequence[MetricFn],
        metric_names: Sequence[str],
        *,
        gradient_clip_norm: float | None,
    ) -> tuple[float, Dict[str, float]]:
        """
        Iterate over the training dataloader, performing optimization, and return loss/metrics averages.
        """
        total_loss = 0.0
        steps = 0
        metric_totals = [0.0 for _ in metrics]
        was_training = self.model.training
        self.model.train(True)

        try:
            for batch in dataloader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)

                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

                if metrics:
                    detached_outputs = outputs.detach()
                    detached_targets = targets.detach()
                    for idx, metric_fn in enumerate(metrics):
                        metric_value = metric_fn(detached_outputs, detached_targets)
                        metric_totals[idx] += self._as_float(metric_value)
        finally:
            self.model.train(was_training)

        if steps == 0:
            raise ValueError("The provided dataloader produced no batches.")

        mean_loss = total_loss / steps
        metric_dict: Dict[str, float] = {}
        if metric_names:
            metric_dict = {name: total / steps for name, total in zip(metric_names, metric_totals)}
        return mean_loss, metric_dict

    def save_model(self) -> None:
        """
        Persist the wrapped model weights and optimizer state to disk.

        Parameters
        ----------
        filepath:
            Destination file path. Parent directories are created automatically.
        """
        if self.optimizer is None:
            raise ValueError(
                "No optimizer available to save. Run training first or load a checkpoint."
            )

        weights_path = env.out() / "model_weights.pt"
        optimizer_path = env.out() / "optimizer_state.pt"
        scheduler_path = env.out() / "scheduler_state.pt"

        torch.save(self.model.state_dict(), weights_path)

        optimizer_state = self.optimizer.state_dict()
        self._log_optimizer_state_snapshot(optimizer_state)
        torch.save(optimizer_state, optimizer_path)
        if self.scheduler is None:
            raise ValueError("Scheduler instance missing; cannot save scheduler state.")
        if not hasattr(self.scheduler, "state_dict"):
            raise TypeError("Scheduler does not provide a state_dict() method.")
        torch.save(self.scheduler.state_dict(), scheduler_path)

        print('--> Model weights and optimizer saved correctly.')

    def load_model(self, input_dir: Union[str, os.PathLike[str]]) -> None:
        """
        Load model and optimizer state from disk, restoring the wrapper for further use.

        Parameters
        ----------
        input_dir:
            Source dir path containing the previously produced by `save_model`.
        """
        weights_path = env.cwd() / input_dir / "model_weights.pt"
        optimizer_path = env.cwd() / input_dir / "optimizer_state.pt"
        scheduler_path = env.cwd() / input_dir / "scheduler_state.pt"

        missing = [p for p in (weights_path, optimizer_path, scheduler_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoint component(s): {', '.join(str(p) for p in missing)}."
            )

        state_dict = torch.load(weights_path, map_location=self.device)
        optimizer_state_dict = torch.load(optimizer_path, map_location=self.device)

        if not isinstance(state_dict, dict):
            raise TypeError("Weights file did not contain a state dict.")
        if not isinstance(optimizer_state_dict, dict):
            raise TypeError("Optimizer file did not contain a state dict.")

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.scheduler = None
        if self.scheduler_cls is not None:
            self.scheduler = self._instantiate_scheduler(self.optimizer)
            if self.scheduler is not None and scheduler_path.exists():
                scheduler_state_dict = torch.load(scheduler_path, map_location=self.device)
                if hasattr(self.scheduler, "load_state_dict"):
                    self.scheduler.load_state_dict(scheduler_state_dict)

    def _normalize_optimizer_kwargs(self, kwargs: Dict[str, object]) -> Dict[str, object]:
        """
        Convert optimizer kwargs into plain Python containers so PyTorch checkpoints stay safe to load.
        """
        return {key: self._normalize_optimizer_value(value) for key, value in kwargs.items()}

    def _normalize_optimizer_value(self, value: object) -> object:
        if isinstance(value, DictConfig):
            return {key: self._normalize_optimizer_value(item) for key, item in value.items()}
        if isinstance(value, ListConfig):
            return tuple(self._normalize_optimizer_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._normalize_optimizer_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._normalize_optimizer_value(item) for item in value)
        return value

    def _log_optimizer_state_snapshot(self, state: Dict[str, object]) -> None:
        """
        Emit a concise snapshot of the optimizer state structure to aid debugging.
        """
        param_groups = state.get("param_groups", [])
        group_summaries = []
        for idx, group in enumerate(param_groups):
            summary = {key: type(value).__name__ for key, value in group.items() if key != "params"}
            summary["params_count"] = len(group.get("params", []))
            summary["index"] = idx
            group_summaries.append(summary)
        if group_summaries:
            print(f"-->\tOptimizer param_groups summary: {group_summaries}")
        else:
            print("-->\tOptimizer param_groups summary: []")

        listconfig_paths: list[str] = []

        def _walk(obj: object, path: str) -> None:
            if isinstance(obj, ListConfig):
                listconfig_paths.append(path or "<root>")
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    next_path = f"{path}.{key}" if path else str(key)
                    _walk(value, next_path)
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    next_path = f"{path}[{idx}]" if path else f"[{idx}]"
                    _walk(value, next_path)

        _walk(state, "")
        if listconfig_paths:
            print(f"-->\tOptimizer state contains ListConfig at: {', '.join(listconfig_paths)}")
        else:
            print("-->\tOptimizer state contains no ListConfig entries.")

    def _instantiate_scheduler(self, optimizer: Optimizer) -> object:
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
        self._scheduler_requires_metric = isinstance(scheduler, ReduceLROnPlateau)
        return scheduler

    def _maybe_step_scheduler(
        self,
        train_loss: float,
        val_loss: float | None,
    ) -> None:
        if self.scheduler is None:
            raise RuntimeError("Scheduler is required but not available.")

        metric = val_loss if val_loss is not None else train_loss
        if self._scheduler_requires_metric:
            if metric is None:
                raise ValueError("Scheduler requires a metric value but none was provided.")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

        current_lrs = tuple(self._current_learning_rates())
        if current_lrs:
            print(f"--> Current learning rate(s) after scheduler step: {current_lrs}")

    def _current_learning_rates(self) -> Iterable[float]:
        if self.optimizer is None:
            return ()
        return (group.get("lr") for group in self.optimizer.param_groups)

    def _run_validation_epoch(
        self,
        dataloader: DataLoader | None,
        metrics: Sequence[MetricFn],
        metric_names: Sequence[str],
    ) -> tuple[float | None, Dict[str, float]]:
        """
        Iterate over the validation dataloader and return aggregated loss plus optional metrics.
        """
        if dataloader is None:
            return None, {}

        total = 0.0
        steps = 0
        metric_totals = [0.0 for _ in metrics]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    inputs, targets = self._unpack_batch(batch)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    total += loss.item()
                    steps += 1
                    for idx, metric_fn in enumerate(metrics):
                        metric_value = metric_fn(outputs, targets)
                        metric_totals[idx] += self._as_float(metric_value)
        finally:
            self.model.train(was_training)

        if steps == 0:
            raise ValueError("Validation dataloader produced no batches.")

        mean_loss = total / steps
        metric_means = [total_metric / steps for total_metric in metric_totals]
        metric_dict: Dict[str, float] = {}
        if metric_names:
            metric_dict = {name: value for name, value in zip(metric_names, metric_means)}
        return mean_loss, metric_dict

    @staticmethod
    def _prepare_destination(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_float(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _metric_name(metric: MetricFn, idx: int) -> str:
        name = getattr(metric, "__name__", None)
        if name:
            return name
        func = getattr(metric, "func", None)
        if func is not None:
            func_name = getattr(func, "__name__", None)
            if func_name:
                return func_name
        return f"metric_{idx}"
