from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import glob
import itertools
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info




class ShardedDatasetBase(IterableDataset, ABC):
    def __init__(
            self,
            shard_dir_or_shard_seq: str | Sequence[str],
            shuffle:
            bool = False
        ):
        if isinstance(shard_dir_or_shard_seq, Sequence) and not isinstance(
            shard_dir_or_shard_seq, (str, bytes)
        ):
            self.paths = list(shard_dir_or_shard_seq)
        else:
            shard_dir = str(shard_dir_or_shard_seq)
            self.paths = sorted(glob.glob(f"{shard_dir}/*.pt"))
        if not self.paths:
            raise ValueError("No dataset shards matched the provided input.")

        self.shuffle = bool(shuffle)

    def __iter__(self):
        info = get_worker_info()
        rng = None

        if self.shuffle:
            paths = list(self.paths)
            if info is None:
                rng = random.Random()
            else:
                # use the worker base seed so every worker sees the same shuffle order
                base_seed = info.seed - info.id
                rng = random.Random(base_seed)
            rng.shuffle(paths)
        else:
            paths = self.paths

        if info is None:
            my_paths = paths
        else:
            # Each worker consumes whole shards; stride assignment prevents overlap.
            my_paths = paths[info.id::info.num_workers]

        for p in my_paths:
            data = torch.load(p, map_location="cpu")["data"]  # [N, L, H, W, C]
            # yield each sample
            if self.shuffle:
                indices = list(range(data.size(0)))
                rng.shuffle(indices)
            else:
                indices = range(data.size(0))

            for i in indices:
                example = data[i].contiguous()
                yield self._produce_example(example)

    def split(
        self,
        val_fraction: float,
        *,
        shuffle: bool = True,
    ) -> tuple[ShardedDatasetBase, ShardedDatasetBase]:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1 (exclusive).")
        if len(self.paths) < 2:
            raise ValueError("At least two shards are required to perform a train/validation split.")

        indices = list(range(len(self.paths)))
        if shuffle:
            random.shuffle(indices)

        val_count = max(1, int(len(indices) * val_fraction))

        val_paths = [self.paths[i] for i in indices[:val_count]]
        train_paths = [self.paths[i] for i in indices[val_count:]]

        return (
            self.__class__(train_paths, shuffle=self.shuffle),
            self.__class__(val_paths, shuffle=self.shuffle),
        )

    @abstractmethod
    def _produce_example(self, example):
        ...


class ShardedDatasetSeries(ShardedDatasetBase):
    def __init__(self, shard_dir: str, shuffle: bool = False):
        super().__init__(shard_dir, shuffle=shuffle)

    def _produce_example(self, example):
        x = example[:-1]      # (L-1, H, W, C)
        y_true = example[1:]  # (L-1, H, W, C)
        return (x, y_true)


class ShardedDatasetSeriesWithMatnum(ShardedDatasetBase):
    def __init__(self, shard_dir: str, shuffle: bool = False):
        super().__init__(shard_dir, shuffle=shuffle)

    def __iter__(self):
        info = get_worker_info()
        rng = None

        if self.shuffle:
            paths = list(self.paths)
            if info is None:
                rng = random.Random()
            else:
                base_seed = info.seed - info.id
                rng = random.Random(base_seed)
            rng.shuffle(paths)
        else:
            paths = self.paths

        if info is None:
            my_paths = paths
        else:
            my_paths = paths[info.id::info.num_workers]

        for p in my_paths:
            shard = torch.load(p, map_location="cpu")
            data = shard["data"]          # [N, L, H, W, 3]
            matnum = shard.get("matnum")  # [N, 1, H, W, 1]
            if matnum is None:
                raise KeyError(f"{p} is missing 'matnum' tensor.")

            if self.shuffle:
                indices = list(range(data.size(0)))
                rng.shuffle(indices)
            else:
                indices = range(data.size(0))

            for i in indices:
                example = data[i].contiguous()
                mat = matnum[i].contiguous()
                yield self._produce_example(example, mat)

    def _produce_example(self, example, matnum):
        x = example[:-1]      # (L-1, H, W, 3)
        y_true = example[1:]  # (L-1, H, W, 3)
        return {"inputs": x, "targets": y_true, "matnum": matnum}


class ShardedDatasetSingle(ShardedDatasetBase):
    def __init__(self, shard_dir: str , shuffle: bool = False):
        super().__init__(shard_dir, shuffle=shuffle)

    def _produce_example(self, example):
        x = example.squeeze(0)  # remove length dim -> (H, W, C)
        return (x, x)


class ShardedDatasetKFoldBase(ShardedDatasetBase):
    """
    Deterministic shard-level k-fold splitter (no shuffling).

    The fold index is applied to the sorted shard list; every shard whose index
    satisfies idx % k == i becomes part of the validation set.
    """

    def __init__(
        self,
        shard_dir_or_shard_seq: str | Sequence[str],
        *,
        k: int,
        i: int,
        shuffle: bool = False,
    ):
        if shuffle:
            raise ValueError("K-fold datasets disable shuffling for deterministic splits.")
        if k < 2:
            raise ValueError("k must be at least 2 for k-fold splits.")

        # Enforce 0-based fold indices.
        if 0 <= i < k:
            fold_idx = i
        else:
            raise ValueError(f"Fold index i must be in [0, {k-1}].")

        # Normalize input to a deterministically ordered shard list.
        if isinstance(shard_dir_or_shard_seq, Sequence) and not isinstance(
            shard_dir_or_shard_seq, (str, bytes)
        ):
            shard_list = sorted(str(p) for p in shard_dir_or_shard_seq)
        else:
            shard_dir = str(shard_dir_or_shard_seq)
            shard_list = sorted(glob.glob(f"{shard_dir}/*.pt"))

        if not shard_list:
            raise ValueError("No dataset shards matched the provided input.")

        self.k = k
        self.fold_idx = fold_idx
        super().__init__(shard_list, shuffle=False)

    def split(
        self,
        val_fraction: float,
        *,
        shuffle: bool = True,
    ) -> tuple[ShardedDatasetKFoldBase, ShardedDatasetKFoldBase]:
        # val_fraction is ignored; fold assignment controls the split.
        # shuffle is ignored to keep shard allocation deterministic.

        grouped = itertools.groupby(
            enumerate(self.paths),
            key=lambda pair: pair[0] % self.k == self.fold_idx,
        )
        val_paths: list[str] = []
        train_paths: list[str] = []
        for is_val, group_iter in grouped:
            for _, path in group_iter:
                if is_val:
                    val_paths.append(path)
                else:
                    train_paths.append(path)

        if not train_paths:
            raise ValueError("K-fold split produced an empty training set.")
        if not val_paths:
            raise ValueError("K-fold split produced an empty validation set.")

        return (
            self.__class__(train_paths, k=self.k, i=self.fold_idx, shuffle=False),
            self.__class__(val_paths, k=self.k, i=self.fold_idx, shuffle=False),
        )


class ShardedDatasetSeriesKFold(ShardedDatasetKFoldBase):
    def __init__(self, shard_dir: str, *, k: int, i: int, shuffle: bool = False):
        super().__init__(shard_dir, k=k, i=i, shuffle=shuffle)

    def _produce_example(self, example):
        x = example[:-1]      # (L-1, H, W, C)
        y_true = example[1:]  # (L-1, H, W, C)
        return (x, y_true)


class ShardedDatasetSingleKFold(ShardedDatasetKFoldBase):
    def __init__(self, shard_dir: str, *, k: int, i: int, shuffle: bool = False):
        super().__init__(shard_dir, k=k, i=i, shuffle=shuffle)

    def _produce_example(self, example):
        x = example.squeeze(0)  # remove length dim -> (H, W, C)
        return (x, x)
