import numpy as np
import json
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset

import glood.src.utils.env as env
import glood.src.datasets.sharded_dataset as sharded_dataset



class ParseBase(ABC):
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        len_seq: int,
        shuffle: bool,
        start_index: int = 0,
    ) -> None:
        # Store only; do not alter len_seq here.
        self.H, self.W, self.C = H, W, C
        self.len_seq = len_seq
        self.shuffle = shuffle
        self.start_index = start_index
        if self.start_index < 0:
            raise ValueError("start_index must be a non-negative integer.")

    @abstractmethod
    def provide_dataset_type(self):
        ...

    @abstractmethod
    def txt2pt(self, file_path_list: List[Path], examples_per_shard: int) -> None:
        ...

    def _update_pt_index(self, index_path: Path, shard_path: Path, *, L: int, H: int, W: int, C: int, dtype: str, N: int) -> None:
        """
        Create/update a tiny JSON index so readers can discover per-shard lengths (N)
        without loading tensors.

        Format:
        {
          "shape": [L, H, W, C],
          "dtype": "float32" | "float16" | "bfloat16" | ...,
          "files": [{"path": "train_000000.pt", "length": 1234}, ...]
        }
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
            data = json.loads(index_path.read_text())
            # sanity: append-only compatible with existing shape/dtype
            if data.get("shape") != [int(L), int(H), int(W), int(C)]:
                raise ValueError(f"{index_path}: shape mismatch; existing {data.get('shape')} vs new {[L,H,W,C]}")
            if data.get("dtype") != dtype:
                raise ValueError(f"{index_path}: dtype mismatch; existing {data.get('dtype')} vs new {dtype}")
        else:
            data = {"shape": [int(L), int(H), int(W), int(C)], "dtype": dtype, "files": []}

        # store path relative to index for portability
        rel = str(Path(shard_path).relative_to(index_path.parent))

        by_path = {f["path"]: int(f["length"]) for f in data["files"]}
        by_path[rel] = int(N)  # insert/overwrite

        data["files"] = [{"path": p, "length": n} for p, n in sorted(by_path.items())]
        index_path.write_text(json.dumps(data, indent=2))


    def pt2dataset(self, input_dir: Path) -> Dataset:
        """
        Convenience helper: glob shards in `input_dir` and return a PTShardedNextStepDataset.
        """
        return self.provide_dataset_type()(str(input_dir), self.shuffle)



    def txt2pt_base(self, file_path_list: List[Path], examples_per_shard: int, one_seq_only: bool) -> None:
        """
        Read ONLY the first L lines from each TXT file and append that one example if one_seq_only is True.
        Raise if a file has fewer than L (non-empty) lines. Flush as soon as the
        buffer reaches `examples_per_shard` examples into train_XXXXXX.pt shards.
        """

        L, H, W, C = int(self.len_seq), int(self.H), int(self.W), int(self.C)
        start_index = int(self.start_index)
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be a positive integer.")

        out_dir = Path(env.out()) / "dataset"
        out_dir.mkdir(parents=True, exist_ok=True)

        buf: List[np.ndarray] = []  # (L, H, W, C) sequences
        shard_idx = 0
        total_examples = 0
        expected_per_line = H * W * C

        def flush() -> None:
            nonlocal buf, shard_idx, total_examples
            if not buf:
                return
            x = np.asarray(buf, dtype=np.float32)   # [N, L, H, W, C]
            t = torch.from_numpy(x)                 # CPU float32
            shard_file = out_dir / f"train_{shard_idx:06d}.pt"
            torch.save({"data": t}, shard_file)

            self._update_pt_index(
                index_path=out_dir / "index.json",
                shard_path=shard_file,
                L=L, H=H, W=W, C=C,
                dtype=str(t.dtype).replace("torch.", ""),  # e.g. "float32"
                N=t.shape[0],
            )

            total_examples += t.shape[0]
            buf.clear()
            shard_idx += 1

        for file_path in file_path_list:
            frames: List[np.ndarray] = []
            with file_path.open("r") as f:
                nonempty_idx = 0
                for line in f:
                    s = line.strip()
                    if not s:
                        continue  # ignore blank lines
                    if nonempty_idx < start_index:
                        nonempty_idx += 1
                        continue
                    arr = np.asarray(s.split(), dtype=np.float32)
                    if arr.size != expected_per_line:
                        raise ValueError(
                            f"{file_path}: expected {expected_per_line} floats per line, got {arr.size}"
                        )
                    if one_seq_only:
                        frame = arr.reshape(H, W, C).transpose(1, 0, 2)  # (W, H, C)
                    else:
                        frame = np.expand_dims(arr.reshape(H, W, C).transpose(1, 0, 2), axis=0)
                    frames.append(frame)
                    nonempty_idx += 1
                    if one_seq_only and len(frames) == L:
                        break

            if one_seq_only and len(frames) < L:
                raise ValueError(
                    f"{file_path}: only {len(frames)} usable non-empty lines found after skipping "
                    f"{start_index}, need {L}"
                )

            if one_seq_only:
                buf.append(np.stack(frames, axis=0))  # (L, H, W, C)
            else:
                buf += frames

            if len(buf) >= examples_per_shard:
                flush()

        if buf:
            flush()

        if total_examples == 0:
            raise ValueError(f"No complete sequences of length {L} found in the provided files.")
        return None


    def txt2pt_base_with_matnum(
        self,
        file_path_list: List[Path],
        examples_per_shard: int,
    ) -> None:
        """
        Parse series TXT files that include an extra matnum channel.

        Expected per-line shape is (H, W, 4) storing (p, ux, uy, matnum). The
        output shards store:
        - data: float32, shape (N, L, H, W, 3)
        - matnum: int32, shape (N, 1, H, W, 1)
        """

        L, H, W = int(self.len_seq), int(self.H), int(self.W)
        start_index = int(self.start_index)
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be a positive integer.")

        out_dir = Path(env.out()) / "dataset"
        out_dir.mkdir(parents=True, exist_ok=True)

        buf_data: List[np.ndarray] = []  # (L, H, W, 3) sequences
        buf_mat: List[np.ndarray] = []   # (1, H, W, 1) matnum per example
        shard_idx = 0
        total_examples = 0
        expected_per_line = H * W * 4

        def flush() -> None:
            nonlocal buf_data, buf_mat, shard_idx, total_examples
            if not buf_data:
                return
            x = np.asarray(buf_data, dtype=np.float32)  # [N, L, H, W, 3]
            m = np.asarray(buf_mat, dtype=np.int32)     # [N, 1, H, W, 1]
            t = torch.from_numpy(x)                     # CPU float32
            t_mat = torch.from_numpy(m)                 # CPU int32
            shard_file = out_dir / f"train_{shard_idx:06d}.pt"
            torch.save({"data": t, "matnum": t_mat}, shard_file)

            self._update_pt_index(
                index_path=out_dir / "index.json",
                shard_path=shard_file,
                L=L, H=H, W=W, C=3,
                dtype=str(t.dtype).replace("torch.", ""),
                N=t.shape[0],
            )

            total_examples += t.shape[0]
            buf_data.clear()
            buf_mat.clear()
            shard_idx += 1

        for file_path in file_path_list:
            frames: List[np.ndarray] = []
            mat0: Optional[np.ndarray] = None
            with file_path.open("r") as f:
                nonempty_idx = 0
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if nonempty_idx < start_index:
                        nonempty_idx += 1
                        continue
                    arr = np.asarray(s.split(), dtype=np.float32)
                    if arr.size != expected_per_line:
                        raise ValueError(
                            f"{file_path}: expected {expected_per_line} floats per line, got {arr.size}"
                        )
                    frame = arr.reshape(H, W, 4).transpose(1, 0, 2)  # (W, H, 4)
                    data_frame = frame[..., :3]  # (W, H, 3)
                    mat_frame = frame[..., 3:4]  # (W, H, 1)
                    frames.append(data_frame)
                    if mat0 is None:
                        mat0 = mat_frame
                    else:
                        if not np.array_equal(mat_frame, mat0):
                            raise ValueError(
                                f"{file_path}: matnum varies across timesteps; expected constant values."
                            )
                    nonempty_idx += 1
                    if len(frames) == L:
                        break

            if len(frames) < L:
                raise ValueError(
                    f"{file_path}: only {len(frames)} usable non-empty lines found after skipping "
                    f"{start_index}, need {L}"
                )

            buf_data.append(np.stack(frames, axis=0))  # (L, H, W, 3)
            if mat0 is None:
                raise ValueError(f"{file_path}: no matnum values found.")
            buf_mat.append(np.expand_dims(mat0.astype(np.int32), axis=0))  # (1, H, W, 1)

            if len(buf_data) >= examples_per_shard:
                flush()

        if buf_data:
            flush()

        if total_examples == 0:
            raise ValueError(f"No complete sequences of length {L} found in the provided files.")
        return None


class ParseSeries(ParseBase):
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        len_seq: int,
        shuffle: bool,
        start_index: int = 0,
    ) -> None:
        if len_seq <= 0:
            raise ValueError("len_seq must be a positive integer.")
        super().__init__(H, W, C, len_seq, shuffle, start_index=start_index)

    def provide_dataset_type(self):
        return sharded_dataset.ShardedDatasetSeries

    def txt2pt(self, file_path_list: List[Path], examples_per_shard: int) -> None:
        self.txt2pt_base(file_path_list, examples_per_shard, True)


class ParseSingle(ParseBase):
    def __init__(self, H: int, W: int, C: int, shuffle: bool) -> None:
        super().__init__(H, W, C, 1, shuffle)

    def provide_dataset_type(self):
        return sharded_dataset.ShardedDatasetSingle

    def txt2pt(self, file_path_list: List[Path], examples_per_shard: int) -> None:
        self.txt2pt_base(file_path_list, examples_per_shard, False)


class ParseSeriesWithMatnum(ParseBase):
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        len_seq: int,
        shuffle: bool,
        start_index: int = 0,
    ) -> None:
        if len_seq <= 0:
            raise ValueError("len_seq must be a positive integer.")
        if C != 4:
            raise ValueError("ParseSeriesWithMatnum expects C=4 (p, ux, uy, matnum).")
        super().__init__(H, W, C, len_seq, shuffle, start_index=start_index)

    def provide_dataset_type(self):
        return sharded_dataset.ShardedDatasetSeries

    def txt2pt(self, file_path_list: List[Path], examples_per_shard: int) -> None:
        self.txt2pt_base_with_matnum(file_path_list, examples_per_shard)
