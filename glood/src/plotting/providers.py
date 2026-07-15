# providers.py
from pathlib import Path
import numpy as np

from glood.src.plotting.data_bundles import PlottableInput
import glood.src.plotting.plot_paths as pp


def _load_optional_matnum(data_dir: Path) -> np.ndarray | None:
    matnum_path = data_dir / "matnum.npy"
    if not matnum_path.exists():
        return None
    return np.load(matnum_path, allow_pickle=True)


class ProviderDataset:
    """Load .pt shards off disks into a torch dataset."""

    def __init__(self, input_dir: Path, max_iterations, parser):
        self.input_dir = input_dir
        self.max_iterations = max_iterations
        self.dataset = parser.pt2dataset(self.input_dir)

    def __iter__(self):
        for i, example in enumerate(self.dataset):
            if i >= self.max_iterations:
                return
            yield PlottableInput(
                id       = i,
                epoch    = 999999,
                base_dir = self.input_dir,
                series_i = None,
                pred     = self._to_plot_array(example[0]),
                true     = self._to_plot_array(example[1]),
                matnum   = None,
            )

    @staticmethod
    def _to_plot_array(tensor):
        array = tensor.numpy()
        if array.ndim == 3:
            array = np.expand_dims(array, axis=0)
        return array


class ProviderFileEpoch:
    """Load .npy files off disk for the latest available epoch."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)

        entries: list[tuple[int, int, Path]] = []
        for data_dir in self.input_dir.glob(f"{pp.EXAMPLE_PREFIX}*/{pp.EPOCH_PREFIX}*/data"):
            example_id, epoch_no = pp.parse_example_epoch(data_dir)
            entries.append((example_id, epoch_no, data_dir))

        if not entries:
            self.latest_epoch: int | None = None
            self.data_dirs: list[Path] = []
            return

        latest_epoch = max(epoch_no for _, epoch_no, _ in entries)
        self.latest_epoch = latest_epoch
        self.data_dirs = [
            data_dir
            for example_id, epoch_no, data_dir in sorted(entries, key=lambda item: item[0])
            if epoch_no == latest_epoch
        ]

    def __iter__(self):
        if not self.data_dirs:
            raise RuntimeError(f"No plot data found under {self.input_dir}.")

        for data_dir in self.data_dirs:
            example_id, epoch_no = pp.parse_example_epoch(data_dir)
            yield PlottableInput(
                id       = example_id,
                epoch    = epoch_no,
                base_dir = self.input_dir,
                series_i = None,
                pred     = self._to_plot_array(np.load(data_dir / "pred.npy", allow_pickle=True)),
                true     = self._to_plot_array(np.load(data_dir / "true.npy", allow_pickle=True)),
                matnum   = _load_optional_matnum(data_dir),
            )

    @staticmethod
    def _to_plot_array(array: np.ndarray) -> np.ndarray:
        if array.ndim == 3:
            array = np.expand_dims(array, axis=0)
        return array


class ProviderFileGeneric:
    """Load .npy files off disk."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)

        # Find every “…/example_*/epoch_*/data” folder
        self.data_dirs = sorted(
            self.input_dir.glob(f"{pp.EXAMPLE_PREFIX}*/{pp.EPOCH_PREFIX}*/data"),
            key=lambda p: pp.parse_example_epoch(p)[:2]   # (example_id, epoch_no)
        )

    def __iter__(self):
        for data_dir in self.data_dirs:
            example_id, epoch_no = pp.parse_example_epoch(data_dir)
            yield PlottableInput(
                id       = example_id,
                epoch    = epoch_no,
                base_dir = self.input_dir,
                series_i = None,
                pred     = self._to_plot_array(np.load(data_dir / "pred.npy", allow_pickle=True)),
                true     = self._to_plot_array(np.load(data_dir / "true.npy", allow_pickle=True)),
                matnum   = _load_optional_matnum(data_dir),
            )

    @staticmethod
    def _to_plot_array(array: np.ndarray) -> np.ndarray:
        """Ensure arrays have a time/series axis as the first dimension.

        If a saved array represents a single timestep it will commonly be
        3D (H, W, C). Other code in the plotting pipeline expects a leading
        series axis, so expand that case to (1, H, W, C).
        """
        if array.ndim == 3:
            array = np.expand_dims(array, axis=0)
        return array
