from pathlib import Path

EXAMPLE_PREFIX = "example_"
EPOCH_PREFIX   = "epoch_"

def parse_example_epoch(path: Path) -> tuple[int, int]:
    """Reverse‑engineer (example_id, epoch_no) from any descendant path."""
    try:
        ex = next(p for p in path.parts if p.startswith(EXAMPLE_PREFIX))
        ep = next(p for p in path.parts if p.startswith(EPOCH_PREFIX))
        return int(ex.removeprefix(EXAMPLE_PREFIX)) - 1, int(ep.removeprefix(EPOCH_PREFIX))
    except StopIteration as e:
        raise ValueError(f"{path} is not inside a recognised run directory") from e

def get_plot_base_path(model_dir: Path, example_id: int, epoch_no: int, series_i: int|None = None) -> Path:
    return model_dir / f"{EXAMPLE_PREFIX}{example_id+1}" / f"{EPOCH_PREFIX}{epoch_no}"

def get_plot_data_path(model_dir: Path, example_id: int, epoch_no: int, series_i: int|None = None) -> Path:
    return get_plot_base_path(model_dir, example_id, epoch_no) / "data"

def get_plot_partials_path(model_dir: Path, example_id: int, epoch_no: int, series_i: int|None = None) -> Path:
    return get_plot_base_path(model_dir, example_id, epoch_no) / "partials"

def get_plot_snapshot_path(model_dir: Path, example_id: int, epoch_no: int, series_i: int) -> Path:
    return get_plot_base_path(model_dir, example_id, epoch_no) / f'timesep_{series_i}'

def get_plot_snapshot_partials_path(model_dir: Path, example_id: int, epoch_no: int, series_i: int) -> Path:
    return get_plot_snapshot_path(model_dir, example_id, epoch_no, series_i) / "partials"