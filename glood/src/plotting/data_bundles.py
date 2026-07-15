import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class _PlottableBase:
    id: int
    epoch: int
    base_dir: Path
    series_i: int | None



@dataclass
class PlottableProcessed(_PlottableBase):
    title_abbr: str
    title_extended: str
    units: str
    cmap: str
    data: List[np.ndarray]
    group: str
    group_dominant: bool = False
    extremes: Sequence[float | None] = (None, None)
    needs_colorbar_in_compound_plot: bool = False
    stride: int = 5


@dataclass
class PlottableInput(_PlottableBase):
    pred: np.ndarray
    true: np.ndarray
    matnum: np.ndarray | None = None

    def build_processed (
        self,
        title_abbr: str,
        title_extended: str,
        units: str,
        cmap: str,
        group: str,
        group_dominant: bool,
        data: List[np.ndarray]
    ) -> PlottableProcessed:
        return PlottableProcessed (
            id             = self.id,
            epoch          = self.epoch,
            base_dir       = self.base_dir,
            title_abbr     = title_abbr,
            title_extended = title_extended,
            units          = units,
            cmap           = cmap,
            group          = group,
            group_dominant = group_dominant,
            series_i       = self.series_i,
            extremes       = (None, None),
            data           = data,
        )
