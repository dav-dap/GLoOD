import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Optional


class PlottingHandle:
    def __init__(self, ni: int = 1, nj: int = 1, remove_title: bool = False):
        self.X: Optional[NDArray] = None
        self.Y: Optional[NDArray] = None
        self.pc: NDArray = np.full((ni, nj), None, dtype=object)
        self.qv: NDArray = np.full((ni, nj), None, dtype=object)
        self.cbar: NDArray = np.full((ni, nj), None, dtype=object)
        self.remove_title = remove_title

        self.fig, self.ax = plt.subplots (
            ni,
            nj,
            figsize=(4.5*nj, 4.5*ni),
            squeeze=False,
            constrained_layout=False,
        )
        for a in self.ax.ravel():
            a.set_box_aspect(1)
            a.set_adjustable("box")
            a.set_aspect("equal")