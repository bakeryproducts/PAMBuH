import numpy as np
from matplotlib.path import Path
from typing import Tuple


def polyg_to_mask(polyg: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
    """Convert polygon to mask where 1 - polygon and 0 - no polygon.
    """

    xsize, ysize = wh[0] // 2, wh[1] // 2
    x, y = np.meshgrid(np.arange(-xsize + 1, xsize + 1), np.arange(-ysize + 1, ysize + 1))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(polyg)
    grid = path.contains_points(points)
    grid = grid.reshape((2 * ysize, 2 * xsize))
    return np.array(grid, dtype=int)
