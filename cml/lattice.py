import numpy as np

from typing import Tuple


class Lattice:

    def __init__(self, shape: Tuple[int, int]):
        self.dimensions = 2
        self.shape = shape
        self.u = np.ndarray(shape)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: Tuple[int, int]):
        self.dimensions: int = len(shape)
        self._shape = shape

    @property
    def u(self) -> np.ndarray:
        return self._u

    @u.setter
    def u(self, lattice: np.ndarray):
        if lattice.ndim != self.dimensions:
            raise Exception(
                f'The lattice must have {self.dimensions} dimensions')

        if lattice.shape != self.shape:
            raise Exception(f'All dimension must have the same size')

        self._u = lattice

    def min_max_normalizer(self):
        self.u = (self.u - np.min(self.u)) / np.ptp(self.u)

    def __str__(self) -> str:
        return (f'Lattice with {self.dimensions} dimensions '
                f'and shape {self.shape}')
