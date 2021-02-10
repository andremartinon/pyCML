from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from scipy.signal import convolve2d

from .lattice import Lattice


class Coupling(ABC):

    def __init__(self, lattice: Lattice = None):
        self._kernel = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]])
        self._neighbors = 4
        self._e = 0.1
        self._lattice = None
        self._with_boundaries = True

        if lattice:
            self.apply(lattice)

    @property
    def e(self) -> float:
        return self._e

    @e.setter
    def e(self, value: float):
        self._e = value

    @property
    def neighbors(self) -> int:
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value: int):
        self._neighbors = value

    @property
    def kernel(self) -> np.ndarray:
        return self._kernel

    @kernel.setter
    def kernel(self, array: np.ndarray):
        self._kernel = array

    @property
    def lattice(self) -> Lattice:
        if self._with_boundaries:
            return self._lattice
        else:
            lattice = Lattice(tuple(np.array(self._lattice.shape) - 2))
            lattice.u = self._lattice.u[1:-1, 1:-1]
            return lattice

    @lattice.setter
    def lattice(self, lattice: Lattice):
        self._lattice = lattice

    def apply(self, lattice: Lattice, with_boundaries=False) -> Lattice:
        self._with_boundaries = with_boundaries
        updated_lattice = Lattice(lattice.shape)
        updated_lattice.u = convolve2d(lattice.u,
                                       self.kernel,
                                       mode='same',
                                       boundary='fill')
        updated_lattice.u = updated_lattice.u / self.neighbors
        self.lattice = updated_lattice
        return self.lattice

    @abstractmethod
    def __str__(self) -> str:
        pass


class FourNeighborCoupling(Coupling):

    def __init__(self, coupling_constant: float = 0.1, lattice: Lattice = None):
        self.e = coupling_constant
        self.neighbors = 4
        self.kernel = np.array([[0, self.e, 0],
                               [self.e, 4 * (1 - self.e), self.e],
                               [0, self.e, 0]])
        if lattice:
            super().__init__(lattice)

    def __str__(self) -> str:
        return f'Four Neighbors Coupling: e={self.e}'
