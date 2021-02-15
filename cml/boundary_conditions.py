import numpy as np

from abc import ABC, abstractmethod
from .lattice import Lattice


class BoundaryCondition(ABC):

    def __init__(self, lattice: Lattice = None):
        self.lattice = None
        if lattice:
            self.apply(lattice)

    @abstractmethod
    def apply(self, lattice: Lattice) -> Lattice:
        shape = tuple(np.array(lattice.shape) + 2)
        self.lattice = Lattice(shape)
        return self.lattice

    @abstractmethod
    def __str__(self) -> str:
        pass


class SteadyBoundaryCondition(BoundaryCondition):

    def __init__(self, value: float = 0.0, lattice: Lattice = None):
        self.constant_value = value
        if lattice:
            super().__init__(lattice)

    @property
    def constant_value(self) -> float:
        return self._constant_value

    @constant_value.setter
    def constant_value(self, value: float):
        assert 0 <= value <= 1, ('The steady boundary value '
                                 'must be between 0 and 1.')
        self._constant_value = value

    def apply(self, lattice: Lattice) -> Lattice:
        super().apply(lattice)
        u = lattice.u

        zeros = np.zeros((1, u.shape[0])).T + self.constant_value
        u = np.column_stack(
            (np.column_stack((zeros, u)), zeros))

        zeros = np.zeros((1, u.shape[1])) + self.constant_value
        u = np.row_stack(
            (np.row_stack((zeros, u)), zeros))

        self.lattice.u = u

        return self.lattice

    def __str__(self) -> str:
        return f'Steady Boundary Condition: value={self.constant_value}'


class ToroidalBoundaryCondition(BoundaryCondition):

    def __init__(self, lattice: Lattice = None):
        if lattice:
            super().__init__(lattice)

    def apply(self, lattice: Lattice) -> Lattice:
        super().apply(lattice)
        u = lattice.u

        u = np.column_stack(
            (np.column_stack((u[:, -1], u)), u[:, 0]))
        u = np.row_stack(
            (np.row_stack((u[-1, :], u)), u[0, :]))

        self.lattice.u = u

        return self.lattice

    def __str__(self) -> str:
        return f'Toroidal Boundary Condition'
