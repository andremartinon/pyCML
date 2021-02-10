import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple

from .lattice import Lattice


class InitialCondition(ABC):

    def __init__(self, shape: Tuple[int, int] = (40, 40)):
        self.lattice = Lattice(shape)
        self.generate()

    @abstractmethod
    def generate(self) -> Lattice:
        return self.lattice

    @abstractmethod
    def __str__(self) -> str:
        pass


class RandomInitialCondition(InitialCondition):

    def __init__(self, shape: Tuple[int, int] = (40, 40), seed: int = None):
        self.seed = seed
        super().__init__(shape)

    def generate(self) -> Lattice:
        if self.seed is not None:
            np.random.seed(self.seed)
        self.lattice.u = np.random.rand(*self.lattice.shape)
        return super().generate()

    def __str__(self) -> str:
        return 'Random Initial Condition'


class GaussianInitialCondition(InitialCondition):

    def __init__(self,
                 shape: Tuple[int, int] = (40, 40),
                 mu: float = 0.5,
                 sigma: float = 0.15):
        self.mu = mu
        self.sigma = sigma
        super().__init__(shape)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        assert 0 <= value <= 1, ('The mean (mu) value '
                                 'must be between 0 and 1.')
        self._mu = value

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        assert 0 <= value <= 1, ('The variance (sigma) value '
                                 'must be between 0 and 1.')
        self._sigma = value

    def generate(self) -> Lattice:
        x, y = np.meshgrid(np.linspace(0, 1, self.lattice.shape[0]),
                           np.linspace(0, 1, self.lattice.shape[1]))

        self.lattice.u = np.exp(-((x - self.mu) ** 2 + (y - self.mu) ** 2) /
                                (2.0 * self.sigma ** 2))
        return super().generate()

    def __str__(self) -> str:
        return ('Gaussian Initial Condition: '
                rf'$\mu = {self.mu}, \sigma = {self.sigma}$')
