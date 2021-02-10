import colorednoise as cn

from abc import ABC, abstractmethod

from .lattice import Lattice


class Map(ABC):

    def __init__(self, lattice: Lattice = None):
        self.lattice = None
        if lattice:
            self.apply(lattice)

    @abstractmethod
    def apply(self, lattice: Lattice) -> Lattice:
        self.lattice = lattice
        # self.lattice.min_max_normalizer()
        return self.lattice

    @abstractmethod
    def __str__(self) -> str:
        pass


class LogisticMap(Map):

    def __init__(self, rho: float = 3.9, lattice: Lattice = None):
        self.rho = rho
        if lattice:
            super().__init__(lattice)

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float):
        assert 0 <= value <= 4, ('The logistic map rho value '
                                 'must be between 0 and 4.')
        self._rho = value

    def apply(self, lattice: Lattice) -> Lattice:
        updated_lattice = Lattice(lattice.shape)
        updated_lattice.u = self.rho * lattice.u * (1 - lattice.u)

        return super().apply(updated_lattice)

    def __str__(self) -> str:
        return (f'Logistic Map: '
                rf'$\rho={self.rho}$')


class NoiseMap(Map):

    def __init__(self, alpha: float = 1.0, noise_contribution: float = 0.1,
                 lattice: Lattice = None):
        self.alpha = alpha
        self.noise_contribution = noise_contribution
        if Lattice:
            super().__init__(lattice)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        assert 0 <= value <= 2, ('The noise map alpha value '
                                 'must be between 0 and 2.')
        self._alpha = value

    def apply(self, lattice: Lattice) -> Lattice:
        noise_lattice = Lattice(lattice.shape)
        noise_lattice.u = cn.powerlaw_psd_gaussian(self.alpha,
                                                   (lattice.shape[0],
                                                    lattice.shape[1]))
        noise_lattice.min_max_normalizer()

        updated_lattice = Lattice(lattice.shape)
        updated_lattice.u = (noise_lattice.u * self.noise_contribution) \
            + lattice.u
        updated_lattice.min_max_normalizer()

        return super().apply(updated_lattice)

    def __str__(self) -> str:
        return rf'$1/f$ Noise Map: $\alpha={self.alpha}$'
