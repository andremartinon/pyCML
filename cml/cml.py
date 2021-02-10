from .initial_conditions import InitialCondition, RandomInitialCondition
from .boundary_conditions import BoundaryCondition, SteadyBoundaryCondition
from .maps import Map, LogisticMap
from .couplings import Coupling, FourNeighborCoupling


class CML:
    lattice = None

    def __init__(self,
                 ic: InitialCondition = RandomInitialCondition(),
                 bc: BoundaryCondition = SteadyBoundaryCondition(),
                 mapping: Map = LogisticMap(),
                 coupling: Coupling = FourNeighborCoupling(),
                 grid_size: int = 40):
        self._ic = ic
        self._bc = bc
        self._mapping = mapping
        self._coupling = coupling
        self._grid_size = grid_size

        self.reset()

    @property
    def ic(self) -> InitialCondition:
        return self._ic

    @ic.setter
    def ic(self, value: InitialCondition):
        self._ic = value
        self.reset()

    @property
    def bc(self) -> BoundaryCondition:
        return self._bc

    @bc.setter
    def bc(self, value: BoundaryCondition):
        self._bc = value
        self.reset()

    @property
    def mapping(self) -> Map:
        return self._mapping

    @mapping.setter
    def mapping(self, value: Map):
        self._mapping = value
        self.reset()

    @property
    def coupling(self) -> Coupling:
        return self._coupling

    @coupling.setter
    def coupling(self, value: Coupling):
        self._coupling = value
        self.reset()

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: int):
        self._grid_size = value
        self.reset()

    def reset(self):
        self.ic.lattice.shape = (self.grid_size,) * 2
        self.lattice = self.ic.generate()

    def update(self):
        self.lattice = self.coupling.apply(
            self.bc.apply(
                self.mapping.apply(self.lattice)), with_boundaries=False)
