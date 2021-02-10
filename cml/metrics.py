import gc
import multiprocessing
import numpy as np

from abc import ABC, abstractmethod
from concurrent import futures
from gpa import GPA
from pathlib import Path
from scipy.stats import describe, stats
from skimage.measure import label
from typing import List, Tuple, Sequence

from .evolution import Evolution
from .lattice import Lattice
from .plot import plot_statistical_moments, plot_gradient, plot_entropy,\
    plot_euler_characteristic, plot_gpa, plot_four_gradients
from .animation import animate_gradient, create_gradient_animation


class Metric(ABC):

    def __init__(self, lattices: Sequence[Lattice] = None,
                 metric_func=None,
                 shape: Tuple[int, int] = None,
                 dtype: str = 'float64',
                 output_dir: Path = Path('/tmp/cml/'),
                 dataset_name: str = None):
        self.lattices = lattices
        self.metrics = np.ndarray(shape, dtype=dtype)
        self.metric_func = metric_func
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.file_name = self.output_dir / self.dataset_name
        self.columns = ''

    def measure(self, parallel: bool = True, *args, **kwargs):
        if parallel:
            workers = multiprocessing.cpu_count()
            with futures.ProcessPoolExecutor(workers) as executor:
                to_do = []
                for i, lattice in enumerate(self.lattices):
                    future = executor.submit(self.metric_func, lattice.u,
                                             *args, **kwargs)
                    future.sid = i
                    to_do.append(future)

                for future in futures.as_completed(to_do):
                    self.metrics[future.sid] = future.result()
        else:
            for i, lattice in enumerate(self.lattices):
                self.metrics[i] = self.metric_func(lattice.u, *args, **kwargs)

    @abstractmethod
    def plot(self):
        pass

    def save(self):
        with open(self.file_name.with_suffix('.txt'), 'w') as outfile:
            outfile.write('# Rows represents snapshots\n')
            outfile.write(f'# Data shape: {self.metrics.shape}\n')
            outfile.write(f'{self.columns}\n')
            np.savetxt(outfile, self.metrics, delimiter=',')


class StatisticalMomentsMetric(Metric):

    def __init__(self, evolution: Evolution = None):
        shape = (evolution.iterations+1, 3)
        super().__init__(evolution.snapshots,
                         StatisticalMomentsMetric._work,
                         shape,
                         'float64',
                         evolution.output_dir,
                         evolution.dataset_name + '_statistical_moments')
        self.columns = 'skewness,kurtosis,variance'

    @staticmethod
    def _work(lattice: np.ndarray):
        moments: stats.DescribeResult = describe(lattice, axis=None)
        return np.array([moments.skewness,
                         moments.kurtosis,
                         moments.variance])

    def plot(self, show: bool = True):
        plot_statistical_moments(self.metrics[:, 0],
                                 self.metrics[:, 1],
                                 self.metrics[:, 2],
                                 show=show,
                                 file_name=self.file_name.with_suffix('.png'))


class GradientMetric(Metric):

    def __init__(self, evolution: Evolution = None):
        shape = (evolution.iterations+1, 4, evolution.cml.grid_size,
                 evolution.cml.grid_size)

        super().__init__(evolution.snapshots,
                         GradientMetric._work,
                         shape,
                         'float64',
                         evolution.output_dir,
                         evolution.dataset_name + '_gradient')

    @staticmethod
    def _work(lattice: np.ndarray):
        v, u = np.gradient(lattice)
        modulus = np.sqrt(u**2 + v**2)
        phases = np.arctan2(v, u)

        return np.array([u, v, modulus, phases])

    def plot(self, index: int = 0, step: int = 2, show: bool = True):
        u = self.metrics[index, 0]
        v = self.metrics[index, 1]
        file_name = self.file_name.name + f'_snapshots{index}'
        file_name = self.file_name.with_name(file_name).with_suffix('.png')

        plot_gradient(u, v, step, show, file_name=file_name)

    def plot_four_gradients(self, step: int = 2, show: bool = True):
        size = len(self.metrics)
        indexes = (
            0,
            int(size / 4),
            size - int(size / 4),
            size - 1
        )

        plot_four_gradients([self.metrics[indexes[0], 0],
                             self.metrics[indexes[1], 0],
                             self.metrics[indexes[2], 0],
                             self.metrics[indexes[3], 0]],
                            [self.metrics[indexes[0], 1],
                             self.metrics[indexes[1], 1],
                             self.metrics[indexes[2], 1],
                             self.metrics[indexes[3], 1]],
                            indexes,
                            step,
                            show=show,
                            file_name=self.file_name.with_suffix('.png'))

    def animate(self, fps: int = 4, step: int = 2, show: bool = True):
        file_name = self.file_name.with_suffix('.mp4')
        if show:
            animate_gradient(self.metrics[:, 0],
                             self.metrics[:, 1],
                             fps=fps,
                             step=step,
                             show=show,
                             file_name=file_name)
        else:
            create_gradient_animation(self.metrics[:, 0],
                                      self.metrics[:, 1],
                                      fps=fps,
                                      step=step,
                                      file_name=file_name)

    def save(self):
        for data, name in [(self.metrics[:, 0], 'u'),
                           (self.metrics[:, 1], 'v'),
                           (self.metrics[:, 2], 'modulus'),
                           (self.metrics[:, 3], 'phases')]:
            file_name = self.file_name.name + f'_{name}'
            file_name = self.file_name.with_name(file_name).with_suffix('.txt')
            with open(file_name, 'w') as outfile:
                outfile.write('# Data shape: {}\n'.format(data.shape))
                for i, snapshot in enumerate(data):
                    outfile.write(f'# Snapshot {i}\n')
                    np.savetxt(outfile, snapshot, fmt='%.4f', delimiter=',')


class EntropyMetric(Metric):

    def __init__(self, evolution: Evolution = None):
        shape = (evolution.iterations + 1, 3)
        super().__init__(evolution.snapshots,
                         EntropyMetric._work,
                         shape,
                         'float64',
                         evolution.output_dir,
                         evolution.dataset_name + '_entropy')

        self.columns = 'lattices,modulus,phases'

    @staticmethod
    def _entropy(img, bins=256):
        marg = np.histogramdd(np.ravel(img), bins=bins)[0] / img.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        return -np.sum(np.multiply(marg, np.log2(marg)))

    @staticmethod
    def _work(lattice: np.ndarray, bins: int = 256):
        u, v, modulus, phases = GradientMetric._work(lattice)

        return np.array([EntropyMetric._entropy(lattice, bins),
                         EntropyMetric._entropy(modulus, bins),
                         EntropyMetric._entropy(phases, bins)])

    def measure(self, bins: int = 256):
        super().measure(bins=bins)

    def plot(self, show: bool = True):
        lattices = self.metrics[:, 0]
        modulus = self.metrics[:, 1]
        phases = self.metrics[:, 2]

        plot_entropy(modulus,
                     phases,
                     lattices,
                     show=show,
                     file_name=self.file_name.with_suffix('.png'))


class EulerCharacteristicMetric(Metric):

    def __init__(self, evolution: Evolution = None):
        shape = (evolution.iterations + 1, 2)
        super().__init__(evolution.snapshots,
                         EulerCharacteristicMetric._work,
                         shape,
                         'float64',
                         evolution.output_dir,
                         evolution.dataset_name + '_euler_characteristic')
        self.columns = 'gamma,chi'

    @staticmethod
    def _chi(gamma: float, gamma_0: float) -> float:
        return (1.0 / (2.0 * np.pi)) * (gamma - gamma_0) * np.exp(
            -0.5 * (gamma - gamma_0) ** 2)

    @staticmethod
    def _gamma(lattice: np.ndarray, threshold: float, connectivity: int):
        img = lattice.copy()
        img[img >= threshold] = 1.0
        img[img < threshold] = 0.0

        labels = label(img, connectivity=connectivity)
        labels_hist = np.histogram(labels,
                                   bins=labels.max(),
                                   range=(1, labels.max()))

        gamma = 0
        for size in labels_hist[0]:
            gamma = gamma + size/img.size

        return gamma / labels.max()

    @staticmethod
    def _work(lattice: np.ndarray, threshold: float, connectivity: int,
              gamma_0: float):

        gamma = EulerCharacteristicMetric._gamma(lattice,
                                                 threshold,
                                                 connectivity)

        chi = EulerCharacteristicMetric._chi(gamma, gamma_0)

        return np.array([gamma, chi])

    def measure(self, threshold: float = 0.68, connectivity: int = 1):

        gamma_0 = EulerCharacteristicMetric._gamma(self.lattices[0].u,
                                                   threshold,
                                                   connectivity)
        super().measure(threshold=threshold,
                        connectivity=connectivity,
                        gamma_0=gamma_0)

        self.metrics[0] = np.array([gamma_0,
                                    EulerCharacteristicMetric._chi(gamma_0, 0)])

    def plot(self, show: bool = True):
        gammas = self.metrics[:, 0]
        chi_gammas = self.metrics[:, 1]

        plot_euler_characteristic(gammas,
                                  chi_gammas,
                                  show=show,
                                  file_name=self.file_name.with_suffix('.png'))


class GPAMetric(Metric):

    def __init__(self, evolution: Evolution = None):
        shape = (evolution.iterations + 1, 4)
        super().__init__(evolution.snapshots,
                         GPAMetric._work,
                         shape,
                         'float64',
                         evolution.output_dir,
                         evolution.dataset_name + '_gpa')
        self.columns = 'G1,G2,G3,G4'

    @staticmethod
    def _work(lattice: np.ndarray, tolerance: float):
        g1, g2, g3, g4 = np.nan, np.nan, np.nan, np.nan

        ga = GPA(lattice, modulus_tolerance=tolerance,
                 phases_tolerance=tolerance)
        g2 = ga.evaluate()

        return np.array([g1, g2, g3, g4])

    def measure(self, tolerance: float = 0.01):
        super().measure(tolerance=tolerance)

    def plot(self, show: bool = True):
        g1 = self.metrics[:, 0].real
        g2 = self.metrics[:, 1].real
        g3 = self.metrics[:, 2].real
        g4 = self.metrics[:, 3]

        plot_gpa(g1, g2, g3, g4, show=show,
                 file_name=self.file_name.with_suffix('.png'))
