import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
from typing import List, Tuple

from .lattice import Lattice


def _get_im_config(grid_size: int):
    return {
        'interpolation': 'none',
        'aspect': 'equal',
        'cmap': plt.get_cmap('gray'),
        'vmin': 0.0,
        'vmax': 1.0,
        'extent': [1, grid_size,
                   grid_size,1]
    }


def show_plots(cml_lattices):
    for cml_lattice in cml_lattices:
        fig, _ = plot_lattice(cml_lattice.lattice, cml_lattice)
        plt.show()
        plt.close(fig)


def plot_lattice(lattice: Lattice, title: str, binary_threshold=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    img = lattice.u.copy()
    if binary_threshold:
        img[img >= binary_threshold] = 1.0
        img[img < binary_threshold] = 0.0

    grid_size = lattice.shape[0]
    im = ax.imshow(img, **_get_im_config(grid_size))
    ax.set_title(title)

    ticks = np.linspace(1, lattice.shape[0], 3, dtype='int64')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    fig.colorbar(im)

    return fig, ax


def plot_three_lattices(snapshots: List[Lattice],
                        indexes: Tuple[int, int, int],
                        show: bool = True,
                        file_name: Path = None):

    mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False

    fig = plt.figure(figsize=(11.7, 8.3))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 3),
                     axes_pad=0.25,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.25)

    grid_size = snapshots[indexes[0]].shape[0]
    ax = None
    im = None
    for i, ax in enumerate(grid):
        im = ax.imshow(snapshots[indexes[i]].u, **_get_im_config(grid_size))
        ax.set_title(f'Snapshot {indexes[i]}')

        ticks = np.linspace(1, grid_size, 3, dtype='int64')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_four_lattices(snapshots: List[Lattice],
                       indexes: Tuple[int, int, int, int],
                       show: bool = True,
                       file_name: Path = None,
                       binary_threshold = None,
                       dataset_name: str = ''):

    mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False

    fig = plt.figure(figsize=(11.7, 8.3))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 2),
                     axes_pad=0.28,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.25)

    grid_size = snapshots[indexes[0]].shape[0]
    ax = None
    im = None
    for i, ax in enumerate(grid):
        img = snapshots[indexes[i]].u.copy()
        if binary_threshold:
            img[img >= binary_threshold] = 1.0
            img[img < binary_threshold] = 0.0

        im = ax.imshow(img, **_get_im_config(grid_size))
        ax.set_title(f'Snapshot {indexes[i]}')

        ticks = np.linspace(1, grid_size, 3, dtype='int64')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if binary_threshold is None:
        fig.suptitle(f'CML Evolution')
    else:
        fig.suptitle(
            f'CML Evolution (threshold = {binary_threshold})')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_statistical_moments(skewness: np.ndarray,
                             kurtosis: np.ndarray,
                             variance: np.ndarray,
                             show: bool = True,
                             file_name: Path = None,
                             dataset_name: str = ''):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', figsize=(11.7, 8.3))

    ax1.plot(skewness)
    ax1.set_ylabel('Skewness')

    ax2.plot(variance)
    ax2.set_ylabel('Variance')

    ax3.plot(kurtosis)
    ax3.set_ylabel('Kurtosis')

    ax3.set_xlim(-20, len(kurtosis)+20)
    ticks = np.linspace(0, len(kurtosis)-1, 9, dtype='int64')
    ax3.set_xticks(ticks)

    fig.suptitle('CML Evolution - Statistical Moments')
    ax3.set_xlabel('Snapshots')
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.98,
                        hspace=0.15, wspace=0.1)
    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_gradient(u: np.ndarray, v: np.ndarray, step: int = 2,
                  show: bool = True, file_name: Path = None,
                  dataset_name: str = '', snapshot_number: int = None):

    x, y = np.meshgrid(np.arange(0, u.shape[1], 1),
                       np.arange(0, v.shape[0], 1))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(x[::step, ::step], y[::step, ::step],
              u[::step, ::step], v[::step, ::step],
              units='y', pivot='mid')

    ax.set_xlim(-step, u.shape[1])
    ax.set_ylim(-step, u.shape[0])

    ticks = np.linspace(-step, u.shape[0], 3, dtype='int64')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    fig.suptitle(f'Gradient Field - Snapshot {snapshot_number}')
    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_four_gradients(u: List[np.ndarray],
                        v: List[np.ndarray],
                        indexes: Tuple[int, int, int, int],
                        step: int = 2,
                        show: bool = True,
                        file_name: Path = None,
                        dataset_name: str = ''):

    fig = plt.figure(figsize=(11.7, 8.3))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 2),
                     axes_pad=0.28,
                     share_all=True)

    grid_size = u[indexes[0]].shape[0]
    for i, ax in enumerate(grid):
        x, y = np.meshgrid(np.arange(0, grid_size, 1),
                           np.arange(0, grid_size, 1))

        ax.quiver(x[::step, ::step], y[::step, ::step],
                  u[i][::step, ::step], v[i][::step, ::step],
                  units='y', pivot='mid')

        ax.set_xlim(-step, grid_size)
        ax.set_ylim(-step, grid_size)

        ticks = np.linspace(-step, grid_size, 3, dtype='int64')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f'Snapshot {indexes[i]}')

    fig.suptitle(f'CML Evolution - Gradient Field')
    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_entropy(modulus_entropy: np.ndarray,
                 phases_entropy: np.ndarray,
                 lattices_entropy: np.ndarray,
                 show: bool = True,
                 file_name: Path = None,
                 dataset_name: str = ''):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', figsize=(11.7, 8.3))

    ax1.plot(modulus_entropy)
    ax1.set_ylabel('Modulus Entropy')

    ax2.plot(phases_entropy)
    ax2.set_ylabel('Phases Entropy')

    ax3.plot(lattices_entropy)
    ax3.set_ylabel('Lattices Entropy')

    ax3.set_xlim(-20, len(lattices_entropy)+20)
    ticks = np.linspace(0, len(lattices_entropy)-1, 9, dtype='int64')
    ax3.set_xticks(ticks)

    fig.suptitle('CML Evolution - Entropy')
    ax3.set_xlabel('Snapshots')
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.98,
                        hspace=0.15, wspace=0.1)

    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_euler_characteristic(gamma: np.ndarray,
                              chi_gamma: np.ndarray,
                              show: bool = True,
                              file_name: Path = None,
                              dataset_name: str = ''):

    fig, ax1 = plt.subplots(1, 1, sharex='all', figsize=(11.7, 3.3))

    ax1.plot(range(len(chi_gamma)), chi_gamma, linewidth=1)

    ax1.set_ylabel(r'$\mathbf{\chi(\Gamma)}$')
    ax1.set_xlabel('Snapshots')

    fig.suptitle('CML Evolution - Euler Characteristic')

    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.08, right=0.98,
                        hspace=0.15, wspace=0.1)

    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)


def plot_gpa(g1: np.ndarray, g2: np.ndarray, g3: np.ndarray, g4: np.ndarray,
             show: bool = True, file_name: Path = None,
             dataset_name: str = ''):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.7, 4.3))
    ax1.plot(g1)
    ax1.plot(g2)
    ax1.plot(g3)
    ax1.legend(labels=['G1', 'G2', 'G3'])
    ax1.set_xlabel('Snapshots')
    ax1.set_ylabel('GPA')

    g4 = np.array(g4)
    ax2.remove()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot3D(g4.real, g4.imag, range(g4.size))
    ax2.legend(labels=['G4'])
    ax2.set_xlabel('real')
    ax2.set_ylabel('imaginary')
    ax2.set_zlabel('snapshots')

    fig.suptitle('CML Evolution - GPA Moments')

    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.07, right=0.98,
                        hspace=0.15, wspace=0.1)

    fig.text(0.99,
             0.99,
             f'[{dataset_name}]',
             horizontalalignment='right',
             verticalalignment='top')

    if show:
        plt.show()
    else:
        plt.savefig(file_name, format='png', dpi=300)
    plt.close(fig)
