import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Tuple

from .cml import CML
from .animation import animate, create_animation
from .plot import plot_four_lattices
from .lattice import Lattice


class Evolution:

    @staticmethod
    def create_from_file(dataset_file: Path = None,
                         dataset_name: str = 'system_dataset'):

        if isinstance(dataset_file, Path) and dataset_file.exists():
            dataset = np.loadtxt(dataset_file, delimiter=',')

            with dataset_file.open() as f:
                line = f.readline().strip()
                if line == '# CML configuration':
                    cml_config = []
                    while line:
                        line = f.readline().strip()
                        if line == '#':
                            break
                        cml_config.append(line.replace('# ', ''))
                    cml = '\n'.join(cml_config)
                else:
                    cml = None

            grid_size = dataset.shape[1]
            shape = (int(dataset.shape[0]/grid_size),
                     grid_size,
                     grid_size)

            dataset = dataset.reshape(shape)

            evolution = Evolution()
            evolution.cml = cml
            evolution.iterations = shape[0] - 1
            evolution.output_dir = dataset_file.parent
            evolution.dataset_name = dataset_name
            evolution.dynamic_mode = False
            evolution.grid_size = grid_size

            for snapshot in dataset:
                lattice = Lattice(shape=(grid_size, grid_size))
                lattice.u = snapshot
                evolution.snapshots.append(lattice)

            return evolution

    def __init__(self,
                 cml: CML = CML(),
                 iterations: int = 1024,
                 output_dir: Path = Path('/tmp/cml'),
                 dataset_name: str = 'cml_dataset',
                 dynamic_mode: bool = True):

        self.cml = cml
        self.iterations = iterations
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.file_name = self.output_dir / self.dataset_name
        self.dynamic_mode = dynamic_mode
        self.grid_size = cml.grid_size
        self.cml_config = ''

        self.snapshots = list()

    def run(self):
        if not self.dynamic_mode:
            print('Cannot run in heterogeneous system.')
            return

        self.snapshots.clear()
        self.snapshots.append(self.cml.lattice)
        for k in range(1, self.iterations + 1):
            self.cml.update()
            self.snapshots.append(self.cml.lattice)

    def save_csv(self):
        with open(self.file_name.with_suffix('.txt'), 'w') as outfile:

            cml_config = str(self).split('\n')
            for i, line in enumerate(cml_config):
                cml_config[i] = '# ' + line

            outfile.write('# CML configuration\n')
            outfile.write('\n'.join(cml_config))
            outfile.write('\n#\n')

            outfile.write('# Array shape: {0}, {1}, {2}\n'.format(
                self.iterations+1, *self.cml.lattice.shape))

            for i, snapshot in enumerate(self.snapshots):
                outfile.write(f'# Snapshot {i}\n')
                np.savetxt(outfile, snapshot.u, delimiter=',')

    def plot(self, show: bool = True, binary_threshold=None):

        if binary_threshold:
            file_name = self.file_name.name + '_binary_plot'
        else:
            file_name = self.file_name.name + '_plot'

        file_name = self.file_name.with_name(file_name).with_suffix('.png')
        indexes = (
            0,
            int((self.iterations + 1) / 4),
            self.iterations - int((self.iterations + 1) / 4),
            self.iterations
        )

        plot_four_lattices(self.snapshots, indexes, show=show,
                           file_name=file_name,
                           binary_threshold=binary_threshold,
                           dataset_name=self.dataset_name,
                           cml_config=str(self))

    def animate(self, show: bool = True, notebook: bool = False, fps: int = 4):
        if notebook:
            animation = animate(self.snapshots, notebook=notebook, fps=fps,
                                dataset_name=self.dataset_name,
                                cml_config=str(self))
            plt.close()
            return animation

        if show:
            animate(self.snapshots, show=show, fps=fps,
                    dataset_name=self.dataset_name, cml_config=str(self))
        else:
            create_animation(self.snapshots, fps=fps,
                             file_name=self.file_name.with_suffix('.mp4'),
                             dataset_name=self.dataset_name,
                             cml_config=str(self))

    def __str__(self):
        if self.dynamic_mode:
            return f'Iterations: {str(self.iterations)}\n{str(self.cml)}'
        else:
            if isinstance(self.cml, str):
                return self.cml
            else:
                return (f'Iterations: {str(self.iterations)}\n'
                        f'Grid size: {self.grid_size} x {self.grid_size}')
