import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import subprocess

from concurrent import futures
from pathlib import Path
from typing import List

from .lattice import Lattice
from .utils import create_output_dir


def animate(snapshots: List[Lattice],
            file_name: Path = None,
            fps: int = 4,
            start: int = 0,
            show: bool = True,
            notebook: bool = False,
            dataset_name: str = '',
            cml_config: str = ''):

    mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False

    if notebook:
        fig = plt.figure(figsize=(6.4, 3.6))
    else:
        fig = plt.figure(figsize=(12.8, 7.2))

    grid_size = snapshots[0].shape[0]

    im = plt.imshow(snapshots[0].u,
                    interpolation='none',
                    aspect='equal',
                    cmap=plt.get_cmap('gray'),
                    vmin=0.0,
                    vmax=1.0,
                    extent=[1, grid_size,
                            grid_size, 1])

    fig.text(0.01,
             0.88,
             f'[{dataset_name}]\n{cml_config}',
             horizontalalignment='left',
             verticalalignment='top')

    fig.suptitle(f'CML Evolution')

    ticks = np.linspace(1, grid_size, 3, dtype='int64')
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.colorbar()

    def animate_func(i):
        if i % 10 == 0 and not show:
            print('.', end='')

        plt.title(f'Snapshot {start + i}')
        _ = im.set_array(snapshots[i].u)
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(snapshots),
        interval=1000/fps
    )

    if notebook:
        fig.subplots_adjust(top=0.85, bottom=0.1, left=0.4, right=0.99,
                            hspace=0.15, wspace=0.1)
        plt.rcParams['animation.embed_limit'] = 40
        return anim.to_jshtml(fps=fps)

    if show:
        plt.show()
    else:
        workers = multiprocessing.cpu_count()
        writer = animation.FFMpegWriter(fps=fps,
                                        codec='libx264',
                                        extra_args=['-threads',
                                                    str(workers)])
        anim.save(str(file_name), writer=writer)


def create_animation(snapshots: List[Lattice],
                     file_name: Path = None,
                     fps: int = 4,
                     dataset_name: str = '',
                     cml_config: str = ''):

    output_dir = Path(f'/tmp/cml_animation_{file_name.with_suffix("").name}')
    create_output_dir(output_dir, clean=True)

    workers = multiprocessing.cpu_count()
    start_snapshot = 0
    with futures.ProcessPoolExecutor(workers) as executor:
        to_do = []
        with (output_dir / 'list.txt').open('w') as f:
            for i, chunk in enumerate(np.array_split(snapshots, workers)):
                print(chunk.shape)
                print(f'Start snapshot={start_snapshot}')

                future = executor.submit(animate,
                                         chunk,
                                         file_name=(output_dir /
                                                    f'chunk_{i}.mp4'),
                                         start=start_snapshot,
                                         fps=fps,
                                         show=False,
                                         dataset_name=dataset_name,
                                         cml_config=cml_config)
                start_snapshot = start_snapshot + chunk.shape[0]
                f.write(f'file chunk_{i}.mp4\n')
                future.sid = f'chunk_{i}.mp4'
                to_do.append(future)

        for future in futures.as_completed(to_do):
            _ = future.result()
            print(f'#{future.sid} successfully generated!')

    subprocess.call(['/usr/bin/ffmpeg',
                     '-f',
                     'concat',
                     '-i',
                     output_dir / 'list.txt',
                     '-c',
                     'copy',
                     file_name,
                     '-y'])


def animate_gradient(u: List[np.ndarray],
                     v: List[np.ndarray],
                     fps: int = 4,
                     start: int = 0,
                     step: int = 2,
                     show: bool = True,
                     file_name: Path = None,
                     notebook: bool = False,
                     dataset_name: str = '',
                     cml_config: str = ''):

    if notebook:
        fig = plt.figure(figsize=(6.4, 3.6))
    else:
        fig = plt.figure(figsize=(12.8, 7.2))

    grid_size = u[0].shape[0]
    x, y = np.meshgrid(np.arange(0, grid_size, 1),
                       np.arange(0, grid_size, 1))

    qv = plt.quiver(x[::step, ::step], y[::step, ::step],
                    u[0][::step, ::step], v[0][::step, ::step],
                    units='y', pivot='mid')
    plt.gca().set_aspect('equal')

    plt.xlim(-step, grid_size)
    plt.ylim(-step, grid_size)

    fig.text(0.01,
             0.88,
             f'[{dataset_name}]\n{cml_config}',
             horizontalalignment='left',
             verticalalignment='top')

    fig.suptitle(f'CML Evolution - Gradient Field')
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.3, right=0.99,
                        hspace=0.15, wspace=0.1)

    ticks = np.linspace(-step, grid_size, 3, dtype='int64')
    plt.xticks(ticks)
    plt.yticks(ticks)

    def animate_func(i):
        if i % 10 == 0 and not show:
            print('.', end='')

        plt.title(f'Snapshot {start + i}')
        qv.set_UVC(u[i][::step, ::step],
                   v[i][::step, ::step])
        return [qv]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(u),
        interval=1000/fps
    )

    if notebook:
        plt.rcParams['animation.embed_limit'] = 40
        return anim.to_jshtml(fps=fps)

    if show:
        plt.show()
    else:
        workers = multiprocessing.cpu_count()
        writer = animation.FFMpegWriter(fps=fps,
                                        codec='libx264',
                                        extra_args=['-threads', str(workers)])
        anim.save(str(file_name), writer=writer)


def create_gradient_animation(u: List[np.ndarray],
                              v: List[np.ndarray],
                              fps: int = 4,
                              step: int = 2,
                              file_name: Path = None,
                              dataset_name: str = '',
                              cml_config: str = ''):

    output_dir = Path(
        f'/tmp/cml_gradient_animation_{file_name.with_suffix("").name}')
    create_output_dir(output_dir, clean=True)

    workers = multiprocessing.cpu_count()
    start_snapshot = 0
    with futures.ProcessPoolExecutor(workers) as executor:
        to_do = []
        with (output_dir / 'list.txt').open('w') as f:
            v_chunks = np.array_split(v, workers)
            for i, u_chunk in enumerate(np.array_split(u, workers)):
                print(u_chunk.shape)

                print(f'Start snapshot={start_snapshot}')

                future = executor.submit(animate_gradient,
                                         u_chunk,
                                         v_chunks[i],
                                         fps=fps,
                                         start=start_snapshot,
                                         step=step,
                                         show=False,
                                         file_name=(output_dir /
                                                    f'chunk_{i}.mp4'),
                                         dataset_name=dataset_name,
                                         cml_config=cml_config)

                start_snapshot = start_snapshot + u_chunk.shape[0]
                f.write(f'file chunk_{i}.mp4\n')
                future.sid = f'chunk_{i}.mp4'
                to_do.append(future)

        for future in futures.as_completed(to_do):
            _ = future.result()
            print(f'#{future.sid} successfully generated!')

    subprocess.call(['/usr/bin/ffmpeg',
                     '-f',
                     'concat',
                     '-i',
                     output_dir / 'list.txt',
                     '-c',
                     'copy',
                     file_name,
                     '-y'])
