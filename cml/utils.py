import shutil

from pathlib import Path


def create_output_dir(output: Path = None, clean=False):
    if output:
        if output.is_dir() and clean:
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)