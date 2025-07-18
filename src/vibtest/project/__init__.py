"""Code developed as part of the project.

This python package holds all the computer code written
as part of the Vibration Testing course (MECA0062-1).

Submodules
----------
statement -- Define the general project statement data.
part_1    -- Finite element analysis of the structure.
part_2    -- Experimental modal analysis.
part_3    -- Detailed experimental analysis.
part_4    -- Comparison between finite element and experimental results.
"""

import importlib
import pathlib

_PROJECT_PATH = pathlib.Path(__file__).parent

submodules = [
    'statement',
    'part_1',
    'part_2',
    'part_3',
    'part_4',
]

__all__ = submodules + [ '_PROJECT_PATH' ]


def __dir__():
    return __all__


# This way of automatically detect and load submodules is
# inspired from the __init__.py file of scipy.
def __getattr__(name):
    if name in submodules:
        return importlib.import_module(f'vibtest.project.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'vibtest.project' has no attribute '{name}'")
