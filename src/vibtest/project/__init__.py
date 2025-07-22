"""Code developed as part of the project.

This python package holds all the computer code written
as part of the Vibration Testing course (MECA0062-1).

Submodules
----------
data            -- Data manipulation utilities.
preliminary_ema -- Preliminary experimental modal analysis.
detailed_ema    -- Detailed experimental modal analysis.
comparison      -- Comparison between FEA and EMAs.
"""

import importlib
import pathlib

_PROJECT_PATH = pathlib.Path(__file__).parent

submodules = [
    'data',
    'preliminary_ema',
    'detailed_ema',
    'comparison',
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
