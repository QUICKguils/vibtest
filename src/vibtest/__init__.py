"""vibtest -- Vibration testing of a plane structure.

This repository holds all the code written for the project carried out as part
of the Vibration Testing and Experimental Modal Analysis course (MECA0062-1),
academic year 2024-2025.

Subpackages
-----------

util -- Common utilities that are used throughout the code

Credits
-------

This __init__.py file, and the general code structure of this project
is inspired from the Scipy project:
https://scipy.org/
"""

import importlib as _importlib
import pathlib as _pathlib

ROOT_PATH = _pathlib.Path(__file__).parent.parent.parent

SUBMODULES = [
    'util'
]


def __getattr__(name):
    if name in SUBMODULES:
        return _importlib.import_module(f'vibtest.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'vibtest' has no attribute '{name}'")
