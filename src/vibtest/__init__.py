"""Experimental modal analysis of a simplified aircraft structure.

This python package holds all the computer code written
as part of the Vibration Testing course (MECA0062-1).

Subpackages
-----------
project -- Code developed as part of the project.

Submodules
----------
mplrc -- Set some global matplotlib parameters.
sdof  -- Single-degree of freedom identification techniques.
"""

import pathlib

from vibtest import mplrc

_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent
_SRC_PATH = pathlib.Path(__file__).parent

mplrc.load_rcparams()
