"""Finite element analysis of the structure.

This module answers the first part of the project.
"""

import numpy as np

from vibtest import sdof
from vibtest.project import statement as stm

DATA = stm.extract_measure(1, 2)
TIME = np.real(DATA["X1"][:, 0])
FREQ = np.real(DATA["H1_2"][:, 0])


def main():
    frf = sdof.compute_frf(DATA)
    cmif = sdof.compute_cmif(frf)
    sdof.plot_cmif(cmif, FREQ)

    return frf, cmif
