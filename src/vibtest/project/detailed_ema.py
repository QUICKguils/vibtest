"""Detailed experimental modal analysis.

Executing the main function of this module will trigger the following tasks.
- Computation of the modal parameters of the plane under study via polyMAX.
- Construction of the corresponding real modes shapes via LSFD.
- Display of the computed results, if desired.
"""

from dataclasses import dataclass

import numpy as np

from vibtest import mdof
from vibtest.project import constant as c

# Define the setup parameters used in the second lab session.
# They are extracted from, e.g., the first hammer test.
# The accelerometer placement is the same for all tests.
_DATA = c.extract_measure(2, 1)
FSAMPLE = np.real(_DATA['H1_2'][:, 0])
FMAX = FSAMPLE[-1]
DW = FSAMPLE[1] - FSAMPLE[0]
TSAMPLE = _DATA['X1'][:, 0]
TMAX = TSAMPLE[-1]
DT = TSAMPLE[1] - TSAMPLE[0]


@dataclass(frozen=True)
class Solution:
    pass


def main(*, out_enabled=True) -> Solution:
    """Execute the detailed EMA."""
    H = build_frf_matrix()

    # Flip n_i and n_o to speed up the computations
    H = H.swapaxes(0, 1)
    print(H.shape)

    sol_polymax = mdof.polymax(FSAMPLE, H, DT, 2)

    sol = Solution()

    if out_enabled:
        print_solution(sol)

    return sol


def build_frf_matrix():
    """Build the matrix of recorded frequency response functions."""

    h_1 = np.fromiter(
        (c.extract_measure(2, i + 1)["H1_2"][:, -1] for i in range(c.N_DOF)),
        dtype=np.dtype((complex, len(FSAMPLE))),
    )
    h_2 = np.fromiter(
        (c.extract_measure(2, i + 1)["H1_3"][:, -1] for i in range(c.N_DOF)),
        dtype=np.dtype((complex, len(FSAMPLE))),
    )
    h_3 = np.fromiter(
        (c.extract_measure(2, i + 1)["H1_4"][:, -1] for i in range(c.N_DOF)),
        dtype=np.dtype((complex, len(FSAMPLE))),
    )

    return np.array((h_1, h_2, h_3))


def plot_stabilization_diagram():
    pass


def plot_argand_diagram(sol):
    pass


def print_solution(sol: Solution):
    print("=== Solutions for the detailed EMA ===")
