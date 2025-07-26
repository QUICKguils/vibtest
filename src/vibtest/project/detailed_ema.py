"""Detailed experimental modal analysis.

Executing the main function of this module will trigger the following tasks.
- Computation of the modal parameters of the plane under study via polyMAX.
- Construction of the corresponding real modes shapes via LSFD.
- Display of the computed results, if desired.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from vibtest import mdof
from vibtest.project import constant as c

# Define the setup parameters used in the second lab session.
# They are extracted from, e.g., the first hammer test.
# The accelerometer placement is the same for all tests.
_DATA = c.extract_measure(2, 1)
FSAMPLE = np.real(_DATA["H1_2"][:, 0])
FMAX = FSAMPLE[-1]
DW = FSAMPLE[1] - FSAMPLE[0]
TSAMPLE = _DATA["X1"][:, 0]
TMAX = TSAMPLE[-1]
DT = TSAMPLE[1] - TSAMPLE[0]


@dataclass(frozen=True)
class Solution:
    stabilization: List[mdof.PolyMAX]


def main(*, out_enabled=True, dump=False) -> Solution:
    """Execute the detailed EMA."""

    H = build_frf_matrix()

    # Flip n_i and n_o to speed up the computations
    H = H.swapaxes(0, 1)

    sol_stabilization = mdof.stabilization_parallel(FSAMPLE, H, DT, 80, debug=True)

    sol = Solution(stabilization=sol_stabilization)

    if out_enabled:
        plot_stabilization_diagram(sol)
        print_solution(sol)

    if dump:
        import pickle
        from vibtest.project import _PROJECT_PATH
        savefile = str(_PROJECT_PATH / "res" / "detailed_ema.pickle")
        with open(savefile, 'wb') as handle:
            pickle.dump(sol, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


def plot_stabilization_diagram(sol: Solution):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # ax.set_xlim(left=0, right=200)
    # ax.set_ylim(bottom=0, top=20)
    ax_frf = ax.twinx()
    ax_frf.set_yscale("log")
    ax_frf.axis("off")

    ax_frf.plot(np.real(_DATA["H1_2"][:, 0]), np.abs(_DATA["H1_2"][:, -1]), color="C7", alpha=0.5)
    ax_frf.plot(np.real(_DATA["H1_3"][:, 0]), np.abs(_DATA["H1_3"][:, -1]), color="C7", alpha=0.5)
    ax_frf.plot(np.real(_DATA["H1_4"][:, 0]), np.abs(_DATA["H1_4"][:, -1]), color="C7", alpha=0.5)

    # Dummy scatter to set axis limits and avoid zero axis dimensions
    ax.scatter([0, FMAX], [0, sol.stabilization[-1].order], marker="none")

    for s in sol.stabilization:
        poles = [pole for pole in s.poles if pole.freq <= FMAX]
        freqs = [p.freq for p in poles]
        statuses = [p.status for p in poles]
        color_status = {
            mdof.PoleStatus.o: "C6",
            mdof.PoleStatus.f: "C5",
            mdof.PoleStatus.d: "C5",
            mdof.PoleStatus.s: "C4",
        }
        for freq, status in zip(freqs, statuses):
            ax.text(freq, s.order, status.name, color=color_status[status], ha="center", va="center")

    fig.show()


def plot_argand_diagram(sol):
    pass


def print_solution(sol: Solution):
    print("=== Solutions for the detailed EMA ===")
