"""Detailed experimental modal analysis.

Executing the main function of this module will trigger the following tasks.
- Computation of the modal parameters of the plane under study via polyMAX.
- Construction of the corresponding real modes shapes via LSFD.
- Display of the computed results, if desired.
"""

import pickle
from dataclasses import dataclass
from typing import List

import numpy as np

from vibtest import mdof
from vibtest.project import _PROJECT_PATH, constant

_DUMP_FILE = str(_PROJECT_PATH / "res" / "detailed_ema.pickle")

# Define the setup parameters used in the second lab session.
# They are extracted from, e.g., the first hammer test.
# The accelerometer placement is the same for all tests.
_DATA = constant.extract_measure(2, 1)
FSAMPLE = np.real(_DATA["H1_2"][:, 0])
FMAX = FSAMPLE[-1]
DW = FSAMPLE[1] - FSAMPLE[0]
TSAMPLE = _DATA["X1"][:, 0]
TMAX = TSAMPLE[-1]
DT = TSAMPLE[1] - TSAMPLE[0]

# Position of the selected stable poles,
# determined a posteriori from the reading of the stabilization diagram.
# Each position of the selected poles is specified by a tuple holding
# the polynomial order and the approximate frequency (hertz) of the pole.
#
# NOTE:
# As for the peaks identification of the preliminary analysis,
# there's no satisfactory ways to automate the pole identification.
# It is generally better to let the testing engineer manually identify
# and pick the satisfactory poles.
SPOTTED_POLES_POS = [
    (86, 18.84),
    (90, 40.18),
    (66, 87.74),
    (94, 89.63),
    (46, 97.53),
    (58, 105.19),
    (66, 117.91),
    (66, 125.2),
    (44, 125.69),
    (80, 129.74),
    (96, 135.12),
    (98, 143.37),
    (88, 166.27),
    # (66, 199),
]


@dataclass(frozen=True)
class Solution:
    stabilization: List[mdof.PolyMAX]
    poles: List[mdof.Pole]
    # mode: List[mdof.LSFD]


def main(*, spit_out=True, load_dump=True, save_dump=False) -> Solution:
    """Execute the detailed EMA.

    Parameters
    ----------
    spit_out: default True
        Spits out the computed results. Print summary in stdout, and display the graphs.
    load_dump: default True
        Load a saved solution from a dump pickle file, instead of actually computing the solution.
    save_dump: default False
        Save the computed solution into a dump pickle file.
    """

    if load_dump:
        sol = _main_load_dump()
    else:
        sol = _main_compute()

    if spit_out:
        print_solution(sol)
        plot_stabilization_diagram(sol)

    if save_dump:
        with open(_DUMP_FILE, "wb") as handle:
            pickle.dump(sol, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sol


def _main_compute() -> Solution:
    H = build_frf_matrix()
    H_swapped = H.swapaxes(0, 1)  # flip n_i and n_o to speed up computations

    sol_stabilization = mdof.stabilization(FSAMPLE, H_swapped, DT, 100)

    extracted_poles = extract_spotted_poles(sol_stabilization)

    residues = mdof.lsfd_residues(FSAMPLE[1:], H_swapped[:, :, 1:], extracted_poles, debug=True)

    return Solution(stabilization=sol_stabilization, poles=extracted_poles)


def _main_load_dump() -> Solution:
    with open(_DUMP_FILE, "rb") as handle:
        return pickle.load(handle)


def build_frf_matrix():
    """Build the matrix of recorded frequency response functions."""

    h_1 = np.fromiter(
        (constant.extract_measure(2, i + 1)["H1_2"][:, -1] for i in range(constant.N_DOF)),
        dtype=np.dtype((complex, len(FSAMPLE))),
    )
    h_2 = np.fromiter(
        (constant.extract_measure(2, i + 1)["H1_3"][:, -1] for i in range(constant.N_DOF)),
        dtype=np.dtype((complex, len(FSAMPLE))),
    )
    h_3 = np.fromiter(
        (constant.extract_measure(2, i + 1)["H1_4"][:, -1] for i in range(constant.N_DOF)),
        dtype=np.dtype((complex, len(FSAMPLE))),
    )

    return np.array((h_1, h_2, h_3))


def extract_spotted_poles(sol_stab: List[mdof.PolyMAX]) -> List[mdof.Pole]:
    """Extract spotted poles. Bruh everthing is in the damn function name."""
    extracted_poles = []
    for order, f_guess in SPOTTED_POLES_POS:
        # TODO: Flexing a bit too much with array comprehension here. Not super readable.
        poles_line = [sol_line.poles for sol_line in sol_stab if sol_line.order == order][0]
        ix_freq = np.argmin(np.abs(np.array([p.freq for p in poles_line]) - f_guess))

        selected_pole = poles_line[ix_freq]
        if selected_pole.status is not mdof.PoleStatus.s:
            print("WARN: A non-stabilized pole has been extracted.")

        extracted_poles.append(selected_pole)

    return extracted_poles


def plot_stabilization_diagram(sol: Solution):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.grid(visible=None, axis="x")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Polynomial order")

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
        status_plot_specs = {
            mdof.PoleStatus.o: {"color": "C2", "alpha": 0.4, "fontsize": "xx-small"},
            mdof.PoleStatus.f: {"color": "C1", "alpha": 0.5, "fontsize": "xx-small"},
            mdof.PoleStatus.d: {"color": "C1", "alpha": 0.5, "fontsize": "xx-small"},
            mdof.PoleStatus.s: {"color": "C0", "alpha": 1, "fontsize": "x-small"},
        }
        for freq, status in zip(freqs, statuses):
            ax.text(
                freq, s.order, status.name, ha="center", va="center", **status_plot_specs[status]
            )

    fig.show()


def plot_argand_diagram(complex_modes) -> None:
    import matplotlib.pyplot as plt

    from vibtest.mplrc import REPORT_TW

    fig, axs = plt.subplots(5, 3, figsize=(REPORT_TW, 2*REPORT_TW), subplot_kw={'projection': 'polar'})
    for ix, mode in enumerate(complex_modes):
        axs.flat[ix].scatter(np.angle(mode), np.abs(mode), alpha=0.3, s=10)

    fig.show()


def plot_modes_shape(sol:Solution):
    pass


def print_solution(sol: Solution):
    print("=== Solutions for the detailed EMA ===")
    print("1. Selected poles")
    for ix, p in enumerate(sol.poles):
        print(f"   Mode {ix:>2} : freq = {p.freq:>7.3f} Hz, damping = {100 * p.damping:>5.3f} %")
