"""Comparison between finite elements analysis and experimental modal analyses.

Executing the main function of this module will trigger the following tasks.
- Construction of the modal assurance criterion (MAC) matrix, and of the autoMAC matrix.
- Display of the computed results, if desired.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import vibtest.project.detailed_ema as dema


@dataclass(frozen=True)
class Solution:
    mac: npt.NDArray
    auto_mac: npt.NDArray


def main(sol_dema: dema.Solution, *, spit_out=True) -> Solution:
    """Compare finite elements analysis and experimental modal analyses.

    Parameters
    ----------
    sol_dema: dema.Solution
        Solution returned by the detailed experimental modal analysis.
    spit_out: default True
        Spits out the computed results. Print summary in stdout, and display the graphs.
    """

    auto_mac = compute_mac_matrix(sol_dema.modes_real, sol_dema.modes_real)
    # TODO: find a way to extract the nx modes shapes
    mac = compute_mac_matrix(sol_dema.modes_real, sol_dema.modes_real)

    sol = Solution(mac=mac, auto_mac=auto_mac)

    if spit_out:
        print_solution(sol)

    return sol


def compute_mac_matrix(modes_exp: npt.NDArray, modes_ref: npt.NDArray):
    assert modes_exp.shape == modes_ref.shape, "Bro, you mix apples with pears."""

    # TODO: double loop not super classy
    n_modes = modes_exp.shape[0]
    mac = np.empty((n_modes, n_modes))
    for i, m_ref in enumerate(modes_ref):
        for j, m_exp in enumerate(modes_exp):
            mac[i, j] = np.dot(m_exp, m_ref)**2 / (np.dot(m_exp, m_exp)*np.dot(m_ref, m_ref))
    return mac

def plot_mac_matrix(mac_matrix: npt.NDArray) -> None:
    import matplotlib.pyplot as plt
    from vibtest.mplrc import REPORT_TW

    fig, ax = plt.subplots(figsize=(0.6 * REPORT_TW, 0.6 * REPORT_TW))

    for i, j in np.ndindex(mac_matrix.shape):
        ax.text(j, i, f"{mac_matrix[i, j]:1.2f}", ha="center", va="center", color="w", fontsize="xx-small")

    ax.imshow(mac_matrix)  #, cmap='gray_r')
    ax.set_xlabel("Experimental modes")
    ax.set_ylabel("Reference modes")
    ax.grid(False)

    fig.show()


def print_solution(sol: Solution) -> None:
    print("=== Solutions for the comparison ===")
