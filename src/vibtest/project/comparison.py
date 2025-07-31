"""Comparison between finite elements analysis and experimental modal analyses.

Executing the main function of this module will trigger the following tasks.
- Construction of the modal assurance criterion (MAC) matrix, and of the autoMAC matrix.
- Display of the computed results, if desired.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import vibtest.project.constant as const
import vibtest.project.detailed_ema as dema


@dataclass(frozen=True)
class Solution:
    auto_mac_nx: npt.NDArray
    auto_mac_exp: npt.NDArray
    mac: npt.NDArray
    prop_damping: tuple[float]


def main(sol_dema: dema.Solution, *, spit_out=True) -> Solution:
    """Compare finite elements analysis and experimental modal analyses.

    Parameters
    ----------
    sol_dema: dema.Solution
        Solution returned by the detailed experimental modal analysis.
    spit_out: default True
        Spits out the computed results. Print summary in stdout, and display the graphs.
    """

    try:
        modes_nx = np.loadtxt(str(const.NX_MODES))
    except FileNotFoundError:
        modes_nx = const.extract_nx_modes()

    auto_mac_nx = compute_mac_matrix(modes_nx, modes_nx)
    auto_mac_exp = compute_mac_matrix(sol_dema.modes_real, sol_dema.modes_real)
    mac = compute_mac_matrix(sol_dema.modes_real, modes_nx)

    prop_damping = update_fea_model(sol_dema)

    sol = Solution(auto_mac_nx=auto_mac_nx, auto_mac_exp=auto_mac_exp, mac=mac, prop_damping=prop_damping)

    if spit_out:
        print_solution(sol)
        plot_mac_matrix(auto_mac_nx, auto=True)
        plot_mac_matrix(auto_mac_exp, auto=True)
        plot_mac_matrix(mac, auto=False)

    return sol


def compute_mac_matrix(modes_exp: npt.NDArray, modes_ref: npt.NDArray):
    assert modes_exp.shape == modes_ref.shape, "Bro, you mix apples with pears."

    # TODO: double loop not super classy
    n_modes = modes_exp.shape[0]
    mac = np.empty((n_modes, n_modes))
    for i, m_ref in enumerate(modes_ref):
        for j, m_exp in enumerate(modes_exp):
            mac[i, j] = np.dot(m_exp, m_ref) ** 2 / (np.dot(m_exp, m_exp) * np.dot(m_ref, m_ref))
    return mac


def update_fea_model(sol_dema: dema.Solution) -> tuple[float]:
    """Compute the proportional damping coefficients.

    Those are evaluated from the two first experimental vibration modes.
    """
    d_1 = sol_dema.poles[0].damping
    d_2 = sol_dema.poles[1].damping
    w_1 = const.NX_FREQ[0] * 2 * np.pi
    w_2 = const.NX_FREQ[1] * 2 * np.pi

    a = 2 * (w_1 * d_1 - w_2 * d_2) / (w_1**2 - w_2**2)
    b = 2 * (w_1 * d_2 - w_2 * d_1) / (w_1**2 - w_2**2) * w_1 * w_2

    return a, b


def plot_mac_matrix(mac_matrix: npt.NDArray, auto=False) -> None:
    import matplotlib.pyplot as plt

    from vibtest.mplrc import REPORT_TW

    fig, ax = plt.subplots(figsize=(0.6 * REPORT_TW, 0.6 * REPORT_TW))

    for i, j in np.ndindex(mac_matrix.shape):
        ax.text(
            j + 1, i + 1,
            f"{mac_matrix[i, j]:1.2f}", ha="center", va="center", color="C0", fontsize="xx-small",
        )

    n_rows, n_cols = mac_matrix.shape
    ax.matshow(mac_matrix, cmap="gray_r", extent=(0.5, n_cols + 0.5, n_rows + 0.5, 0.5))
    ax.xaxis.set_label_position("top")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    if auto:
        ax.set_xlabel("Modes")
        ax.set_ylabel("Modes")
    else:
        ax.set_xlabel("Experimental modes")
        ax.set_ylabel("Reference modes")
    ax.grid(False)

    fig.show()


def inspect_all_nx_modes_shape(nx_modes: npt.NDArray) -> None:
    import matplotlib.pyplot as plt

    from vibtest.mplrc import REPORT_TW

    n_mode = nx_modes.shape[0]
    n_col = 3
    n_row = int(np.ceil(n_mode/n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=(2 * REPORT_TW, 2 * REPORT_TW), subplot_kw={"projection": "3d"}, dpi=100)

    for k, mode in enumerate(nx_modes[:n_mode]):
        dema.plot_mode_shape(mode, (fig, axs.flat[k]))


def print_solution(sol: Solution) -> None:
    print("=== Solutions for the comparison ===")
    print("Proportional damping parameter for model updating")
    print(f"a = {sol.prop_damping[0]:.4g}")
    print(f"b = {sol.prop_damping[1]:.4g}")
