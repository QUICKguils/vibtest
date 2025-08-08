"""Comparison between finite elements analysis and experimental modal analyses.

Executing the main function of this module will trigger the following tasks.
- Construction of the modal assurance criterion (MAC) matrix, and of the autoMAC matrix.
- Display of the computed results, if desired.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vibtest.project import _PROJECT_PATH
import vibtest.project.constant as const
import vibtest.project.detailed_ema as dema

NX_MODES_UPDATED_DIR = _PROJECT_PATH / "res" / "nx_modes_updated"
NX_MODES_UPDATED_PATH = NX_MODES_UPDATED_DIR/ "modes_filtered.csv"

NX_FREQ_UPDATED = [  # identified from the updated NX model
    1.782700E+01,  # mode 1
    3.930481E+01,  # mode 2
    8.628332E+01,  # mode 3
    8.945659E+01,  # mode 4
    9.660116E+01,  # mode 5
    1.023301E+02,  # mode 6
    1.199808E+02,  # mode 7
    1.206662E+02,  # mode 8
    1.282630E+02,  # mode 9
    1.426406E+02,  # mode 10
    1.723441E+02,  # mode 11
    1.773359E+02,  # mode 12
    1.780801E+02,  # mode 13
]


@dataclass(frozen=True)
class Solution:
    auto_mac_nx: npt.NDArray
    auto_mac_exp: npt.NDArray
    mac: npt.NDArray
    mac_updated: npt.NDArray


def main(sol_dema: dema.Solution, *, spit_out=True) -> Solution:
    """Compare finite elements analysis and experimental modal analyses.

    Parameters
    ----------
    sol_dema: dema.Solution
        Solution returned by the detailed experimental modal analysis.
    spit_out: default True
        Spits out the computed results. Print summary in stdout, and display the graphs.
    """

    # Fetch (or extract) the NX modes
    try:
        modes_nx = np.loadtxt(str(const.NX_MODES_PATH))
    except FileNotFoundError:
        modes_nx = const.extract_nx_modes(const.NX_MODES_DIR, const.NX_MODES_PATH)
    try:
        modes_nx_updated = np.loadtxt(str(NX_MODES_UPDATED_PATH))
    except FileNotFoundError:
        modes_nx_updated = const.extract_nx_modes(NX_MODES_UPDATED_DIR, NX_MODES_UPDATED_PATH)

    # Check autoMACs
    auto_mac_nx = compute_mac_matrix(modes_nx, modes_nx)
    auto_mac_exp = compute_mac_matrix(sol_dema.modes_real, sol_dema.modes_real)

    # DEMA and FEA comparison
    mac = compute_mac_matrix(sol_dema.modes_real, modes_nx)

    # DEMA and updated FEA comparison
    mac_updated = compute_mac_matrix(sol_dema.modes_real, modes_nx_updated)

    sol = Solution(
        auto_mac_nx=auto_mac_nx,
        auto_mac_exp=auto_mac_exp,
        mac=mac,
        mac_updated=mac_updated,
    )

    if spit_out:
        print_solution(sol_dema)
        plot_mac_matrix(auto_mac_nx, auto=True)
        plot_mac_matrix(auto_mac_exp, auto=True)
        plot_mac_matrix(mac, auto=False)
        plot_mac_matrix(mac_updated, auto=False)

    return sol


def compute_mac_matrix(modes_exp: npt.NDArray, modes_ref: npt.NDArray):
    assert modes_exp.shape == modes_ref.shape, "Bro, you mix apples with pears."

    # TODO: double loop not super classy, I know
    n_modes = modes_exp.shape[0]
    mac = np.empty((n_modes, n_modes))
    for i, m_ref in enumerate(modes_ref):
        for j, m_exp in enumerate(modes_exp):
            mac[i, j] = np.dot(m_exp, m_ref) ** 2 / (np.dot(m_exp, m_exp) * np.dot(m_ref, m_ref))
    return mac


def plot_mac_matrix(mac_matrix: npt.NDArray, auto=False) -> None:
    import matplotlib.pyplot as plt

    from vibtest.mplrc import REPORT_TW

    fig, ax = plt.subplots(figsize=(0.6 * REPORT_TW, 0.6 * REPORT_TW))

    for i, j in np.ndindex(mac_matrix.shape):
        ax.text(
            j + 1, i + 1, f"{mac_matrix[i, j]:1.2f}",
            ha="center", va="center", color="C0", fontsize="xx-small",
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
    n_row = int(np.ceil(n_mode / n_col))
    fig, axs = plt.subplots(
        n_row, n_col,
        figsize=(2 * REPORT_TW, 2 * REPORT_TW), subplot_kw={"projection": "3d"}, dpi=100,
    )

    for k, mode in enumerate(nx_modes[:n_mode]):
        dema.plot_mode_shape(mode, (fig, axs.flat[k]))


def update__proportional_damping(sol_dema: dema.Solution) -> tuple[float]:
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


def update__engine_inertia() -> tuple[float]:
    MASS = 2.15  # Mass of each engine [kg]
    RHO = 7800  # Density of steel composing the engines [kg/m³]
    L = 0.1  # Length of engine parallelepiped [m]

    a = np.sqrt(MASS / (L * RHO))

    I_XX = 1 / 12 * MASS * 2 * a**2
    I_YY = 1 / 12 * MASS * (L**2 + a**2)
    I_ZZ = I_YY

    return I_XX, I_YY, I_ZZ


def print_solution(sol_dema: dema.Solution) -> None:
    print("=== Solutions for the comparison ===")

    print("Proportional damping parameter for model updating")
    a, b = update__proportional_damping(sol_dema)
    print(f"a = {a:.4g}")
    print(f"b = {b:.4g}")

    print("Engine inertias for model updating")
    I_XX, I_YY, I_ZZ = update__engine_inertia()
    print(f"I_XX = {I_XX:.4g} kg/m²")
    print(f"I_YY = {I_YY:.4g} kg/m²")
    print(f"I_ZZ = {I_ZZ:.4g} kg/m²")
