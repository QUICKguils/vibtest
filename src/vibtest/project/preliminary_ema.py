"""Preliminary experimental modal analysis.

Executing the main function of this module will trigger the following tasks.
- Identification of the dominant vibtration frequencies, through the complex
  mode indicator function (CMIF).
- Use of single-degree-of-freedom methods to estimate the damping factor of
  the first vibration mode. Peak-picking and circle fit are employed.
- Display of the computed results, if desired.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from vibtest import sdof
from vibtest.project import statement as stm

DATA = stm.extract_measure(1, 1)
TIME = np.real(DATA["X1"][:, 0])
FREQ = np.real(DATA["H1_2"][:, 0])

# Natural frequencies, determined a posteriori from the reading of the CMIF
#
# There's no robust way to automatically select the resonance peaks.
# Despite the great flexibility offered by scipy.signal.find_peaks,
# there's no satisfactory filter combination that selects perfectly
# all the desired peaks, and they should better be picked manually,
# using both common sense and knowledge from the NX simulations.
EIGFREQ_BOUND = [
    (18, 20),        # freq 1
    (39, 39.6),      # freq 2
    (39, 41),        # freq 3
    (87, 88),        # freq 4
    (89, 90),        # freq 5
    (96, 98),        # freq 6
    (104, 106),      # freq 7
    (117, 119),      # freq 8
    (125.5, 126.5),  # freq 9
    (128, 131),      # freq 10
    (131, 131.5),    # freq 11
    (142, 144),      # freq 12
    (166, 167),      # freq 13
]


class Solution(NamedTuple):
    frf: npt.NDArray
    cmif: npt.NDArray
    eigfreq: npt.NDArray
    peak_picking: sdof.PeakPicking
    circle_fit: sdof.CircleFit


def main(*, out_enabled=True):
    """Execute the preliminary EMA."""

    ## Extract relevant data and identify frequencies

    frf = extract_frf(DATA)
    coh = extract_coherence(DATA)
    cmif = sdof.compute_cmif(frf)
    print(cmif)
    peaks = find_peaks(FREQ, cmif, EIGFREQ_BOUND)

    ## SDOF analysis

    # Identify and isolate the first frequency peak from the CMIF plot
    id_low, id_high = np.searchsorted(FREQ, EIGFREQ_BOUND[0])
    frf_sdof = frf[0, 0, id_low : id_high + 1]  # 0, 0 chosen bcs has the most detailed 1st freq peak
    freq_sdof = FREQ[id_low : id_high + 1]

    # Peak-picking method
    peak_picking = sdof.peak_picking_method(freq_sdof, frf_sdof)

    # Circle fit method
    circle_fit = sdof.circle_fit_method(freq_sdof, frf_sdof)

    ## Build solution and show results

    sol = Solution(frf=frf, cmif=cmif, eigfreq=peaks[:, 0], peak_picking=peak_picking, circle_fit=circle_fit)

    if out_enabled:
        print_solution(sol)
        # plot_frf_coherence(FREQ, frf, coh)
        plot_cmif_peaks(FREQ, cmif, peaks)
        # plot_peak_picking(freq_sdof, peak_picking)
        # plot_circle_fit(freq_sdof, circle_fit)
        # plot_circle_fit_dampings(circle_fit)

    return sol


def extract_frf(data):
    """Frequency response function (FRF).

    It is assumed that the `data` are recorded from a SIMO model
    of one input (hammer) and three outputs (accelerometers).
    The FRF matrix is then of size 3x1xf_sample.
    """
    h_12 = data["H1_2"][:, -1].reshape(1, 1, -1)
    h_13 = data["H1_3"][:, -1].reshape(1, 1, -1)
    h_14 = data["H1_4"][:, -1].reshape(1, 1, -1)

    return np.vstack((h_12, h_13, h_14))


def extract_coherence(data):
    """Coherence function.

    It is assumed that the `data` are recorded from a SIMO model
    of one input (hammer) and three outputs (accelerometers).
    The returned coherence is then of size 3x1xf_sample.
    """
    c_12 = data["C1_2"][:, -1].reshape(1, 1, -1)
    c_13 = data["C1_3"][:, -1].reshape(1, 1, -1)
    c_14 = data["C1_4"][:, -1].reshape(1, 1, -1)

    return np.vstack((c_12, c_13, c_14))


def find_peak(x, signal, bounds: tuple[float]) -> tuple[float]:
    """Isolate peak contained in given bounds."""
    # WARN: x should be sorted
    id_low, id_high = np.searchsorted(x, bounds)
    id_peak = id_low + np.argmax(signal[id_low:id_high+1])
    return x[id_peak], signal[id_peak]


def find_peaks(x, signal, bounds_list: np.array) -> np.array:
    """Isolate peaks contained in the given bounds."""
    peaks = np.zeros_like(bounds_list)
    for i, bounds in enumerate(bounds_list):
        peaks[i] = find_peak(x, signal, bounds)
    return peaks


def plot_frf_coherence(freq, frf, coh):
    import matplotlib.pyplot as plt

    from vibtest.mplrc import REPORT_TW

    fig, (ax_frf, ax_coh) = plt.subplots(2, 1, figsize=(REPORT_TW, REPORT_TW))
    for frf_sdof in frf:
        ax_frf.plot(freq, np.abs(frf_sdof.reshape(-1)))
    for coh_sdof in coh:
        ax_coh.plot(freq, coh_sdof.reshape(-1))

    ax_frf.set_xlabel("Frequency (Hz)")
    ax_frf.set_ylabel("FRF (g/N)")
    ax_coh.set_xlabel("Frequency (Hz)")
    ax_coh.set_ylabel("Coherence")
    ax_frf.set_yscale("log")

    fig.show()


def plot_cmif_peaks(freq, cmif, peaks):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(freq, cmif)
    ax.scatter(peaks[:, 0], peaks[:, 1], marker="x", zorder=2.5, color="C1")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("CMIF (+units)")
    ax.set_yscale("log")

    fig.show()


def plot_peak_picking(freq, pp: sdof.PeakPicking):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(freq, pp.frf_ampl, color="C0")
    ax.hlines(pp.half_power, pp.f_low, pp.f_high, color="C1")
    ax.scatter(pp.f_peak, pp.ampl_peak, marker="+", zorder=2.5, color="C3")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("FRF amplitude (g/N)")

    fig.show()


def plot_circle_fit(freq, cf: sdof.CircleFit):
    import matplotlib.pyplot as plt

    from vibtest.mplrc import REPORT_TW

    fig, ax = plt.subplots(figsize=(0.5*REPORT_TW, 0.5*REPORT_TW))
    ax.scatter(*cf.mobility, marker="x", s=15, linewidths=1)
    ax.scatter(*cf.center, marker="+", color='C1', s=30, linewidths=1.5)
    ax.set_xlabel("Re(I)/(g/(s*N))")
    ax.set_ylabel("Im(I)/(g/(s*N))")
    fitted_circle = plt.Circle(cf.center, cf.radius, fill=False, color='C7', linestyle='dashed')
    ax.add_patch(fitted_circle)
    ax.set_aspect("equal", adjustable="box")

    fig.show()


def plot_circle_fit_dampings(cf: sdof.CircleFit):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from vibtest.mplrc import REPORT_TW

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(0.5*REPORT_TW, 0.5*REPORT_TW))
    ax.plot_surface(*cf.damping_grid, cmap=cm.viridis)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel(r"$\zeta$")

    fig.show()


def print_solution(sol: Solution):
    print("=== Solutions for the preliminary EMA ===")
    print("1. CMIF")
    print("   Identified natural frequencies:")
    for i, freq in enumerate(sol.eigfreq):
        print(f"   Freq. {i+1:>2}: {freq:>6.2f} Hz")
    print("2. Peak-picking")
    print(f"   Damping ratio: {100 * sol.peak_picking.damping:.2} %")
    print("3. Circle fit")
    print(f"   Damping ratio: {100 * sol.circle_fit.damping:.2} %")
