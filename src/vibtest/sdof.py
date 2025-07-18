"""sdof -- Single-degree-of-freedom identification techniques."""

from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.linalg import svd


def compute_cmif(frf):
    """Complex mode indicator function (CMIF).

    As the domain space of the FRF is assumed to be 1D (only one input),
    it happens that the CMIF is simply the square of the FRF norm.
    More details in the report.

    NOTE:
    The following solution is also valid,
    but slightly slower (import scipy.linalg):
    >>> return linalg.norm(frf, axis=0)**2
    """
    return np.sum(np.real(np.conj(frf) * frf), axis=0).reshape(-1)
    # S = np.zeros((frf.shape[2], 1))
    # print(S.shape)
    # for id_w, h in enumerate(frf):
    #     _, S_array, _ = svd(h)
    #     S[id_w] = S_array[0]
    # return S * S


class PeakPicking(NamedTuple):
    """Computed quantities from the peak-picking method."""

    frf_ampl: npt.NDArray
    id_peak: int
    ampl_peak: float
    f_peak: float
    f_low: float
    f_high: float
    half_power: float
    damping: float


def peak_picking_method(freq, frf_sdof):
    """Get damping of SDOF model via the peak-picking method.

    Parameters
    ----------
    freq, frf_sdof : np.array[float]
        SDOF FRF and its frequencies, near the resonance peak.
    """
    frf_ampl = np.abs(frf_sdof)

    # Spot the maximum
    id_peak = np.argmax(frf_ampl)
    ampl_peak = frf_ampl[id_peak]
    f_peak = freq[id_peak]

    half_power = ampl_peak / np.sqrt(2)

    # Find intersection of frf peak with half-power level
    f_low = np.interp(half_power, frf_ampl[: id_peak + 1], freq[: id_peak + 1])
    f_high = np.interp(-half_power, -frf_ampl[id_peak:], freq[id_peak:])

    damping = (f_high - f_low) / (2 * f_peak)

    return PeakPicking(
        frf_ampl=frf_ampl,
        id_peak=id_peak,
        ampl_peak=ampl_peak,
        f_peak=f_peak,
        f_low=f_low,
        f_high=f_high,
        half_power=half_power,
        damping=damping,
    )


class CircleFit(NamedTuple):
    """Computed quantities from the circle fit method."""

    mobility: Tuple[npt.NDArray]
    center: Tuple[float]
    radius: float
    damping_grid: Tuple[npt.NDArray]
    damping: float


def circle_fit(x: npt.NDArray, y: npt.NDArray):
    """Circle fitting via simple least square method.

    Parameters
    ----------
    x, y: 1D array
        x and y coordinates of the points to fit.

    Returns
    -------
    xc, yc: float
        Coordinates of the center of the fitted circle.
    radius: float
        Radius of the fitted circle.
    """

    def func(center: Tuple, x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        """Distances between the points (x, y) and the circle centered at (xc, yc)."""
        xc, yc = center
        radii = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        mean = np.mean(radii)
        return radii - mean

    centroid = x.mean(), y.mean()  # centroid as inital guess
    sol = least_squares(func, centroid, args=(x, y))
    xc, yc = sol.x

    radius = np.mean(np.sqrt((x - xc) ** 2 + (y - yc) ** 2))

    return xc, yc, radius


def circle_fit_method(freq, frf_sdof) -> CircleFit:
    """Get damping of SDOF model via the circle fit method."""
    # Compute the mobility: speed/force ratio
    mobility_sdof = frf_sdof / (1j * freq)
    x, y = mobility_sdof.real, mobility_sdof.imag

    xc, yc, radius = circle_fit(x, y)

    peak_picking = peak_picking_method(freq, frf_sdof)
    id_peak = peak_picking.id_peak
    f_peak = peak_picking.f_peak

    # Mobility frequencies before and after the peak
    f_low, f_high = freq[:id_peak], freq[id_peak + 1 :]
    f_low_grid, f_high_grid = np.meshgrid(f_low, f_high)

    # Theta angles, computed from the center of the fitter circle.
    theta_low = -np.atan2((y[:id_peak] - yc), -(x[:id_peak] - xc))
    theta_high = np.atan2((y[id_peak + 1 :] - yc), -(x[id_peak + 1 :] - xc))
    theta_low_grid, theta_high_grid = np.meshgrid(theta_low, theta_high)

    damping_grid = (f_high_grid**2 - f_low_grid**2) / (
        2 * f_peak
        * (f_low_grid * np.tan(theta_low_grid/2) + f_high_grid * np.tan(theta_high_grid/2))
    )

    # Extract damping for the f_low and f_high closest to f_peak
    damping = damping_grid[0, -1]

    return CircleFit(
        mobility=(x, y),
        center=(xc, yc),
        radius=radius,
        damping_grid=(f_low_grid, f_high_grid, damping_grid),
        damping=damping,
    )
