"""sdof -- Single-degree-of-freedom identification techniques."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import peak_widths

def compute_frf(data):
    """Frequency response function (FRF).

    It is assumed that the `data` are recorded from a SIMO model
    of one input (hammer) and three outputs (accelerometers).
    The FRF matrix is then of size 3x1.
    """
    h_12 = data["H1_2"][:, -1]
    h_13 = data["H1_3"][:, -1]
    h_14 = data["H1_4"][:, -1]

    return np.vstack((h_12, h_13, h_14))


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
    return np.sum(np.real(np.conj(frf)*frf), axis=0)


def plot_cmif(cmif, freq):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(freq, cmif)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("CMIF (+units)")
    ax.set_yscale('log')

    fig.show()


def peak_picking(frf, freq, id_acc=1):
    frf_ampl = np.abs(frf[id_acc, :])
    fn_idx = np.argmax(frf_ampl)
    halfpeaks_pitch = peak_widths(frf_ampl, [fn_idx], rel_height=1/np.sqrt(2))
    f1, f2 = np.interp(
        [halfpeaks_pitch[2][0], halfpeaks_pitch[3][0]],
        np.arange(0, len(freq)),
        freq
    )
    half_power = halfpeaks_pitch[1][0]

    return half_power


def circle_fit(frf):
    pass
