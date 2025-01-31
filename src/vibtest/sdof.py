"""sdof -- Single-degree of freedom (SDOF) identification techniques."""

import matplotlib.pyplot as plt
import numpy as np

from vibtest.util import labdata

DATA = labdata.extract_measure(1, 2)
TIME = np.real(DATA["X1"][:, 0])
FREQ = np.real(DATA["H1_2"][:, 0])


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
    it happens that the cmif is simply the square of the FRF norm.
    More details in the report.

    NOTE:
    The following solution is also valid,
    but slightly slower (import scipy.linalg):
    >>> return linalg.norm(frf, axis=0)**2
    """
    return np.sum(np.real(np.conj(frf)*frf), axis=0)


def plot_cmif(cmif):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(FREQ, cmif)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("CMIF (+units)")
    ax.set_yscale('log')

    fig.show()


if __name__ == "__main__":
    frf = compute_frf(DATA)
    cmif = compute_cmif(frf)
    plot_cmif(cmif)
