"""mdof -- Multi-degree-of-freedom identification techniques."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy.typing as npt
import numpy as np


class PoleStatus(Enum):
    """Pole stabilization possible status."""

    o = auto()  # New pole detected
    f = auto()  # Frequency matches with previous pole (1%)
    d = auto()  # Damping matches with previous pole (5%)
    s = auto()  # All match with previous pole


@dataclass
class Pole:
    """Ecapsulate data related to one FRF pole."""

    status: PoleStatus
    value: complex
    freq: float
    damping: float


@dataclass
class PolyMAX:
    """Solution of a polyMAX analysis."""

    order: int
    poles: List[Pole]


def polymax(freqs: npt.NDArray, frf: npt.NDArray, dt: float, n_q: int) -> PolyMAX:
    """Compute modal parameters via polyMAX FRF fitting.

    Parameters
    ----------
    freqs : (1 x n_f) numpy array of float
        Recorded frequency lines (Hz)
    frf : (n_o x n_i x n_f) numpy array of complex
        Recorded frequency response functions.
    dt : float
        Time resolution of the recoded data.
        That is, the inverse of the sampling frequency.
    n_q : int
        Polynomial order utilized.

    Returns
    ------
    PolyMAX
        Solution dataclass, mainly holding the computed poles.
    """

    # Problem dimensions
    n_o, n_i, n_f = frf.shape

    ## Least square solution of the reduced normal equations: M * a = b

    w = (2 * np.pi * freqs).reshape(-1, 1)  # TODO: check freq units
    q = np.arange(n_q + 1).reshape(1, -1)

    # No scalar weighting of the outputs are considered here.
    # This means that X_r matrices are constants across outputs.
    X = np.exp(1j * w * q * dt)

    def Y(r: int) -> npt.NDArray:
       return np.array([-np.kron(X[ix_f, :], frf[r, :, ix_f]) for ix_f in range(n_f)])

    def R(r: int) -> npt.NDArray:
        return np.real(np.conj(X).T @ X)

    def S(r: int) -> npt.NDArray:
        return np.real(np.conj(X).T @ Y(r))

    def T(r: int) -> npt.NDArray:
        return np.real(np.conj(Y(r)).T @ Y(r))

    def M_partial(r: int) -> npt.NDArray:
        return T(r) - np.dot(S(r).T, np.linalg.solve(R(r), S(r)))

    M_full = 2 * sum(M_partial(r) for r in range(n_o))

    # Avoid trivial solution: add constraint "alpha_q = indentity"
    M = M_full[:(n_i*n_q),:(n_i*n_q)]
    b = - M_full[:(n_i*n_q), (n_i*n_q):]

    # Least square solution
    a = np.linalg.lstsq(M, b)[0]

    print(f"{M_full.shape=}")
    print(f"{M.shape=}")
    print(f"{b.shape=}")
    print(f"{a.shape=}")

    ## Determine the resulting poles

    return None


def stabilization(frf: npt.NDArray) -> List[PolyMAX]:
    pass


def lsfd(poles):
    pass
