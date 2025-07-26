"""mdof -- Multi-degree-of-freedom identification techniques."""

from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Pool
from typing import List

import numpy as np
import numpy.typing as npt


class PoleStatus(Enum):
    """Pole stabilization possible status."""

    o = auto()  # New pole detected
    f = auto()  # Frequency matches with previous pole
    d = auto()  # Damping matches with previous pole
    s = auto()  # All match with previous pole


@dataclass
class Pole:
    """Ecapsulate data related to one FRF pole."""

    status: PoleStatus
    value: complex
    freq: float  # [Hz]
    damping: float


@dataclass
class PolyMAX:
    """Solution of a polyMAX analysis."""

    order: int
    poles: List[Pole]


def polymax(freqs: npt.NDArray, frf: npt.NDArray, dt: float, q: int, debug=False) -> PolyMAX:
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
    q : int
        Polynomial order utilized.

    Returns
    ------
    PolyMAX
        Solution dataclass, mainly holding the computed poles.
    """

    # Problem dimensions
    n_o, n_i, n_f = frf.shape

    ## Least square solution of the reduced normal equations: M * x = b

    # No scalar weighting of the outputs are considered here.
    # This means that X_r matrices are constants across outputs.
    X_w = (2 * np.pi * freqs).reshape(-1, 1)
    X_q = np.arange(q + 1).reshape(1, -1)
    X = np.exp(1j * X_w * X_q * dt)

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

    # Avoid trivial solution: add constraint "a_q = indentity"
    M = M_full[: (n_i * q), : (n_i * q)]
    b = -M_full[: (n_i * q), (n_i * q) :]

    # Least square solution
    x = np.linalg.lstsq(M, b)[0]

    ## Determine the resulting poles

    upper_left = np.zeros((n_i * (q - 1), n_i))
    upper_right = np.identity(n_i * (q - 1))

    C = np.block([[upper_left, upper_right], [-x.T]])

    eigvals = np.linalg.eigvals(C)

    poles = np.log(eigvals) / dt

    dampings = -np.real(poles) / np.abs(poles)
    freqs = np.abs(poles) / (2 * np.pi)  # Hz

    ## Debug prints
    if debug:
        print(f"{M_full.shape=}")
        print(f"{M.shape=}")
        print(f"{b.shape=}")
        print(f"{x.shape=}")
        print(f"{C.shape=}")
        print(f"{freqs=}")
        print(f"{dampings=}")

    ## Build the returned solution

    pole_list = []
    for pole, freq, damping in zip(poles, freqs, dampings):
        pole_list.append(Pole(status=PoleStatus.o, value=pole, freq=freq, damping=damping))

    return PolyMAX(order=q, poles=pole_list)


def determine_statuses(sol: PolyMAX, sol_prev: PolyMAX, f_rtol=1e-2, d_rtol=5e-2) -> None:
    """Write in `sol.poles` the stabilization status of each poles."""

    # Fetch freqs and dampings of previous polymax solution
    freqs_prev = np.array([p_prev.freq for p_prev in sol_prev.poles])
    dampings_prev = np.array([p_prev.damping for p_prev in sol_prev.poles])

    for p in sol.poles:
        # Index in list of previous poles,
        # for the one that is the closest to the current pole `p`
        ix_f = np.argmin(np.abs((p.freq - freqs_prev) / freqs_prev))

        # Determine the stabilization status via relative tolerances
        if np.abs((p.freq - freqs_prev[ix_f]) / freqs_prev[ix_f]) <= f_rtol:
            p.status = PoleStatus.f
        if np.abs((p.damping - dampings_prev[ix_f]) / dampings_prev[ix_f]) <= d_rtol:
            if p.status == PoleStatus.f:
                p.status = PoleStatus.s
            else:
                p.status = PoleStatus.d

    return None


def _refresh_statuses(list_sol: List[PolyMAX]) -> None:
    for sol, sol_prev in zip(list_sol[1:], list_sol[:-1]):
        determine_statuses(sol, sol_prev)


def stabilization(
    freqs: npt.NDArray, frf: npt.NDArray, dt: float, n_q: int, *, debug=False
) -> List[PolyMAX]:
    """Build a stabilization diagram."""

    # Instantiate the list of polymax solution with order 2 solution
    list_sol = [polymax(freqs, frf, dt, 2, debug)]

    for q in range(4, n_q + 2, 2):
        sol = polymax(freqs, frf, dt, q, debug)
        determine_statuses(sol, list_sol[-1])
        list_sol.append(sol)

    return list_sol

def stabilization_parallel(
    freqs: npt.NDArray, frf: npt.NDArray, dt: float, n_q: int, *, debug=False
) -> List[PolyMAX]:
    """Build a stabilization diagram."""

    # Instantiate the list of polymax solution with order 2 solution
    list_sol = [polymax(freqs, frf, dt, 2, debug)]

    # Pool of processes, to run polymax for different orders in parallel
    with Pool() as pool:
        args = [(freqs, frf, dt, q, debug) for q in range(4, n_q + 2, 2)]
        results = pool.starmap(polymax, args)

    # Fill the list of polymax solutions
    for sol in results:
        determine_statuses(sol, list_sol[-1])
        list_sol.append(sol)

    return list_sol


def lsfd(poles):
    pass
