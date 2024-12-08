import os
import json
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

import optax
from scipy.sparse.linalg import eigsh

import netket as nk
import netket.experimental as nkx


def linear_solver_pinv_smooth(rcond: float = 1e-6):
    """
    Returns a linear solver function with pinv_smooth.

    Args:
        rcond (float, optional): Regularization condition. Defaults to 1e-6.

    Returns:
        Callable: The linear solver function.
    """
    return lambda A, b: nk.optimizer.solver.pinv_smooth(A, b, rtol=rcond, rtol_smooth=rcond)[0]

    
def find_closest_previous_state(L: int, g: float, base_dir: str) -> Tuple[float, str]:
    closest_g = None
    closest_g_path = None
    min_diff = float("inf")
    
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.startswith(f"params_L{L}_g"):
                g_str = filename.split("_g")[1].replace(".mpack", "")
                try:
                    g_val = float(g_str)
                    diff = abs(g - g_val)
                    if diff < min_diff and diff != 0:
                        closest_g = g_val
                        closest_g_path = os.path.join(dirpath, filename)
                        min_diff = diff
                except ValueError:
                    continue

    return closest_g, closest_g_path


def build_hamiltonian(L: int, g: float, J: float = -1.0) -> Tuple[nk.operator.GraphOperator, nk.hilbert.AbstractHilbert]:
    hi = nk.hilbert.Spin(s=1 / 2, N=L, inverted_ordering=False)
    H = sum([J * g * nk.operator.spin.sigmax(hi, i) for i in range(L)])
    H += sum([J * nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L)])
    return H, hi


def exact_ground_state_energy_and_correlation(H, corr_operator) -> Tuple[float, float, float]:
    sparse_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sparse_h, k=1, which="SA")
    psi = eig_vecs[:, 0]
    exact_energy = float(eig_vals[0])
    
    sparse_corr = corr_operator.to_sparse()
    exact_corr = psi @ (sparse_corr @ psi)
    corr_squared = sparse_corr @ sparse_corr
    exact_corr_squared = psi @ (corr_squared @ psi)
    variance_corr = exact_corr_squared - exact_corr ** 2

    return exact_energy, float(exact_corr), float(variance_corr)


def build_two_point_correlation_operator(L: int, hilbert: nk.hilbert.AbstractHilbert) -> nk.operator.LocalOperator:
    return sum([
        nk.operator.spin.sigmaz(hilbert, i) * nk.operator.spin.sigmaz(hilbert, (i + L // 2) % L)
        for i in range(L)
    ])


def measure_energy(hamiltonian, vstate) -> Tuple[float, float]:
    energy = vstate.expect(hamiltonian)
    return energy.mean.real, energy.variance.real


def measure_two_point_correlation(L: int, hilbert: nk.hilbert.AbstractHilbert, vstate: nk.vqs.MCState) -> Tuple[float, float]:
    n_samples = vstate.n_samples
    vstate.n_samples = 409600
    two_point_op = build_two_point_correlation_operator(L, hilbert)
    expectation = vstate.expect(two_point_op)
    vstate.n_samples = n_samples
    return expectation.mean.real / L, expectation.variance.real / (L ** 2)


def save_vstate(vstate, file_path):
    import flax.serialization
    with open(file_path, "wb") as file:
        file.write(flax.serialization.to_bytes(vstate))
    print(f"Variational state saved to {file_path}")


def load_vstate(file_path, vstate):
    import flax.serialization
    with open(file_path, "rb") as file:
        vstate = flax.serialization.from_bytes(vstate, file.read())
    print(f"Variational state loaded from {file_path}")
    return vstate
