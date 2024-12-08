import os
import csv
import json
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax import linen as nn

import netket as nk
import netket.experimental as nkx

from helpers import *
from model import ViTNQS


def run_vmc_for_coupling(
    L: int,
    g: float,
    base_dir: str,
    machine: nn.Module,
    optimizer: optax.GradientTransformation,
    rcond: float,
    diag_shift: float,
    n_chains: int,
    n_samples: int,
    chunk_size: int,
    iterations: int,
    timeout: int,
    max_norm: float,
    rng_key: jax.random.PRNGKey
) -> Tuple[str, float, float, float, float, float, float, float]:
    """
    Runs VMC optimization for a given lattice size L and coupling g.

    Args:
        L (int): Lattice size.
        g (float): Coupling constant.
        base_dir (str): Base directory to save results.
        machine (nn.Module): The neural network model.
        optimizer (optax.GradientTransformation): Optimizer.
        rcond (float): Regularization condition for linear solver.
        diag_shift (float): Diagonal shift for linear solver.
        n_chains (int): Number of Monte Carlo chains.
        n_samples (int): Number of samples.
        chunk_size (int): Chunk size for sampling.
        iterations (int): Maximum number of iterations.
        timeout (int): Timeout in seconds.
        max_norm (float): Maximum gradient norm for clipping.
        rng_key (jax.random.PRNGKey): RNG key for reproducibility.

    Returns:
        Tuple[str, float, float, float, float, float, float, float]: Path to saved parameters, 
        exact energy, VMC energy, variance of energy, mean two-point correlation function, 
        variance of two-point correlation, exact two-point correlation, variance of exact correlation.
    """
    # Create directory for current run
    run_dir = os.path.join(base_dir, f"L_{L}", f"g_{g:.4f}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Starting VMC for L={L}, g={g:.4f}")

    # Build Hamiltonian and Hilbert space
    H, hilbert = build_hamiltonian(L, g)
    two_point_corr_operator = build_two_point_correlation_operator(L, hilbert)

    # Compute exact ground state energy and two-point correlation
    E_exact, exact_corr, exact_corr_variance = exact_ground_state_energy_and_correlation(H, two_point_corr_operator)

    # Normalize Two-point correlation function 
    exact_corr, exact_corr_variance = exact_corr / L, exact_corr_variance / (L**2)

    # Initialize sampler
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)

    # Define the path to the current mpack file
    mpack_path = os.path.join(run_dir, f"params_L{L}_g{g:.4f}.mpack")

    # Initialize variational state
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=machine,
        n_samples=n_samples,
        n_discard_per_chain=0,
        chunk_size=chunk_size,
        seed=rng_key
    )

    # Find and load closest previous state
    closest_g, closest_g_path = find_closest_previous_state(L, g, base_dir)
    if closest_g_path:
        print(f"Closest state for L={L}, g={g}: {closest_g_path} (g = {closest_g:.4f})")
        try:
            vstate = load_vstate(closest_g_path, vstate)
            print(f"Loaded parameters from {closest_g_path} for transfer learning.")
        except Exception as e:
            print(f"Failed to load parameters from {closest_g_path}: {e}")
    else:
        print(f"No previous state found for L={L}, g={g}.")

    # Define optimizer with gradient clipping
    optimizer_chain = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(max_norm=max_norm),
        optimizer
    )

    # Initialize VMC driver
    gs = nkx.driver.VMC_SRt(
        H,
        optimizer_chain,
        diag_shift=diag_shift,
        variational_state=vstate,
        jacobian_mode="complex",
        linear_solver_fn=linear_solver_pinv_smooth(rcond=rcond)
    )

    # Set target convergence threshold
    rel_target = 1e-6
    target = E_exact + np.abs(E_exact) * rel_target

    # Define ConvergenceStopping callback
    convergence_callback = nk.callbacks.ConvergenceStopping(
        target=target,
        monitor="mean",
        smoothing_window=5,
        patience=5
    )

    # Run VMC optimization
    gs.run(
        n_iter=iterations,
        out=os.path.join(run_dir, "state"),
        callback=[
            nk.callbacks.Timeout(timeout=timeout),
            nk.callbacks.InvalidLossStopping(monitor="mean", patience=1),
            convergence_callback
        ]
    )

    # Save final state parameters
    save_vstate(vstate, mpack_path)

    # Measure energy
    E_vmc, var_E = measure_energy(H, vstate)

    # Measure two-point correlation function
    mean_corr, var_corr = measure_two_point_correlation(L, hilbert, vstate)

    # Ensure all values are real
    E_vmc_real = float(E_vmc.real)
    var_E_real = float(var_E.real)
    mean_corr_real = float(mean_corr)
    var_corr_real = float(var_corr)

    # Compute relative error
    rel_error = abs((E_vmc_real - E_exact) / (abs(E_exact) if E_exact != 0 else 1.0))

    # Log summary
    summary = {
        "L": L,
        "g": g,
        "E_exact": E_exact,
        "E_vmc_real": E_vmc_real,
        "Var_E": var_E_real,
        "Mean_Two_Point_Correlation": mean_corr_real,
        "Var_Two_Point_Correlation": var_corr_real,
        "Exact_Two_Point_Correlation": exact_corr,
        "Exact_Two_Point_Correlation_Variance": exact_corr_variance,
        "rel_error": rel_error
    }

    # Save summary to JSON for individual run
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Completed VMC for L={L}, g={g:.4f}")
    print(f"Energy: Exact={E_exact:.6f}, VMC={E_vmc_real:.6f}, Rel Error={rel_error:.6e}")
    print(f"Two-Point Correlation: Mean={mean_corr_real:.6f}, Variance={var_corr_real:.6e}")
    print(f"Exact Two-Point Correlation: {exact_corr:.6f}, Variance: {exact_corr_variance:.6e}\n")

    return mpack_path, E_exact, E_vmc_real, var_E_real, mean_corr_real, var_corr_real, exact_corr, exact_corr_variance


# Main Script Execution
if __name__ == "__main__":
    # Define system sizes
    L_values = [12, 16, 20]

    # Define coupling scans
    g_coarse = np.linspace(1.05, 0.95, 51)

    # Hyperparameters
    lr_initial = 0.1
    rcond = 1e-6
    diag_shift = 1e-4
    n_chains = 32
    n_samples = 4096
    chunk_size = 4096
    iterations = 500
    timeout = 1200  # seconds
    max_norm = 1.0

    # Initialize random key for reproducibility
    seed = 42
    rng = random.PRNGKey(seed)

    # Base directory for results
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    # Summary CSV file
    summary_csv = os.path.join(base_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "L", "g", "E_exact", "E_vmc_real",
            "Var_E", "Mean_Two_Point_Correlation",
            "Var_Two_Point_Correlation", "Exact_Two_Point_Correlation",
            "Exact_Two_Point_Correlation_Variance", "rel_error"
        ])

    # Iterate over system sizes
    for L in L_values:
        print(f"Starting scans for L={L}\n")
        # Initialize model
        machine = ViTNQS(
            Lx=L,
            patch_size=4,
            d_model=32,
            num_heads=2,
            num_layers=1,
            param_dtype=jnp.complex64
        )

        # Initialize model parameters
        rng, model_rng = random.split(rng)
        input_batch = jnp.ones((64, L))  # Dummy input for initialization
        params = machine.init(model_rng, input_batch)

        # Define optimizer
        lr_schedule = optax.cosine_decay_schedule(init_value=lr_initial, decay_steps=iterations)
        optimizer = optax.sgd(learning_rate=lr_schedule)

        # Coarse Scan
        print("Performing scan from g=1.05 to g=0.95\n")
        for g in g_coarse:
            rng, run_rng = random.split(rng)
            (
                params_path,
                E_exact,
                E_vmc_real,
                var_E_real,
                mean_corr_real,
                var_corr_real,
                exact_corr,
                exact_corr_variance
            ) = run_vmc_for_coupling(
                L=L,
                g=g,
                base_dir=base_dir,
                machine=machine,
                optimizer=optimizer,
                rcond=rcond,
                diag_shift=diag_shift,
                n_chains=n_chains,
                n_samples=n_samples,
                chunk_size=chunk_size,
                iterations=iterations,
                timeout=timeout,
                max_norm=max_norm,
                rng_key=run_rng
            )

            # Append to summary CSV
            with open(summary_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    L, g, E_exact, E_vmc_real, var_E_real,
                    mean_corr_real, var_corr_real, exact_corr,
                    exact_corr_variance, abs((E_vmc_real - E_exact) / E_exact)
                ])

    print("Scanning completed. Summary saved to:", summary_csv)
