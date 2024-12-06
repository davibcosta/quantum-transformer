import os
import json
import csv
import numpy as np
from math import pi
from typing import Any, Tuple
import flax.serialization

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

import optax
from scipy.sparse.linalg import eigsh

import netket as nk
import netket.experimental as nkx

#------------------------------------
# Helper Functions
#------------------------------------

def find_closest_previous_state(L: int, g: float, base_dir: str) -> str:
    """
    Finds the path to the closest previous state based on the coupling constant g.

    Args:
        L (int): Lattice size.
        g (float): Coupling constant for the current state.
        base_dir (str): Base directory where state files are stored.

    Returns:
        str: Path to the closest state file, or None if no valid state file is found.
    """
    closest_g = None
    closest_g_path = None
    min_diff = float("inf")
    
    # Scan the base directory for state files
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.startswith(f"params_L{L}_g"):
                g_str = filename.split("_g")[1].replace(".mpack", "")
                try:
                    g_val = float(g_str)
                    # Compute absolute difference between g and g_val
                    diff = abs(g - g_val)
                    if diff < min_diff and diff != 0:  # Ignore the current state
                        closest_g = g_val
                        closest_g_path = os.path.join(dirpath, filename)
                        min_diff = diff
                except ValueError:
                    continue  # Ignore files with invalid g-values

    if closest_g_path:
        print(f"Closest state for L={L}, g={g}: {closest_g_path} (g = {closest_g:.4f})")
    else:
        print(f"No previous state found for L={L}, g={g}.")
    
    return closest_g, closest_g_path



def build_hamiltonian(L: int, g: float, J: float = -1.0) -> Tuple[nk.operator.GraphOperator, nk.hilbert.AbstractHilbert]:
    """
    Constructs the Hamiltonian for the 1D Transverse Field Ising Model (TFIM).

    Args:
        L (int): Lattice size.
        g (float): Coupling constant.
        J (float, optional): Transverse field strength. Defaults to -1.0.

    Returns:
        Tuple[nk.operator.GraphOperator, nk.hilbert.AbstractHilbert]: The Hamiltonian and Hilbert space.
    """
    hi = nk.hilbert.Spin(s=1 / 2, N=L, inverted_ordering=False)
    H = sum([J * g * nk.operator.spin.sigmax(hi, i) for i in range(L)])
    H += sum([J * nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L)])
    return H, hi

def exact_ground_state_energy_and_correlation(
    H: nk.operator.GraphOperator,
    corr_operator: nk.operator.GraphOperator,
) -> Tuple[float, float, float]:
    """
    Computes the exact ground state energy and the two-point correlation function
    using sparse diagonalization.

    Args:
        H (nk.operator.GraphOperator): The Hamiltonian.
        corr_operator (nk.operator.GraphOperator): The two-point correlation operator.

    Returns:
        Tuple[float, float, float]: Exact ground state energy, exact two-point correlation function,
                                    and the variance of the two-point correlation.
    """
    # Convert Hamiltonian to sparse matrix
    sparse_h = H.to_sparse()

    # Compute the ground state eigenvector and eigenvalue
    eig_vals, eig_vecs = eigsh(sparse_h, k=1, which="SA")  # SA: smallest algebraic
    psi = eig_vecs[:, 0]
    exact_energy = float(eig_vals[0])

    # Convert the two-point correlation operator to a sparse matrix
    sparse_corr = corr_operator.to_sparse()

    # Compute expectation value and variance of the two-point correlation
    exact_corr = psi @ (sparse_corr @ psi)  # ⟨ψ|Corr|ψ⟩
    corr_squared = sparse_corr @ sparse_corr
    exact_corr_squared = psi @ (corr_squared @ psi)  # ⟨ψ|Corr²|ψ⟩
    variance_corr = exact_corr_squared - exact_corr**2  # Var(Corr) = ⟨Corr²⟩ - ⟨Corr⟩²

    return exact_energy, float(exact_corr), float(variance_corr)

def linear_solver_pinv_smooth(rcond: float = 1e-6):
    """
    Returns a linear solver function with pinv_smooth.

    Args:
        rcond (float, optional): Regularization condition. Defaults to 1e-6.

    Returns:
        Callable: The linear solver function.
    """
    return lambda A, b: nk.optimizer.solver.pinv_smooth(A, b, rtol=rcond, rtol_smooth=rcond)[0]

def build_magnetization(L: int, hi: nk.hilbert.AbstractHilbert) -> nk.operator.GraphOperator:
    """
    Constructs the total Sz operator for magnetization.

    Args:
        L (int): Lattice size.
        hi (nk.hilbert.AbstractHilbert): The Hilbert space.

    Returns:
        nk.operator.Operator: The total Sz operator.
    """
    Sz = sum([nk.operator.spin.sigmaz(hi, i) for i in range(L)])
    return Sz

def measure_magnetization(L: int, hilbert: nk.hilbert.AbstractHilbert, vstate: nk.vqs.MCState) -> Tuple[float, float]:
    """
    Measures the mean and variance of magnetization along z.

    Args:
        L (int): Lattice size.
        hilbert (nk.hilbert.AbstractHilbert): The Hilbert space of the system.
        vstate (nk.vqs.MCState): The variational state.

    Returns:
        Tuple[float, float]: Mean magnetization and its variance.
    """
    # increase number of samples to lower variance 
    n_samples = vstate.n_samples
    vstate.n_samples = 409600
    
    Sz = build_magnetization(L, hilbert)  # Explicitly pass Hilbert space
    expectation = vstate.expect(Sz)
    
    # Mean magnetization normalized by lattice size
    mean_mz = expectation.mean.real / L
    
    # Variance of magnetization normalized by lattice size squared
    var_mz = expectation.variance.real / (L**2)
    
    # reset number of samples
    vstate.n_samples = n_samples
    
    return mean_mz, var_mz

def build_two_point_correlation_operator(
    L: int, 
    hilbert: nk.hilbert.AbstractHilbert
) -> nk.operator.LocalOperator:
    """
    Constructs the two-point correlation operator ⟨σz_i σz_{i+L/2}⟩.

    Args:
        L (int): Lattice size.
        hilbert (nk.hilbert.AbstractHilbert): The Hilbert space of the system.

    Returns:
        nk.operator.LocalOperator: Two-point correlation operator.
    """
    return sum(
        [
            nk.operator.spin.sigmaz(hilbert, i) * nk.operator.spin.sigmaz(hilbert, (i + L // 2) % L)
            for i in range(L)
        ]
    )


def measure_two_point_correlation(
    L: int,
    hilbert: nk.hilbert.AbstractHilbert,
    vstate: nk.vqs.MCState
) -> Tuple[float, float]:
    """
    Measures the two-point correlation function ⟨σz_i σz_{i+L/2}⟩ and its variance.

    Args:
        L (int): Lattice size.
        hilbert (nk.hilbert.AbstractHilbert): The Hilbert space of the system.
        vstate (nk.vqs.MCState): The variational state.

    Returns:
        Tuple[float, float]: Mean two-point correlation function and its variance.
    """
    # increase number of samples to lower variance 
    n_samples = vstate.n_samples
    vstate.n_samples = 409600
    
    # Build the two-point correlation operator
    two_point_op = build_two_point_correlation_operator(L, hilbert)

    # Measure the expectation value
    expectation = vstate.expect(two_point_op)

    # Reset the number of samples
    vstate.n_samples = n_samples

    # Return normalized mean and variance
    return expectation.mean.real / L, expectation.variance.real / (L**2)




def measure_energy(hamiltonian: nk.operator.GraphOperator, vstate: nk.vqs.MCState) -> Tuple[float, float]:
    """
    Measures the mean and variance of energy.

    Args:
        hamiltonian (nk.operator.GraphOperator): The Hamiltonian of the system.
        vstate (nk.vqs.MCState): The variational state.

    Returns:
        Tuple[float, float]: Mean energy and its variance.
    """

    energy = vstate.expect(hamiltonian)  # Explicitly pass Hamiltonian
    
    # Mean energy
    mean_e = energy.mean.real
    
    # Variance of energy
    var_e = energy.variance.real

    # Since hamiltonian has zero-variance principle we dont need many samples 
    
    return mean_e, var_e


def save_vstate(vstate, file_path):
    """
    Save the variational state to a file in mpack format.

    Args:
        vstate (nk.vqs.MCState): The variational state to save.
        file_path (str): Path to save the serialized state.
    """
    import flax.serialization
    
    with open(file_path, "wb") as file:
        file.write(flax.serialization.to_bytes(vstate))
    print(f"Variational state saved to {file_path}")


def load_vstate(file_path, vstate):
    """
    Load the variational state from a file in mpack format.

    Args:
        file_path (str): Path to the serialized state file.
        vstate (nk.vqs.MCState): A new variational state instance with matching architecture.

    Returns:
        nk.vqs.MCState: The loaded variational state with parameters restored.
    """
    import flax.serialization
    
    with open(file_path, "rb") as file:
        vstate = flax.serialization.from_bytes(vstate, file.read())
    print(f"Variational state loaded from {file_path}")
    return vstate


#------------------------------------
# Model Definition
#------------------------------------

class ViTNQS(nn.Module):
    Lx: int  # Lattice size in x direction (total sites for 1D input)
    patch_size: int
    d_model: int  # Embedding dimension
    num_heads: int
    num_layers: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, σ):
        # σ: (batch_size, N_sites)
        # N_sites = Lx
        batch_size = σ.shape[0]
        N_sites = σ.shape[1]

        # Extract patches: (batch_size, n_patches, patch_size)
        patches = extract_patches_1d(σ, self.patch_size)

        # Flatten patches: (batch_size, n_patches, patch_size)
        # No reshaping required beyond patch extraction

        # Linear embedding of patches into d-dimensional space
        patch_embedding = nn.Dense(self.d_model, use_bias=True, param_dtype=self.param_dtype, name='patch_embedding')
        x = patch_embedding(patches)  # (batch_size, n_patches, d_model)

        # Positional encoding
        n_patches = x.shape[1]
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.05), (1, n_patches, self.d_model), self.param_dtype)
        x = x + pos_embedding

        # ViT encoder blocks
        for _ in range(self.num_layers):
            x = ViTEncoderBlock(d_model=self.d_model, num_heads=self.num_heads, param_dtype=self.param_dtype)(x)

        # Hidden representation z
        z = jnp.sum(x, axis=1)  # (batch_size, d_model)

        # Final mapping to complex logarithm of amplitude
        # Parameters w, b
        w = self.param('w', nn.initializers.normal(stddev=0.1), (self.d_model,), jnp.complex64)
        b = self.param('b', nn.initializers.normal(stddev=0.1), (self.d_model,), jnp.complex64)

        # Compute pre-activation: (batch_size, d_model)
        pre_activation = w * z + b

        # Apply g(·) = logcosh(·)
        g = lambda x: jnp.log(jnp.cosh(x))

        # Handle complex inputs in logcosh
        log_psi = jnp.sum(g(pre_activation), axis=-1)  # (batch_size,)
        return log_psi


def extract_patches_1d(σ, patch_size):
    """
    Extract patches for 1D input without reshaping.
    Args:
        σ: (batch_size, Lx), where Lx is the lattice size.
        patch_size: Size of each patch.
    Returns:
        patches: (batch_size, n_patches, patch_size), where n_patches = Lx // patch_size.
    """
    batch_size, Lx = σ.shape
    n_patches = Lx // patch_size

    # Truncate σ to ensure full patches
    σ = σ[:, :n_patches * patch_size]

    # Split into patches along the last axis
    patches = σ.reshape(batch_size, n_patches, patch_size)
    return patches

class ViTEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, x):
        # Pre-Layer Normalization
        y = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        # Factored Multi-Head Attention
        y = FactoredMultiHeadAttention(num_heads=self.num_heads, d_model=self.d_model, param_dtype=self.param_dtype)(y)
        # Skip connection
        x = x + y

        # Pre-Layer Normalization
        y = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        # Feedforward network
        y = nn.Dense(4*self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        #y = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        # Skip connection
        x = x + y

        return x

class FactoredMultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, x):
        batch_size, n_patches, _ = x.shape
        head_dim = self.d_model // self.num_heads

        # Linear projection for V: Values
        V = nn.DenseGeneral((self.num_heads, head_dim), axis=-1, dtype=self.param_dtype, param_dtype=self.param_dtype, use_bias=False, name='V_proj')(x)
        V = V.transpose(0, 2, 1, 3)  # (batch_size, num_heads, n_patches, head_dim)

        # Compute attention weights αF_ij = p_{i-j}
        positions = jnp.arange(n_patches)
        relative_positions = positions[None, :] - positions[:, None]  # (n_patches, n_patches)

        # Positional biases p: (num_heads, 2 * n_patches - 1)
        p = self.param('p', nn.initializers.normal(stddev=0.02), (self.num_heads, 2 * n_patches - 1), self.param_dtype)

        # Map relative positions to indices
        relative_position_indices = relative_positions + n_patches - 1  # Shift indices to be >=0

        # Get attention weights: (num_heads, n_patches, n_patches)
        attention_weights = p[:, relative_position_indices]

        # Expand to include batch dimension: (batch_size, num_heads, n_patches, n_patches)
        attention_weights = jnp.broadcast_to(attention_weights[None, :, :, :], (batch_size, self.num_heads, n_patches, n_patches))

        # # Softmax over the key dimension
        # attention_weights = nn.softmax(attention_weights, axis=-1)

        # Compute attention output
        attention_output = jnp.matmul(attention_weights, V)  # (batch_size, num_heads, n_patches, head_dim)

        # Reshape and project back to d_model
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, n_patches, self.d_model)
        output = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype, use_bias=False, name='output_proj')(attention_output)
        
        return output

#------------------------------------
# VMC Execution Function
#------------------------------------

# def run_vmc_for_coupling(
#     L: int,
#     g: float,
#     base_dir: str,
#     machine: nn.Module,
#     optimizer: optax.GradientTransformation,
#     rcond: float,
#     diag_shift: float,
#     n_chains: int,
#     n_samples: int,
#     chunk_size: int,
#     iterations: int,
#     timeout: int,
#     max_norm: float,
#     rng_key: jax.random.PRNGKey
# ) -> Tuple[str, float, float, float, float, float]:
#     """
#     Runs VMC optimization for a given lattice size L and coupling g.

#     Args:
#         L (int): Lattice size.
#         g (float): Coupling constant.
#         base_dir (str): Base directory to save results.
#         machine (nn.Module): The neural network model.
#         optimizer (optax.GradientTransformation): Optimizer.
#         rcond (float): Regularization condition for linear solver.
#         diag_shift (float): Diagonal shift for linear solver.
#         n_chains (int): Number of Monte Carlo chains.
#         n_samples (int): Number of samples.
#         chunk_size (int): Chunk size for sampling.
#         iterations (int): Maximum number of iterations.
#         timeout (int): Timeout in seconds.
#         max_norm (float): Maximum gradient norm for clipping.
#         rng_key (jax.random.PRNGKey): RNG key for reproducibility.

#     Returns:
#         Tuple[str, float, float, float, float, float]: Path to saved parameters, exact energy, VMC energy, variance of energy, magnetization, variance of magnetization.
#     """
#     # Create directory for current run
#     run_dir = os.path.join(base_dir, f"L_{L}", f"g_{g:.4f}")
#     os.makedirs(run_dir, exist_ok=True)

#     print(f"Starting VMC for L={L}, g={g:.4f}")

#     # Build Hamiltonian and Hilbert space
#     H, hi = build_hamiltonian(L, g)

#     # Compute exact ground state energy
#     E_exact = exact_ground_state_energy(H)

#     # Initialize sampler
#     sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, dtype=jnp.int8)

#     # Define the path to the current mpack file
#     mpack_path = os.path.join(run_dir, f"params_L{L}_g{g:.4f}.mpack")
    
#     # Initialize variational state
#     vstate = nk.vqs.MCState(
#         sampler=sampler,
#         model=machine,
#         n_samples=n_samples,
#         n_discard_per_chain=0,
#         chunk_size=chunk_size,
#         seed=rng_key
#     )
    
#     # Find the closest coupling `g_prev`
#     closest_g = None
#     closest_g_path = None
#     min_diff = float("inf")
    
#     # Iterate over all files in the base directory to find g-values
#     for dirpath, _, filenames in os.walk(base_dir):
#         for filename in filenames:
#             if filename.startswith(f"params_L{L}_g"):
#                 g_str = filename.split("_g")[1].replace(".mpack", "")
#                 try:
#                     g_val = float(g_str)
#                     # Compute absolute difference between g and g_val
#                     diff = abs(g - g_val)
#                     if diff < min_diff: 
#                         closest_g = g_val
#                         closest_g_path = os.path.join(dirpath, filename)
#                         min_diff = diff
#                 except ValueError:
#                     continue  # Ignore files with invalid g-values
    
#     # Debugging: print all detected g-values
#     #print(f"Detected g-values and paths for L={L}, g={g}:")
#     for dirpath, _, filenames in os.walk(base_dir):
#         for filename in filenames:
#             if filename.startswith(f"params_L{L}_g"):
#                 g_str = filename.split("_g")[1].replace(".mpack", "")
#                 try:
#                     g_val = float(g_str)
#                     print(f"g = {g_val:.4f}, path = {os.path.join(dirpath, filename)}")
#                 except ValueError:
#                     continue
    
#     # Apply transfer learning if a previous state is found
#     if closest_g is not None and closest_g_path is not None:
#         print(f"Closest state for L={L}, g={g}: {closest_g_path} (g = {closest_g:.4f})")
#         try:
#             vstate = load_vstate(closest_g_path, vstate)
#             print(f"Loaded parameters from {closest_g_path} for transfer learning.")
#         except Exception as e:
#             print(f"Failed to load parameters from {closest_g_path}: {e}")
#     else:
#         print(f"No previous state found for L={L}, g={g}.")

#     # Define optimizer with gradient clipping
#     optimizer_chain = optax.chain(
#         optax.zero_nans(),
#         optax.clip_by_global_norm(max_norm=max_norm),
#         optimizer
#     )

#     # Initialize VMC driver
#     gs = nkx.driver.VMC_SRt(
#         H,
#         optimizer_chain,
#         diag_shift=diag_shift,
#         variational_state=vstate,
#         jacobian_mode="complex",
#         linear_solver_fn=linear_solver_pinv_smooth(rcond=rcond)
#     )

#     # Set target convergence threshold
#     target = E_exact + np.abs(E_exact) * 1e-4

#     # Define ConvergenceStopping callback
#     convergence_callback = nk.callbacks.ConvergenceStopping(
#         target=target,
#         monitor="mean",
#         smoothing_window=5,
#         patience=1
#     )

#     # Run VMC optimization
#     gs.run(
#         n_iter=iterations,
#         out=os.path.join(run_dir, "state"),
#         callback=[
#             nk.callbacks.Timeout(timeout=timeout),
#             nk.callbacks.InvalidLossStopping(monitor="mean", patience=1),
#             convergence_callback
#         ]
#     )

#     # Save final state parameters
#     save_vstate(vstate, mpack_path)

#     # Measure energy and magnetization
#     Mz, var_Mz = measure_magnetization(L, hi, vstate)
#     E_vmc, var_E = measure_energy(H, vstate)

#     # Ensure all values are real
#     E_vmc_real = float(E_vmc.real)
#     var_E_real = float(var_E.real)
#     Mz_real = float(Mz)
#     var_Mz_real = float(var_Mz)

#     # Compute relative error
#     rel_error = abs((E_vmc_real - E_exact) / (abs(E_exact) if E_exact != 0 else 1.0))

#     # Log summary
#     summary = {
#         "L": L,
#         "g": g,
#         "E_exact": E_exact,
#         "E_vmc_real": E_vmc_real,
#         "Var_E": var_E_real,
#         "Mz_real": Mz_real,
#         "Var_Mz_real": var_Mz_real,
#         "rel_error": rel_error
#     }

#     # Save summary to JSON for individual run
#     with open(os.path.join(run_dir, "summary.json"), "w") as f:
#         json.dump(summary, f, indent=4)

#     print(f"Completed VMC for L={L}, g={g:.4f}")
#     print(f"Energy: Exact={E_exact:.6f}, VMC={E_vmc_real:.6f}, Rel Error={rel_error:.6e}")
#     print(f"Magnetization: Mz={Mz_real:.6f}, Variance={var_Mz_real:.6e}\n")

#     return mpack_path, E_exact, E_vmc_real, var_E_real, Mz_real, var_Mz_real

#------------------------------------
# VMC Execution Function
#------------------------------------
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
        Tuple[str, float, float, float, float, float, float, float]:
            Path to saved parameters, exact energy, VMC energy, variance of energy,
            mean two-point correlation, variance of two-point correlation,
            exact two-point correlation, variance of exact two-point correlation.
    """
    # Create directory for current run
    run_dir = os.path.join(base_dir, f"L_{L}", f"g_{g:.4f}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Starting VMC for L={L}, g={g:.4f}")

    # Build Hamiltonian and Hilbert space
    H, hilbert = build_hamiltonian(L, g)

    # Build the two-point correlation operator
    two_point_op = build_two_point_correlation_operator(L, hilbert)

    # Compute exact ground state energy and exact two-point correlation
    E_exact, exact_corr, exact_corr_variance = exact_ground_state_energy_and_correlation(H, two_point_op)
    exact_corr, exact_corr_variance = exact_corr / L, exact_corr_variance / (L**2) # normalize appropriately 
    
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
        patience=1
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

    return (
        mpack_path,
        E_exact,
        E_vmc_real,
        var_E_real,
        mean_corr_real,
        var_corr_real,
        exact_corr,
        exact_corr_variance,
    )



#------------------------------------
# Model Info Function
#------------------------------------

def print_model_info(model: nn.Module, input_batch: jnp.ndarray, rng_key: jax.random.PRNGKey):
    """
    Prints the model architecture and parameter count.

    Args:
        model (nn.Module): The neural network model.
        input_batch (jnp.ndarray): A dummy input batch for initialization.
        rng_key (jax.random.PRNGKey): RNG key for initialization.
    """
    params = model.init(rng_key, input_batch)
    dtypes = jax.tree_util.tree_map(lambda x: x.dtype, params)
    print("Parameter Data Types:")
    print(dtypes)
    print("\nModel Architecture:")
    print(model)
    total_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(params))
    print(f"\nTotal Trainable Parameters: {int(total_params)}\n")


#------------------------------------
# Main function
#------------------------------------

if __name__ == "__main__":

    # Define system sizes
    L_values = [12, 16, 20]

    # Define coupling scans
    g_coarse = np.linspace(2, 0.0, 201)

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
            "Var_Two_Point_Correlation", 
            "Exact_Two_Point_Correlation", 
            "Exact_Two_Point_Correlation_Variance",
            "rel_error"
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
        print("Performing scan from g=2.0 to g=0\n")
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
                    mean_corr_real, var_corr_real, 
                    exact_corr, exact_corr_variance,
                    abs((E_vmc_real - E_exact) / E_exact)
                ])

print("Scanning completed. Summary saved to:", summary_csv)