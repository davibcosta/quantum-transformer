import netket as nk
import numpy as np
import json
from math import pi
import jax
import jax.numpy as jnp

L = 4
# Build square lattice with nearest and next-nearest neighbor edges
lattice = nk.graph.Square(L, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=lattice.n_nodes, inverted_ordering=False)
# Heisenberg with coupling J=1.0 for nearest neighbors
# and J=0.5 for next-nearest neighbors
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.5])



input = hi.random_state(size=64, key=jax.random.PRNGKey(1))
input = jnp.array(input)
print(input)
print(input.shape)


# Print model architecture and parameter count
def print_model_info(model):
    """Prints the model architecture and parameter count."""
    key = jax.random.PRNGKey(0)
    params = model.init(key, input)
    print("Model Architecture:")
    print(model)
    total_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(params))
    print(f"Total Trainable Parameters: {total_params}")

# Find an approximate ground state
machine = nk.models.GCNN(
    symmetries=lattice,
    parity=1,
    layers=4,
    features=4,
    param_dtype=complex,
)

print_model_info(machine)

n_chains = 128
n_samples = 4096
n_discard_per_chain = 10  # Number of samples to discard per chain
chunk_size = 4096

sampler = nk.sampler.MetropolisExchange(hi, n_chains=1024, graph=lattice, d_max=L, dtype=jnp.int8)
vstate = nk.vqs.MCState(sampler=sampler,model=machine,n_samples=n_samples,n_discard_per_chain=n_discard_per_chain,chunk_size=chunk_size)

opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=1e-4, holomorphic=False)

gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
gs.run(n_iter=100, out="ground_state")

data = json.load(open("ground_state.log"))
print("Energy averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]))
print("Energy per site averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))
print("Energy std over last ten steps:", np.std(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))

from scipy.sparse.linalg import eigsh

eig_vals, eig_vecs = eigsh(H.to_sparse(), k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]