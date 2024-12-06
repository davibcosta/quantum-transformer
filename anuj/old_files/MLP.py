# Import necessary libraries
import netket.experimental as nkx
import functools
from functools import partial
import flax.linen as nn
import numpy as np
import jax.numpy as jnp
import flax
import optax
import csv
import numpy as np
import os
import netket.experimental as nkx
import sys 
from math import pi
import json 

os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
# Check NetKet installation and print version
import netket as nk
print(f"NetKet version: {nk.__version__}")
# Print available JAX devices for the current process
print(jax.devices())


L = 4
# Build square lattice with nearest and next-nearest neighbor edges
lattice = nk.graph.Square(L, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
# Heisenberg with coupling J=1.0 for nearest neighbors
# and J=0.5 for next-nearest neighbors
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.51])

sparse_ham = H.to_sparse()
sparse_ham.shape

from scipy.sparse.linalg import eigsh

eig_vals, eig_vecs = eigsh(sparse_ham, k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]
print("Ground state energy from Exact Diagonalization:", E_gs)

class FFN(nn.Module):
    alpha : int = 1
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.alpha * x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nk.nn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

model = FFN(alpha=2)
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=1024)

optimizer = nk.optimizer.Sgd(learning_rate=0.01)

sr = nk.optimizer.SR(diag_shift=1e-6, holomorphic=False)
# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=sr)

log=nk.logging.RuntimeLog()
gs.run(n_iter=500,out=log)

ffn_energy=vstate.expect(H)
error=abs((ffn_energy.mean-E_gs)/E_gs)
print("Optimized energy and relative error: ",ffn_energy,error)