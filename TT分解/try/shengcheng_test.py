import numpy as np
from scipy.linalg import orth

def compress(tensor, modematrices):
    U, V, W = modematrices
    I, J, K = tensor.shape
    L = U.shape[1]
    M = V.shape[1]
    N = W.shape[1]
    G1 = np.reshape(tensor, newshape=[I, J * K])
    G2 = np.reshape(U.T.__matmul__(G1), newshape=[L * J, K]).T
    G3 = np.reshape(W.T.__matmul__(G2), newshape=[L * N, J]).T
    G4 = np.reshape(V.T.__matmul__(G3), newshape=[M * N, L]).T
    return np.reshape(G4, newshape=[L, M, N])

I = 60
R = 10
J = 60
K = 60
U = np.random.normal(size=[I, R])
V = np.random.normal(size=[J, R])
W = np.random.normal(size=[K, R])

U = orth(U)
V = orth(V)
W = orth(W)

Y = compress(tensor, [U, V, W])

