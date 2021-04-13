import numpy as np
import scipy.linalg as sl
import scipy
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from scipy.linalg import orth


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def kruskal_to_tensor(factors):
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)

def HCS(T, sketch_dim, return_s_atom=False, return_h_atom=False):
    l = len(np.shape(T))
    assert len(sketch_dim) == l
    nk_list = list(np.shape(T))
    Hk_list = []
    sk_list = []
    hk_list = []
    for k in range(len(nk_list)):
        nk = nk_list[k]
        mk = sketch_dim[k]
        hk = np.random.randint(low=0, high=mk, size=nk)
        sk = np.random.normal(size=[nk, 1])
        sk = np.sign(sk)
        Hk = np.zeros(shape=[nk, mk])
        x = np.arange(nk)
        # print(x)
        y = hk[x]
        hk_list.append(hk)
        Hk[x, y] = 1
        # print(len(np.nonzero(Hk)[0]))
        Hk_list.append(Hk)
        sk_list.append(sk)
    S = kruskal_to_tensor(sk_list)
    A = S * T
    for mode in range(l):
        shape = list(np.shape(A))
        A = unfold(A, mode)
        shape[mode] = sketch_dim[mode]
        A = Hk_list[mode].T @ A
        A = fold(A, mode, shape)
    ans = [A, Hk_list, S]
    if return_s_atom:
        ans.append(sk_list)
    if return_h_atom:
        ans.append(hk_list)
    return ans

def DeHCS(HCS_Y, Hk_list, sk_list, i):
    H = Hk_list[i].T
    HCS_Y = H @ HCS_Y
    #HCS_Y = HCS_Y * sk_list[i]
    for j in sk_list[i].shape[0]:
        for k in HCS_Y.shape[1]:
            HCS_Y[j][k]*=sk_list[i][j]
    return HCS_Y



if __name__ == '__main__':
    I = 100
    J = 100
    K = 100
    R = max(I, J, K) // 20
    creatA = np.random.normal(size=[I, R])
    creatB = np.random.normal(size=[J, R])
    creatC = np.random.normal(size=[K, R])
    creatA = orth(creatA)
    creatB = orth(creatB)
    creatC = orth(creatC)
    tensor = kruskal_to_tensor([creatA, creatB, creatC])
    sketch_dim = [20, 20, 20]
    Y,HK_list,S, sk_list  = HCS(tensor, sketch_dim, 1)




