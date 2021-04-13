import sys
sys.path.append('d:/python37/lib/site-packages')
import numpy as np

def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def kr(matrices):
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)
    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))

def khatri_rao(matrices, skip_matrix=None, reverse=False):
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    n_columns = matrices[0].shape[1]

    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                i, matrix.ndim))
        if matrix.shape[1] != n_columns:
            raise ValueError('All matrices must have same number of columns!'
                             'Matrix {} has {} columns != {}.'.format(
                i, matrix.shape[1], n_columns))

    n_factors = len(matrices)

    if reverse:
        matrices = matrices[::-1]
        # Note: we do NOT use .reverse() which would reverse matrices even outside this function

    return kr(matrices)

def kruskal_to_tensor(factors):
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)


def HCS(T, sketch_dim):
    l = len(np.shape(T))
    assert len(sketch_dim) == l
    nk_list = list(np.shape(T))
    Hk_list = []
    sk_list = []
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
    return A, Hk_list, S


def DeHCS(T, Hk_list, S):
    l = len(np.shape(T))
    for mode in range(l):
        shape = list(np.shape(T))#[n1,n2,n3,n4,..,nk]
        shape[mode] = np.shape(Hk_list[mode])[0]#HK_shape=[nk, mk]
        T = unfold(T, mode)
        T = Hk_list[mode] @ T
        T = fold(T, mode, shape)#得到shape[n1,n2,n3,...,nk]
    T = S * T
    return T


def DeHCS_mode(T, Hk_list, S,mode_list):
    for mode in mode_list:
        shape = list(np.shape(T))
        shape[mode] = np.shape(Hk_list[mode])[0]
        T = unfold(T, mode)
        T = Hk_list[mode] @ T
        T = fold(T, mode, shape)
    return T





def tt(A, epsilon):
    """
    third-ordered tensor train decomposition
    :param A: n1 * n2 * n3
    :param epsilon: prescribed accuracy
    :return: cores g1, g2, g3, TT-ranks r0, r1, r2, r3
    """
    c = A
    r = [1, 1, 1, 1]
    n = [A.shape[0], A.shape[1], A.shape[2]]
    g = [1, 1, 1]
    for k in [1, 2]:
        i = k
        prod = 1
        while i <= 2:
            prod *= n[i]
            i += 1
        c = np.reshape(c, [r[k-1]*n[k-1], prod])
        U, S, V_tran = np.linalg.svd(c)
        s = np.diag(S)
        delta = epsilon * np.linalg.norm(S) / np.sqrt(2)
        r[k] = s.shape[0] - 1
        gamma = np.zeros(1)
        while gamma <= delta:
            gamma += s[r[k]][r[k]] ** 2
            r[k] -= 1
        r[k] += 1
        U = U[:, 0:r[k]]
        s = s[0:r[k], 0:r[k]]
        V_tran = V_tran[0:r[k], :]
        g[k-1] = np.reshape(U, [r[k-1], n[k-1], r[k]])
        c = np.reshape(np.dot(s, V_tran), [r[k], prod, r[k+1]])
    g[2] = c

    return g, r


def norm(tensor):
    return np.sum(tensor**2)
'''
A = np.random.rand(3, 4, 5)
B = np.random.rand(5, 6, 7)
unfold_C = fold(np.transpose(np.transpose(unfold(B, 0)).dot(unfold(A, 2))),0,shape=[3,4,6,7])
ein_C = np.einsum('ijk,klm->ijlm', A, B)
print(ein_C.shape)
print(norm(ein_C-unfold_C))
print(norm(unfold(ein_C,0)-np.transpose(np.transpose(unfold(B, 0)).dot(unfold(A, 2)))))
'''
g1 = np.random.rand(1, 4, 3)
g2 = np.random.rand(3, 3, 4)
g3 = np.random.rand(4, 5, 1)
rand_A = np.einsum('ail,ljm,mka->ijk', g1, g2, g3)
# rand_A = np.random.rand(4, 3, 5)
# print(rand_A)
epsilon = 1e-35
g, r = tt(rand_A, epsilon)
recover_A = np.einsum('ail,ljm,mka->ijk', g[0], g[1], g[2])
hcs_A, hk_list, s = HCS(recover_A, [2, 2, 2])
g, r = tt(hcs_A, epsilon)
#DeHCS g
g[0] = DeHCS_mode(g[0], hk_list, s, [1])
g[1] = DeHCS_mode(g[1], hk_list, s, [1])
g[2] = DeHCS_mode(g[2],hk_list,s, [1])
recover_A = np.einsum('ail,ljm,mka->ijk', g[0], g[1], g[2])
print('hsifjsdp')
print(recover_A.shape)
# print(rand_A)
print('='*80)
# print(recover_A)
print('*'*80)
# print(rand_A-recover_A)
print('$'*80)
print(np.mean(rand_A))
print(np.mean(recover_A))
print(((recover_A-rand_A) ** 2).mean())
