import numpy as np
import scipy.linalg as sl
import scipy
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from scipy.linalg import orth


def prod(n: np.ndarray):
    ans = 1
    for elem in n:
        ans *= elem
    return int(ans)#返回n的所有元素的乘积


def kr(matrices):
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


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

    return kr(matrices)#除了实现kr(matrices)之外的一些操作


def norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))#求模


def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)


def kruskal_to_tensor(factors):
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)


def ttr1svd(tensor: np.ndarray):#tensor(I,J,K)
    n = np.shape(tensor)#(I,J,K)
    r = np.zeros((len(n) - 1,))#2维
    for i in range(len(r)):
        r[i] = min(n[i], prod(n[i + 1:]))#prod:返回所有元素的乘积
    # print(r)
    totalsvd = 1
    svdsuperlevel = np.zeros_like(r)
    svdsuperlevel[0] = 1
    for i in range(1, len(r)):
        svdsuperlevel[i] = prod(r[:i])
        totalsvd = totalsvd + svdsuperlevel[i]
    nleaf = prod(r)
    # print(svdsuperlevel)
    U = []
    S = []
    V = []
    [Ut, St, Vt] = np.linalg.svd(np.reshape(tensor, [n[0], prod(n[1:])]), full_matrices=False)
    Vt = Vt.T
    # print('-----------')
    # print(Ut.shape)
    # print(St.shape)
    # print(Vt.shape)
    # print('-----------')
    U.append(Ut)
    S.append(St.reshape((-1, 1)))
    V.append(Vt)
    counter = 1
    whichcounter = 0
    sigmas = kr([S[0], np.ones((int(nleaf // len(S[0])), 1))])
    # print(1)
    # print(sigmas.shape)
    for i in range(0, len(r) - 1):
        Slevel = []
        for j in range(prod(r[:i + 1])):
            # print(i, j)
            if (j + 1) % r[i] == 0:
                col = r[i] - 1
            else:
                col = (j + 1) % r[i] - 1
            col = int(col)
            # print('col:{}'.format(col))
            # print(V[whichcounter][:, col])
            # print(n)
            # print(V[whichcounter].shape)
            # print(whichcounter)
            # print(V[whichcounter][:, col].shape)
            [Ut, St, Vt] = np.linalg.svd(np.reshape(V[whichcounter][:, col], [n[i + 1], -1]), full_matrices=False)
            # print('v')
            Vt = Vt.T
            # print(Vt.shape)
            U.append(Ut)
            S.append(St.reshape((-1, 1)))
            V.append(Vt)
            if len(Slevel) == 0:
                Slevel = S[counter]
            else:
                # print(S[counter].shape)
                Slevel = np.row_stack([Slevel, S[counter]])
            counter += 1
            if (j + 1) % len(S[whichcounter]) == 0:
                whichcounter += 1
        Slevel = kr([Slevel, np.ones((int(nleaf / len(Slevel)), 1))])
        sigmas = sigmas * Slevel
    return U, S, V, sigmas


def ttr1(tensor: np.ndarray):
    shape = np.shape(tensor)
    svd_r = [shape[0], min(shape[1], shape[2])]
    U_i = []
    U_ij = []
    V_ij = []
    sig_i = []
    sig_ij = []
    A = np.reshape(tensor, [svd_r[0], -1])
    U, S, V = np.linalg.svd(A, full_matrices=False)
    pass


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


def compress_ff(tensor, compression_matrices):
    shape = list(tensor.shape)
    A = unfold(tensor, 0)
    A = compression_matrices[0] @ A
    shape[0] = A.shape[0]
    A = fold(A, 0, shape)

    A = unfold(tensor, 1)
    A = compression_matrices[1] @ A
    shape[1] = A.shape[0]
    A = fold(A, 1, shape)

    A = unfold(tensor, 2)
    A = compression_matrices[2] @ A
    shape[2] = A.shape[0]
    A = fold(A, 2, shape)

    return A


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


def form_mm_ttr1(U, V, Sigmas, size):
    I, J, K = size
    modeA = np.zeros(shape=[I, 1])
    modeB = np.zeros(shape=[J, 1])
    modeC = np.zeros(shape=[K, 1])
    for i in range(I):
        for j in range(min(J, K)):
            tmp = [np.reshape(U[0][:, i], [-1, 1]), np.reshape(U[i + 1][:, j], [-1, 1]),
                   np.reshape(V[i + 1][:, j], [-1, 1])]
            modeA = np.column_stack([modeA, tmp[0]])
            modeB = np.column_stack([modeB, tmp[1]])
            modeC = np.column_stack([modeC, tmp[2]])
    modeA = modeA[:, 1:]
    modeB = modeB[:, 1:]
    modeC = modeC[:, 1:]
    for col in range(modeA.shape[1]):
        modeA[:, col] *= Sigmas[col]
    return modeA, modeB, modeC


def permColmatch(iA, dA):
    """
    该方法依靠匈牙利算法，寻找一个列重排矩阵pai使得 L2 norm ||iA-dA@pai|| 尽可能小
    :param iA:目标矩阵
    :param dA: 待匹配矩阵
    :return: 列重排矩阵
    """
    iA = np.array(iA)
    dA = np.array(dA)
    pre = iA.T @ dA
    # print("permColmatch before trace:{}".format(np.trace(pre)))
    cost = np.max(pre) - pre
    LAProw, LAPcol = linear_sum_assignment(cost)
    _, F = iA.shape
    zeors = np.zeros([F, F])
    for row, col in zip(LAProw, LAPcol):
        zeors[row, col] = 1
    zeors = zeors.T
    # print("permColmatch after trace:{}".format(np.trace(pre @ zeors)))
    print("after perm anchor difference L2 norm:{}".format(np.linalg.norm(iA - dA @ zeors)))
    return zeors


if __name__ == "__main__":
    # I = J = K = 10
    I = 60
    J = 60
    K = 60
    R = max(I, J, K) // 5#最高维度
    #tensor = np.random.normal(size=[I, J, K])#0.922
    tensor = np.zeros([I, J, K])#0.87
    for r in range(R):
        u = np.random.normal(size=[I, 1])#u (T,1)
        v = np.random.normal(size=[J, 1])#v (J,1)
        w = np.random.normal(size=[K, 1])#w (K,1)
        u /= norm(u)#归一化
        v /= norm(v)
        w /= norm(w)
        s = np.random.randint(low=0, high=I)
        u *= s#随机初始化一个s
        tensor += kruskal_to_tensor([u, v, w])#(u,v,w)

    U, S, V, sigmas = ttr1svd(tensor)#tensor += kruskal_to_tensor([u, v, w])
    #print('U.shape', len(U), len(U[0]), len(U[0][0]))#61 60 60
    #print(U[0])
    #print('U[0][:, 0]', U[0][:, 0])#遍历U【0】,将U[0]的每一个的第一个存下来
    #print('S.shape', len(S), )
    #print('V.shape', len(V))

    #压缩成U， S， V
    #SVD后得到对应的矩阵

    true_sigmas = sigmas
    nleaf = I * min(J, K)
    leaf = []
    modeA = np.zeros(shape=[I, 1])#(60, 1)
    modeB = np.zeros(shape=[J, 1])
    modeC = np.zeros(shape=[K, 1])
    for i in range(I):#I: 60
        for j in range(min(J, K)):
            tmp = [np.reshape(U[0][ :,i], [-1, 1]), np.reshape(U[i + 1][:, j], [-1, 1]),
                   np.reshape(V[i + 1][:, j], [-1, 1])]
            #print(tmp[0].shape)#(60, 1)
            modeA = np.column_stack([modeA, tmp[0]])
            #if(j==0):
                #print('modeA.shape', modeA.shape)#(60, 2)
            #if(j==1):
                #print('modeB.shape', modeA.shape)#(60, 3)
            #每次对应的(60, 1)的列向量，堆叠到对应的mode中
            modeB = np.column_stack([modeB, tmp[1]])
            modeC = np.column_stack([modeC, tmp[2]])
            # modeA.append(U[0][:, i])
            # modeB.append(U[i + 1][:, j])
            # modeC.append(V[i + 1][:, j])
            leaf.append(tmp)
    print('modeA.shape_before', modeA.shape)#(60, 3601)
    modeA = modeA[:, 1:]
    modeB = modeB[:, 1:]
    modeC = modeC[:, 1:]
    print('modeA.shape', modeA.shape)#60, 3600
    # print(modeB.shape)
    # print(modeC.shape)
    for col in range(modeA.shape[1]):
        modeA[:, col] *= sigmas[col]
    estTensor = kruskal_to_tensor([modeA, modeB, modeC])#baseline的恢复值
    est = 0
    for i, node in enumerate(leaf):
        est += sigmas[i] * kruskal_to_tensor(node)
    print('vec est norm:{}'.format(norm(est - tensor) / norm(tensor)))
    print('mat est norm:{}'.format(norm(estTensor - tensor) / norm(tensor)))


    #之前是纯粹的TT-r1SVD
    #现在要加入压缩过程
    L = I // 5
    M = J // 5
    N = K // 5
    P = 60
    mergeA = 0
    mergeB = 0
    mergeC = 0
    compress_sigmas = []
    hash_eigvalue = []
    for p in range(P):
        U = np.random.normal(size=[I, L])
        V = np.random.normal(size=[J, M])
        W = np.random.normal(size=[K, N])

        U = orth(U)
        V = orth(V)
        W = orth(W)

        Y = compress(tensor, [U, V, W])#(U, V, W)是随机生成后正交化的解
        Ut, St, Vt, sigmast = ttr1svd(Y)
        sigmast = np.reshape(sigmast, -1)
        sigmast = list(sigmast)
        compress_sigmas.extend(list(np.reshape(sigmast, -1)))
        # print(sigmast.reshape(-1))
        # print(true_sigmas.reshape(-1))
        modeAt, modeBt, modeCt = form_mm_ttr1(Ut, Vt, sigmast, [L, M, N])
        estA = U @ modeAt
        estB = V @ modeBt
        estC = W @ modeCt
        if p == 0:
            for i, eig_value in enumerate(sigmast):
                add = True
                for elem_exits in hash_eigvalue:
                    if np.abs(eig_value - elem_exits) < 1e-2:
                        add = False
                        break
                if add:
                    hash_eigvalue.append(eig_value)
                    if mergeA is 0:
                        mergeA = estA[:, i]
                        mergeB = estB[:, i]
                        mergeC = estC[:, i]
                    else:
                        mergeA = np.column_stack([mergeA, np.reshape(estA[:, i], (-1, 1))])
                        mergeB = np.column_stack([mergeB, np.reshape(estB[:, i], (-1, 1))])
                        mergeC = np.column_stack([mergeC, np.reshape(estC[:, i], (-1, 1))])

        else:
            """mergeA = np.column_stack([mergeA, estA])
            mergeB = np.column_stack([mergeB, estB])
            mergeC = np.column_stack([mergeC, estC])"""

            for eig_index, eig_value in enumerate(sigmast):
                add = True
                like_elem = None
                for elem_exits in hash_eigvalue:
                    if np.abs(eig_value - elem_exits) < 1e-3:
                        add = False
                        like_elem = elem_exits
                        break
                if not add:
                    col = hash_eigvalue.index(like_elem)
                    mergeA[:, col] = (1 / 2) * (mergeA[:, col] + estA[:, eig_index])
                    mergeB[:, col] = (1 / 2) * (mergeB[:, col] + estB[:, eig_index])
                    mergeC[:, col] = (1 / 2) * (mergeC[:, col] + estC[:, eig_index])
                else:
                    mergeA = np.column_stack([mergeA, np.reshape(estA[:, eig_index], (-1, 1))])
                    mergeB = np.column_stack([mergeB, np.reshape(estB[:, eig_index], (-1, 1))])
                    mergeC = np.column_stack([mergeC, np.reshape(estC[:, eig_index], (-1, 1))])
                    hash_eigvalue.append(eig_value)
    hash_eigvalue_argsort = np.argsort(hash_eigvalue)
    maxK = 0
    max_index = hash_eigvalue_argsort[-maxK:]
    est_tt_tensor = kruskal_to_tensor([mergeA[:, max_index], mergeB[:, max_index], mergeC[:, max_index]])
    print('compress norm:{}'.format(norm(est_tt_tensor - tensor) / norm(tensor)))
    # print(norm(modeA))
    # print(norm(mergeA))
    # print(norm(modeB))
    # print(norm(mergeB))
    # print(norm(modeC))
    # print(norm(mergeC))
    compress_sigmas = list(set(compress_sigmas))
    true_sigmas = list(set(list(true_sigmas.reshape(-1))))
    hash_eigvalue = list(set(hash_eigvalue))
    compress_sigmas.sort()
    true_sigmas.sort()
    hash_eigvalue.sort()
    # print(np.array(compress_sigmas)[max_index])
    # print(true_sigmas)
    # print(len(true_sigmas))
    # print(np.array(hash_eigvalue)[max_index])
    # print(norm(mergeC[:, 0]))
