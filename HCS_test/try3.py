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
    return int(ans)


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

    return kr(matrices)


def norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))


def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)


def kruskal_to_tensor(factors):
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)


def ttr1svd(tensor: np.ndarray, R=None):
    n = np.shape(tensor)
    r = np.zeros((len(n) - 1,))
    for i in range(len(r)):
        r[i] = min(n[i], prod(n[i + 1:]))
    r[0] = R
    r[1] = R
    if R == None:
        R = tensor.shape[0] * tensor.shape[1]
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
    # print('-----------')
    # print(Ut.shape)#(20,20)
    # print(St.shape)#(20,)
    # print(Vt.shape)#(20,400)
    # print('-----------')
    Ut = Ut[:, :R]  # 原来是
    St = St[:R]
    Vt = Vt[:R, :]  # 取前R个
    Vt = Vt.T
    # print('-----------')
    # print(Ut.shape)#(20, 5)
    # print(St.shape)#(5,)
    # print(Vt.shape)#(400, 5)
    # print('-----------')
    U.append(Ut)
    S.append(St.reshape((-1, 1)))  # 变成列向量
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
            Ut = Ut[:, :R]
            St = St[:R]
            Vt = Vt[:R, :]
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
        Hk = np.zeros(shape=[nk, mk])  # (I,L)
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


def form_mm_ttr1(U, V, Sigmas, size, index=None):
    I, J, K = size
    modeA = np.zeros(shape=[I, 1])
    modeB = np.zeros(shape=[J, 1])
    modeC = np.zeros(shape=[K, 1])
    for i in range(R):
        for j in range(R):
            tmp = [np.reshape(U[0][:, i], [-1, 1]), np.reshape(U[i + 1][:, j], [-1, 1]),
                   np.reshape(V[i + 1][:, j], [-1, 1])]
            modeA = np.column_stack([modeA, tmp[0]])
            modeB = np.column_stack([modeB, tmp[1]])
            modeC = np.column_stack([modeC, tmp[2]])
    modeA = modeA[:, 1:]
    modeB = modeB[:, 1:]
    modeC = modeC[:, 1:]
    if index is not None:
        modeA = modeA[:, index]
        modeB = modeB[:, index]
        modeC = modeC[:, index]
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


def drawRandomMatrices(A, L, iter=5, tol=1e-10):
    n = A.shape[1]
    Q = np.random.normal(size=[n, L])
    loss = norm(A - Q @ Q.T)
    print('iter:{} loss:{}'.format(-1, loss))
    for i in range(iter):
        Q, _ = scipy.linalg.qr(A @ Q, mode='economic')
        Q, _ = scipy.linalg.qr(A.T @ Q, mode='economic')
        loss = norm(A - Q @ Q.T)
        if loss < tol:
            break
        # print('iter:{} loss:{}'.format(i, loss))
    # Q, _ = scipy.linalg.qr(A @ Q, mode='economic')
    return Q


def juge_zhengjiao(Q, estA, estB, estC):
    '''
    :param mergeA:新的A波浪线
    :param Q: 原来的
    :return: 筛选添加之后的矩阵
    '''
    # result = []
    A = estA.T
    B = estB.T
    C = estC.T
    # print('A.T.shape', A.shape)#(25, 100)
    # print('Q.shape', Q.shape)#(1, 100)
    sui = 0
    alli = 0
    for i in range(A.shape[0]):
        type = 0
        # rint('A[0]:', A[0])
        # print('Q', Q)
        # print('Q[0]', Q[0])
        # print('Q[1]', Q[1])
        # print('Q[2]', Q[2])
        if (((A[i] @ Q[0]) ** 2).sum() < 1e-5 or ((B[i] @ Q[1]) ** 2).sum() < 1e-5 or (
                (C[i] @ Q[2]) ** 2).sum() < 1e-5):
            # result.append(mergeA)
            # print('A[i]', A[i])
            # print('B[i]', B[i])
            # print('C[i]', C[i])
            if ((A[i] @ Q[0]) ** 2).sum() < 1e-5:
                type = 1
            elif ((B[i] @ Q[0]) ** 2).sum() < 1e-5:
                type = 2
            elif ((C[i] @ Q[0]) ** 2).sum() < 1e-5:
                type = 3
            alli += 1
            if ((type == 1 and ((A[i] ** 2).sum()) ** 0.5 > 1e-3) or (
                    type == 2 and ((B[i] ** 2).sum()) ** 0.5 > 1e-3) or (
                    type == 3 and ((C[i] ** 2).sum()) ** 0.5 > 1e-3)):
                sui += 1
                Q[0] = np.column_stack([Q[0], A[i].T])
                Q[1] = np.column_stack([Q[1], B[i].T])
                Q[2] = np.column_stack([Q[2], C[i].T])
    # print(Q.shape)
    #print('alli:', alli)
    #print('sui:', sui)
    return Q


def DeHCS(HCS_Y, Hk_list, sk_list, i):
    H = Hk_list[i]
    # print('H.shape', H.shape)
    # print('HCS_Y.shape', HCS_Y.shape)
    HCS_Y = H @ HCS_Y
    # HCS_Y = HCS_Y * sk_list[i]
    for j in range(sk_list[i].shape[0]):
        for k in range(HCS_Y.shape[1]):
            HCS_Y[j][k] *= sk_list[i][j]
    return HCS_Y








if __name__ == "__main__":
    # I = J = K = 10
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
    U, S, V, sigmas = ttr1svd(tensor, R=R)

    true_sigmas = sigmas
    nleaf = I * min(J, K)
    leaf = []
    modeA = np.zeros(shape=[I, 1])
    modeB = np.zeros(shape=[J, 1])
    modeC = np.zeros(shape=[K, 1])
    for i in range(R):
        for j in range(R):
            tmp = [np.reshape(U[0][:, i], [-1, 1]), np.reshape(U[i + 1][:, j], [-1, 1]),
                   np.reshape(V[i + 1][:, j], [-1, 1])]
            modeA = np.column_stack([modeA, tmp[0]])
            modeB = np.column_stack([modeB, tmp[1]])
            modeC = np.column_stack([modeC, tmp[2]])
            # modeA.append(U[0][:, i])
            # modeB.append(U[i + 1][:, j])
            # modeC.append(V[i + 1][:, j])
            leaf.append(tmp)
    modeA = modeA[:, 1:]
    modeB = modeB[:, 1:]
    modeC = modeC[:, 1:]
    print(modeA.shape)
    print(modeB.shape)
    print(modeC.shape)
    for col in range(modeA.shape[1]):
        modeA[:, col] *= sigmas[col]
    estTensor = kruskal_to_tensor([modeA, modeB, modeC])
    est = 0
    for i, node in enumerate(leaf):
        est += sigmas[i] * kruskal_to_tensor(node)
    print('vec est norm:{}'.format(norm(est - tensor) / norm(tensor)))
    print('mat est norm:{}'.format(norm(estTensor - tensor) / norm(tensor)))
    # exit()
    L = I // 5
    M = J // 5
    N = K // 5
    #P = 30000
    P = 2000

    #compress loss1
    # 10000:0.144
    # 20000:0.102155
    # 30000:0.08361

    #compress loss2
    #暴力均值 P =1000 : 1.000001622
    #P = 2000 : 1.000001496839


    #compress loss3
    #正交筛选（初始（简单）版本） P=2000  :

    mergeA = 0
    mergeB = 0
    mergeC = 0
    compress_sigmas = []
    hash_eigvalue = []
    max_iters_random = 5
    svd_random_tol = 1e-10
    choose_k = int(L * min(M, N) // 1)
    # QA = np.zeros([I, 1])
    # QB = np.zeros([J, 1])
    # QC = np.zeros([K, 1])
    # Q = np.zeros([[I,1], [J,1], [K,1]])
    #Q = [np.zeros([I, 1]), np.zeros([J, 1]), np.zeros([K, 1])]

    all_a = np.zeros([100, 25])
    all_b = np.zeros([100, 25])
    all_c = np.zeros([100, 25])

    sketch_dim = [20, 20, 20]
    Y1 = np.zeros([20, 20, 20])
    Q = [np.zeros([100, 1]), np.zeros([100, 1]), np.zeros([100, 1])]
    meanT = 0
    for p in range(P):
        # U = np.random.normal(size=[I, L])
        # V = np.random.normal(size=[J, M])
        # W = np.random.normal(size=[K, N])
        # print(U.shape)
        # U = drawRandomMatrices(np.eye(I), L, iter=max_iters_random, tol=svd_random_tol)
        # V = drawRandomMatrices(np.eye(J), M, iter=max_iters_random, tol=svd_random_tol)
        # W = drawRandomMatrices(np.eye(K), N, iter=max_iters_random, tol=svd_random_tol)
        # print(U.shape)
        # Y = HCS(tensor, sketch_dim, 1)#(20, 20, 20)
        Y, HK_list, S, sk_list = HCS(tensor, sketch_dim, 1)#(20, 20, 20)
        # Y = compress(tensor, [U, V, W])
        # Y = HCS(Y)
        Ut, St, Vt, sigmast = ttr1svd(Y, R)#(20, 25)
        sigmast = np.reshape(sigmast, -1)
        # sigmast_index = np.argsort(sigmast)[::-1][:choose_k]
        # sigmast = sigmas[np.where(sigmast < 1e-4)]
        # sigmast = list(sigmast[sigmast_index])
        compress_sigmas.extend(list(np.reshape(sigmast, -1)))
        # print(sigmast.reshape(-1))
        # print(true_sigmas.reshape(-1))
        modeAt, modeBt, modeCt = form_mm_ttr1(Ut, Vt, sigmast, [L, M, N])  ##TTr1分解
        modeAt = DeHCS(modeAt, HK_list, sk_list, 0)
        modeBt = DeHCS(modeBt, HK_list, sk_list, 1)
        modeCt = DeHCS(modeCt, HK_list, sk_list, 2)
        juge_zhengjiao(Q, modeAt, modeBt, modeCt)
        #print('modeAt.shape', modeAt.shape)
        all_a += modeAt
        all_b += modeBt
        all_c += modeCt
        est_compY = kruskal_to_tensor([modeAt, modeBt, modeCt])
        print('compress Y reconstruct norm:{}'.format(norm(est_compY - tensor) / norm(tensor)))
        meanT +=est_compY
    meanT/=P
    print('compress Y reconstruct norm:{}'.format(norm(meanT - tensor) / norm(tensor)))
    all_a /= P
    all_b /= P
    all_c /= P
    test_tensor = kruskal_to_tensor([all_a, all_b, all_c])
    for i in range(3):
        Q[i] = Q[i][:, 1:]
    print('compress2 Y reconstruct norm:{}'.format(norm(test_tensor - tensor) / norm(tensor)))
    test_tensor2 = kruskal_to_tensor([Q[0], Q[1], Q[2]])
    print('compress3 Y reconstruct norm:{}'.format(norm(test_tensor2 - tensor) / norm(tensor)))

        # print('modeAt.shape', modeAt.shape)#(20, 25)
        # print('modeBt.shape', modeBt.shape)#(20, 25)
        # print('est_compy.shape', est_compY.shape)#(20, 20, 20)
        # print('U.shape', U.shape)#(100, 20)
        # estA = U @ modeAt#A波浪线
        # print('estA.shape', estA.shape)
        # estB = V @ modeBt
        # estC = W @ modeCt
        # QA=juge_zhengjiao(estA, QA)#estA(100, 25)
        # QB=juge_zhengjiao(estB, QB)
        # QC=juge_zhengjiao(estC, QC)
        # Q = juge_zhengjiao(Q, estA, estB, estC)
        # print('QA.shape', Q[0].shape)

    # print('QA_shape', Q[0].shape)
    # Q[0] = Q[0][:, 1:]
    # Q[1] = Q[1][:, 1:]
    # Q[2] = Q[2][:, 1:]
    # print('QB_shape', QB.shape)
    # print('QC_shape', QC.shape)
    """hash_eigvalue_argsort = np.argsort(hash_eigvalue)
    maxK = 0"""
    # max_index = hash_eigvalue_argsort[-maxK:]
    # QA = QA[1:]
    # QB = QB[1:]
    # QC = QC[1:]
    # est_tt_tensor = kruskal_to_tensor([mergeA, mergeB, mergeC])
    """est_tt_tensor = kruskal_to_tensor([Q[0], Q[1], Q[2]])
    print(Q[0].shape)
    print(Q[1].shape)
    print(Q[2].shape)
    print(est_tt_tensor.shape)
    print('hi')
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
"""
