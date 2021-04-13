import numpy as np
import scipy.linalg as sl
import scipy
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from scipy.linalg import orth
import time


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
        # print(i)
        # print(r)
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


def compress_hcs(tensor, mode_matrices, block_dim):
    S, U, V, W = mode_matrices
    tensor = S * tensor
    L = U.shape[1]
    M = V.shape[1]
    N = W.shape[1]
    B_I = block_dim[0]
    B_J = block_dim[1]
    B_K = block_dim[2]
    G1 = np.reshape(tensor, newshape=[B_I, B_J * B_K])
    G2 = np.reshape(U.T.__matmul__(G1), newshape=[L * B_J, B_K]).T
    G3 = np.reshape(W.T.__matmul__(G2), newshape=[L * N, B_J]).T
    G4 = np.reshape(V.T.__matmul__(G3), newshape=[M * N, L]).T
    return np.reshape(G4, newshape=[L, M, N])


def HCS(T, sketch_dim, return_s_atom=False, return_h_atom=False, S_common=None, so_list=None, ho_list=None):
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
        if S_common:
            hk[:S_common] = ho_list[k][:S_common]
        sk = np.random.normal(size=[nk, 1])
        if S_common:
            sk[:S_common] = so_list[k][:S_common]
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


def generate_HCS(dim, sketch_dim):
    l = len(dim)
    assert len(sketch_dim) == l
    nk_list = dim
    Hk_list = []
    sk_list = []
    for k in range(len(nk_list)):
        nk = nk_list[k]
        mk = sketch_dim[k]
        hk = np.random.randint(low=0, high=mk, size=nk)
        sk = np.random.normal(size=[nk, 1])
        sk = np.sign(sk)
        Hk = np.zeros(shape=[nk, mk])  # (I,L)
        x = np.arange(nk)
        y = hk[x]
        Hk[x, y] = 1
        Hk_list.append(Hk)
        sk_list.append(sk)
    return Hk_list, sk_list


def form_mm_ttr1(U, V, Sigmas, size, index=None, mul_col=True):
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
    Sigmas = np.array(Sigmas).reshape(-1)
    sort_sigmas_index = np.argsort(Sigmas)[::-1]
    sort_sigmas = Sigmas[sort_sigmas_index]
    modeA = modeA[:, sort_sigmas_index]
    modeB = modeB[:, sort_sigmas_index]
    modeC = modeC[:, sort_sigmas_index]
    if mul_col is True:
        for col in range(modeA.shape[1]):
            modeA[:, col] *= sort_sigmas[col]
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
    print("permColmatch before norm:{}".format(norm(iA - dA)))
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


"""
P:2000
I:100
L:10
Block_size:50
error:0.7984724577428853
times:11.422080755233765 secs

P:4000
I:100
L:10
Block_size:50
error:0.5681741426015549
times:23.502922534942627 secs

P:6000
I:100
L:10
Block_size:50
error:0.46474483055522703
times:28.200517892837524 secs

P:8000
I:100
L:10
Block_size:50
error:0.4026188786580816
times:37.252630949020386 secs

P:10000
I:100
L:10
Block_size:50
error:0.35803015442331193
times:43.60045766830444 secs

P:12000
I:100
L:10
Block_size:50
error:0.3283162475210547
times:62.27067494392395 secs

P:14000
I:100
L:10
Block_size:50
error:0.3018316880036839
times:62.09387731552124 secs

P:16000
I:100
L:10
Block_size:50
error:0.2828270856873273
times:68.0651228427887 secs

P:18000
I:100
L:10
Block_size:50
error:0.2673959276538337
times:83.05286407470703 secs

P:20000
I:100
L:10
Block_size:50
error:0.25422445336221294
times:91.0388994216919 secs

P:22000
I:100
L:10
Block_size:50
error:0.24261067507352246
times:98.4898247718811 secs

P:24000
I:100
L:10
Block_size:50
error:0.23122357361565066
times:105.15047907829285 secs

P:26000
I:100
L:10
Block_size:50
error:0.22309691542229307
times:122.03778147697449 secs

P:28000
I:100
L:10
Block_size:50
error:0.21440451912279862
times:118.33251905441284 secs



P:30000
I:100
L:10
Block_size:50
error:0.20759615612224377
times:136.61985659599304 secs

"""

"""
P:2000
I:100
L:5
Block_size:50
error:0.32302554064070876
times:13.596962213516235 secs

P:4000
I:100
L:5
Block_size:50
error:0.2280368003461815
times:28.598660945892334 secs


"""

if __name__ == "__main__":
    # I = J = K = 10
    I = 100
    J = 100
    K = 100
    # R = max(I, J, K) // 20
    R = 1
    creatA = np.random.normal(size=[I, R])
    creatB = np.random.normal(size=[J, R])
    creatC = np.random.normal(size=[K, R])
    creatA = orth(creatA)
    creatB = orth(creatB)
    creatC = orth(creatC)

    L = I // 5
    M = J // 5
    N = K // 5
    # P = 30000
    P = 4000

    block_i = 50
    block_j = 50
    block_j = 50
    block_k = 50

    n_i = I // block_i
    n_j = J // block_j
    n_k = K // block_k

    block_dim = [block_i, block_j, block_k]

    est_A_list = []
    est_B_list = []
    est_C_list = []
    all_t = 0.
    #print('hello')
    for p in range(P):
        Hk_list, sk_list = generate_HCS([I, J, K], [L, M, N])
        sub_hcs_tensor = np.zeros(shape=[L, M, N])
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    mode_a = creatA[i * block_i:(i + 1) * block_i, :]
                    mode_b = creatB[j * block_j:(j + 1) * block_j, :]
                    mode_c = creatC[k * block_k:(k + 1) * block_k, :]
                    sub_tensor = kruskal_to_tensor([mode_a, mode_b, mode_c])
                    start1 = time.time()
                    mode_si = sk_list[0][i * block_i:(i + 1) * block_i]
                    mode_sj = sk_list[1][j * block_j:(j + 1) * block_j]
                    mode_sk = sk_list[2][k * block_k:(k + 1) * block_k]
                    sub_S = kruskal_to_tensor([mode_si, mode_sj, mode_sk])
                    mode_hi = Hk_list[0][i * block_i:(i + 1) * block_i, :]
                    mode_hj = Hk_list[1][j * block_j:(j + 1) * block_j, :]
                    mode_hk = Hk_list[2][k * block_k:(k + 1) * block_k, :]
                    sub_hcs_tensor += compress_hcs(sub_tensor, [sub_S, mode_hi, mode_hj, mode_hk], block_dim)
                    end1 = time.time()
                    all_t+=end1-start1
        start2 = time.time()
        Ut, St, Vt, sigmast = ttr1svd(sub_hcs_tensor, R)  # Ut,St,Vt,sigmast = ttr1svd(Y,R)

        sigmast = np.reshape(sigmast, -1)

        modeAt, modeBt, modeCt = form_mm_ttr1(Ut, Vt, sigmast, [L, M, N], mul_col=True)
        # print('check:{}'.format(norm(sub_hcs_tensor-kruskal_to_tensor([modeAt,modeBt,modeCt]))))
        modeAt = Hk_list[0] @ modeAt
        modeBt = Hk_list[1] @ modeBt
        modeCt = Hk_list[2] @ modeCt
        sa = np.column_stack([sk_list[0] for ttt in range(modeAt.shape[1])])
        sb = np.column_stack([sk_list[1] for ttt in range(modeBt.shape[1])])
        sc = np.column_stack([sk_list[2] for ttt in range(modeCt.shape[1])])
        #sa = sa.T
        #sb = sb.T
        #sc = sc.T
        estAt = sa * modeAt
        estBt = sb * modeBt
        estCt = sc * modeCt
        end2 = time.time()
        all_t += end2-start2
        est_A_list.append(estAt)
        est_B_list.append(estBt)
        est_C_list.append(estCt)
        if (p+1)% 100 ==0:
            print(p+1)
    diff = 0.
    all = 0.
    est_block = np.zeros(shape=[block_i, block_j, block_k])
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                est_block[:, :, :] = 0
                for p in range(P):
                    mode_est_a = est_A_list[p][i * block_i:(i + 1) * block_i, :]
                    mode_est_b = est_B_list[p][j * block_j:(j + 1) * block_j, :]
                    mode_est_c = est_C_list[p][k * block_k:(k + 1) * block_k, :]
                    est_block += (1 / P) * kruskal_to_tensor([mode_est_a, mode_est_b, mode_est_c])
                mode_a = creatA[i * block_i:(i + 1) * block_i, :]
                mode_b = creatB[j * block_j:(j + 1) * block_j, :]
                mode_c = creatC[k * block_k:(k + 1) * block_k, :]
                block = kruskal_to_tensor([mode_a, mode_b, mode_c])
                diff += np.sum((block - est_block) ** 2)
                all += np.sum(block ** 2)
    print('diff norm:{}'.format(np.sqrt(diff) / np.sqrt(all)))
    print('times:', all_t, 'secs')
