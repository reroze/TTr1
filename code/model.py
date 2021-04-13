import numpy as np
import scipy


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))#按第mode维展开


def kr(matrices):
    n_columns = matrices[0].shape[1]#列数
    n_factors = len(matrices)#矩阵的个数

    start = ord('a')#返回一个a字符的ascll码 #chr（）：输入一个数字，返回一个字符
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))#比如n=3 则为'a','b','c'
    source = ','.join(i + common_dim for i in target)#source为'a'+'z',...
    operation = source + '->' + target + common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


def khatri_rao(matrices, skip_matrix=None, reverse=False):
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]#可选的跳过对应的skip_matrix

    n_columns = matrices[0].shape[1]#列数

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
#克鲁斯卡尔积转成tensor Aijk = add(i form 1 to R) air bjr ckr


def HCS(T, sketch_dim):
    l = len(np.shape(T))#T的第一维是l
    assert len(sketch_dim) == l
    print('sketch_dim', sketch_dim)
    nk_list = list(np.shape(T))#[n1,n2,n3,...,nk]
    Hk_list = []
    sk_list = []
    for k in range(len(nk_list)):
        nk = nk_list[k]
        mk = sketch_dim[k]#对应的超参数？
        hk = np.random.randint(low=0, high=mk, size=nk) #取到mk-1的一个随机映射
        sk = np.random.normal(size=[nk, 1])
        sk = np.sign(sk)#随机hash到+-1
        Hk = np.zeros(shape=[nk, mk])
        x = np.arange(nk)#从0到nk-1
        y = hk[x]
        Hk[x, y] = 1#hk[k]->y
        print(len(np.nonzero(Hk)[0]))#多少个非0的
        Hk_list.append(Hk)
        sk_list.append(sk)#对每个维度都进行这样的操作
    S = kruskal_to_tensor(sk_list)
    A = S * T
    for mode in range(l):
        shape = list(np.shape(A))
        A = unfold(A, mode)
        shape[mode] = sketch_dim[mode]#第mode维展开
        A = Hk_list[mode].T @ A
        A = fold(A, mode, shape)
    return A, Hk_list, S


def DeHCS(T, Hk_list, S):
    l = len(np.shape(T))
    for mode in range(l):
        shape = list(np.shape(T))
        shape[mode] = np.shape(Hk_list[mode])[0]
        T = unfold(T, mode)
        T = Hk_list[0] @ T
        T = fold(T, mode, shape)
    T = S * T
    return T


def check():
    """检查矩阵运算与爱因斯坦求和公式的等价性"""
    l = 10
    m = 20
    n = 30
    tensor = np.random.normal(size=[l, m, n])
    i = 5
    j = 7
    k = 8
    U = np.random.normal(size=[i, l])
    V = np.random.normal(size=[j, m])
    W = np.random.normal(size=[k, n])
    print('check mode 0')
    tensor_0 = unfold(tensor, 0)
    tensor_0 = U @ tensor_0
    tensor_0 = fold(tensor_0, 0, [i, m, n])
    einsum_0 = np.einsum('pjk,tp->tjk', tensor, U)
    print(norm(einsum_0 - tensor_0))
    print('check mode 1')
    tensor_1 = unfold(tensor, 1)
    tensor_1 = V @ tensor_1
    tensor_1 = fold(tensor_1, 1, [l, j, n])
    einsum_1 = np.einsum('ijk,tj->itk', tensor, V)
    print(norm(einsum_1 - tensor_1))
    print('check mode 2')
    tensor_2 = unfold(tensor, 2)
    tensor_2 = W @ tensor_2
    tensor_2 = fold(tensor_2, 2, [l, m, k])
    einsum_2 = np.einsum('ijk,tk->ijt', tensor, W)
    print(norm(einsum_2 - tensor_2))


if __name__ == "__main__":
    """I = J = K = 150
    L = M = N = 2
    tensor = np.random.normal(size=[I, J, K]).astype(np.float64)
    sketch_tensor, Hk_list, S = HCS(tensor, [L, M, N])
    recover_tensor = DeHCS(sketch_tensor, Hk_list, S)
    print(norm(recover_tensor-tensor))"""
    check()
