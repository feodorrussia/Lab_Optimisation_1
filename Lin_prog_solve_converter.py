import numpy as np
from tabulate import tabulate

from SLE_solver import make_identity, gaussPivotFunc


def get_equality_ind(arr, val, f_eq=False):
    if f_eq:
        return list(filter(lambda x: x >= 0, [i if arr[i] == val else -1 for i in range(len(arr))]))
    else:
        return list(filter(lambda x: x >= 0, [i if arr[i] != val else -1 for i in range(len(arr))]))


def scalmul(vector_a, vector_b):
    return sum([vector_a[i] * vector_b[i] for i in range(len(vector_a))])


def check_system_equation(vector_x, matrix_A, vector_b, sign_m, vector_c):
    print("X_opt:", vector_x)
    print("Function value:", scalmul(vector_x, vector_c))
    print("Limitations:")

    info = []
    for i in range(len(matrix_A)):
        info.append([i, scalmul(matrix_A[i], vector_x), sign_m[i], vector_b[i]])

    print(tabulate(info, headers=['\\', "A*x", "|", "b"]))


def sle_preproc(matrix, x):
    new_x = x.copy()
    new_m = matrix.copy()
    c_del = 0
    for i in range(new_m.shape[1]):
        if abs(sum(matrix[:, i])) == 0:
            new_x = np.delete(new_x, i - c_del)
            new_m = np.delete(new_m, i - c_del, 1)
            c_del += 1
    if new_m.shape[1] <= 1:
        print("\nSystem have inf solve")
        return None, None
    else:
        # print(f"\nSystem have {new_m.shape[1]} param")
        return new_m, new_x


def convert_solve(pf):
    """
    Function for converting dual problem solve to direct problem solve
    :param pf: structure for dual problem CANON formalisation
        {"param": np.array([0, 0]), y[M] >= 0
            "vector_c": np.array(), c[N] (for dual)
            "matrix_A": np.array(), A[M,N] (= everywhere)
            "vector_b": np.array(), b[M] (for dual)
            "sign_m": list("<="|"="|">="), sign in limit system (for dual)}
    :return: np.array() - direct problem solve (x[N])
    """

    N = pf["matrix_A"].shape[1]
    M = pf["matrix_A"].shape[0]
    matrix_A = pf["matrix_A"]
    direct_solve = np.array([None] * M)  # x[M] (> 0; N != 5 - CANONISATION !!!)

    print("\nInput:")
    print(f"Function value: {scalmul(pf['param'], pf['vector_c'])}")
    print(f"Y_opt: {pf['param']}", pf['vector_c'])

    for i in range(M):
        if scalmul(matrix_A[i], pf["param"]) > pf["vector_b"][i]:
            print(scalmul(matrix_A[i], pf["param"]), pf["vector_b"][i])
            direct_solve[i] = 0

    if len(get_equality_ind(direct_solve, 0)) == 0:
        print(f"\nNULL SOLVE {direct_solve}")
        return direct_solve

    indexes = get_equality_ind(pf["param"], 0)
    sle_matrix_A = pf["matrix_A"].T[indexes]
    sle_vector_b = pf["vector_c"][indexes]

    ind_x = np.arange(sle_matrix_A.shape[1])
    sle_matrix_A, ind_x = sle_preproc(sle_matrix_A, ind_x)
    # print(indexes, "\n")
    if sle_matrix_A is not None and sle_vector_b is not None:
        # SLE solve realisation

        # SLE_meth_1(sle_matrix_A, sle_vector_b) -> sle_result
        # SLE_meth_2(sle_matrix_A, sle_vector_b) -> sle_result
        sle_result = make_identity(gaussPivotFunc(np.column_stack([sle_matrix_A, sle_vector_b])))[:, 3]

        direct_solve[ind_x] = sle_result

    free_x = get_equality_ind(direct_solve, None, f_eq=True)
    if len(free_x) > 1:
        print("\n!Warning!")
        print(direct_solve)

    elif len(free_x) == 1:
        print("\nFunction check. Pre-solution:", direct_solve, "Vector c:", pf["vector_b"].tolist())
        i_x = free_x[0]
        direct_solve[i_x] = 0.0
        last_x = (scalmul(pf["param"], pf["vector_c"]) -
                             scalmul(direct_solve, pf["vector_b"])) / pf["vector_c"][i_x]
        direct_solve[i_x] = last_x

    print("\nResult:")
    print(f"Function value: {scalmul(direct_solve, pf['vector_b'])}")
    print(f"X_opt: {direct_solve}")
    return scalmul(direct_solve, pf['vector_b']), direct_solve.astype(float)
