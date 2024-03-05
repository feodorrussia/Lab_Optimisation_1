import numpy as np
from SLE_solver import make_identity, gaussPivotFunc


def get_eq_ind(arr, val):
    return list(filter(lambda x: x > 0, [i if arr[i] != val else -1 for i in range(len(arr))]))


def get_direct_solve(dual_pf):
    """
    Function for converting dual problem solve to direct problem solve
    :param dual_pf: structure for dual problem CANON formalisation
        {"param_y": np.array([0, 0]), y[M] >= 0
            "vector_c": np.array(), c[N]
            "matrix_A": np.array(), A[M,N] (= everywhere)
            "vector_b": np.array(), b[M]}
    :return: np.array() - direct problem solve (x[N])
    """
    # structure for direct problem formalisation
    direct_solve = np.array([None, None, None, None, None])  # x[M] (> 0; N != 5 - CANONISATION !!!)

    M = len(dual_pf["param_y"])
    N = len(dual_pf["vector_c"])
    matrix_A = dual_pf["matrix_A"].transpose()

    for i in range(N):
        if np.matmul(dual_pf["param_y"], matrix_A[i]) > dual_pf["vector_c"][i]:
            direct_solve[i] = 0

    indexes = get_eq_ind(dual_pf["param_y"], 0)
    sle_matrix_A = dual_pf["matrix_A"][indexes]
    sle_vector_b = dual_pf["vector_b"][indexes]

    # SLE solve realisation

    # SLE_meth_1(sle_matrix_A, sle_vector_b) -> sle_result
    # SLE_meth_2(sle_matrix_A, sle_vector_b) -> sle_result
    sle_result = make_identity(gaussPivotFunc(np.column_stack([sle_matrix_A, sle_vector_b])))[:, 3]
    print(sle_result)

    direct_solve[indexes] = sle_result

    free_x = get_eq_ind(direct_solve, None)
    if len(free_x) > 1:
        print("!Warning!")

    elif len(free_x) == 1:
        i_x = free_x[0]
        direct_solve[i_x] = 0
        direct_solve[i_x] = (dual_pf["param_y"] * dual_pf["vector_b"] -
                             direct_solve * dual_pf["vector_c"]) / dual_pf["vector_c"][i_x]

    print("Result:")
    print(f"Function value: {direct_solve * dual_pf['vector_c']}")
    print(f"X_opt: {direct_solve}")
    return direct_solve
