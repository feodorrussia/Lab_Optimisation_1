from Problem_preprocessing import *


def dual_revert_sign(sign, is_vector_x=False):
    if is_vector_x:
        if sign == "=":
            return "-"
        return sign
    else:
        if sign == "-":
            return "="
        elif sign == ">=":
            return "<="
        return ">="


def convert_DirectToDual(c, m, b, signs_x, signs_m, is_max: bool = True):
    """
    Function for converting direct initial problem to dual canon
    :param c: np.array(), target function coefficients
    :param m: np.array(), limitation matrix
    :param b: np.array(), vector of limitations
    :param signs_x: list(str), signs of params (>= | - | <=)
    :param signs_m: list(str), signs in limitation system (>= | = | <=)
    :param is_max: bool = True, maximising or minimising target function
    :return: structure for dual problem CANON formalisation:
        {param_y_name: list(str),
        "param_y": np.array(),
        "vector_c": np.array(), c[N]
        "matrix_A": np.array(), A[M,N]
        "vector_b": np.array(), b[M]}
    """

    matrix_A = m.transpose()
    vector_b = c.copy()
    vector_c = b.copy()

    dual_signs_m = list(map(lambda x: dual_revert_sign(x, is_vector_x=False), signs_x))
    dual_signs_x = list(map(lambda x: dual_revert_sign(x, is_vector_x=True), signs_m))

    # matrix_A_prep, vector_b_prep, vector_c_prep, len_less, len_more, len_y_pos, N = preproc_make_canon(matrix_A,
    #                                                                                                    vector_b,
    #                                                                                                    dual_indices_x,
    #                                                                                                    dual_indices_m,
    #                                                                                                    vector_c)
    #
    # matrix_A_canon, vector_b_canon, y_canon, vector_c_canon = make_canon_form(matrix_A_prep, vector_b_prep,
    #                                                                           vector_c_prep,
    #                                                                           len_y_pos, len_less, len_more,
    #                                                                           is_max=is_max, is_dual=True)

    return matrix_A, vector_b, dual_signs_x, dual_signs_m, vector_c
