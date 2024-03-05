from Extreme_points_iterating import corner_dots_method
from Problem_preprocessing import *

# M = 5 | N_0 = 5
matrix_A = np.array([[6, 2, 4, 3, 15],
                     [12, 10, 19, 8, 17],
                     [2, 11, 14, 1, 14],
                     [4, 7, 7, 12, 10],
                     [15, 23, 17, 32]])  # A[M, N]
vector_b = np.array([19, 21, 21, 18, 8])  # b[M]
sign_x = ["-", ">=", "-", ">=", ">="]  #
sign_m = ["<=", ">=", "=", "=", "="]  #
vector_c = np.array([3, 2, 4, 1, 5])  # c[N]
x = np.array([0, 0, 0, 0, 0])  # x[N]

matrix_A_prep, vector_b_prep, vector_c_prep, len_less, len_more, len_x_pos, N = preproc_make_canon(matrix_A, vector_b,
                                                                                                   sign_x, sign_m,
                                                                                                   vector_c)
matrix_A_canon, vector_b_canon, x_canon, vector_c_canon = make_canon_form(matrix_A_prep, vector_b_prep, vector_c_prep,
                                                                          len_x_pos, len_less, len_more)

# structure for direct problem formalisation
direct_pf = {"param_x_name": x_canon,
             "param_x": np.array([None] * len(x_canon)),  # x[M] (>= 0; N != 5 - CANONISATION !!!)
             "vector_c": vector_c_canon,  # c[N]
             "matrix_A": matrix_A_canon,  # A[M,N]
             "vector_b": vector_b_canon}  # b[M]

print(direct_pf)

corner_solve = corner_dots_method(direct_pf["vector_c"], direct_pf["vector_b"], direct_pf["matrix_A"], 0)
