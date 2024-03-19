from Dual_problem_processing import convert_DirectToDual
from Extreme_points_iterating import corner_dots_method
from Lin_prog_solve_converter import convert_solve, scalmul
from Problem_preprocessing import *
from Simplex_method import simplex

# M = 5 | N_0 = 5
matrix_A = np.array([[1, 5, 1, 3, 4],
                     [2, 3, 4, 1, 2],
                     [2, 1, 7, 2, 6],
                     [3, 3, 2, 3, 1],
                     [1, 8, 1, 1, 3]])  # A[M, N]
vector_b = np.array([2, 8, 3, 6, 2])  # b[M]
sign_x = [">=", ">=", ">=", "-", "-"]  #
sign_m = [">=", "<=", "=", "=", "="]  #
vector_c = np.array([1, -4, 2, -3, -1])  # c[N]
x = np.array([0, 0, 0, 0, 0])  # x[N]

direct_pf = canonisation(matrix_A, vector_b,
                         sign_x, sign_m,
                         vector_c, is_max=False)

print("Direct canon problem:")
print_system(direct_pf["matrix_A"].tolist(), direct_pf["vector_b"].tolist(), direct_pf["sign_m"],
             direct_pf["vector_c"].tolist(), [">="] * len(direct_pf["vector_c"]))

dir_simplex_solve = simplex(direct_pf["vector_c"].tolist(), direct_pf["matrix_A"].tolist(),
                        direct_pf["vector_b"].tolist())

dir_f_value, dir_points = corner_dots_method(direct_pf["vector_c"].copy(), direct_pf["vector_b"].copy(),
                                             direct_pf["matrix_A"].copy())

print('\nSimplex method func result: ', scalmul(dir_simplex_solve, direct_pf["vector_c"].tolist()))
print('\nSolve (canon basis):', dir_simplex_solve)
print()

print('\nExtreme point func result: ', dir_f_value)
print('\nPoints (canon basis):\n' + "\n".join([f"({', '.join(list(map(str, point)))})" for point in dir_points]))
print()

dual_A, dual_b, dual_sign_x, dual_sign_m, dual_c = convert_DirectToDual(vector_c, matrix_A, vector_b, sign_x, sign_m)
dual_pf = canonisation(dual_A, dual_b,
                       dual_sign_x, dual_sign_m,
                       dual_c)

print("Dual problem:")
print_system(dual_A.tolist(), dual_b.tolist(), dual_sign_m, dual_c.tolist(), dual_sign_x)
# print("Dual canon problem:")
# print_system(dual_pf["matrix_A"].tolist(), dual_pf["vector_b"].tolist(), dual_pf["sign_m"],
#              dual_pf["vector_c"].tolist(), [">="] * len(dual_pf["vector_c"]))
dual_f_value, dual_points = corner_dots_method(dual_pf["vector_c"].copy(), dual_pf["vector_b"].copy(),
                                               dual_pf["matrix_A"].copy())

print('\nExtreme point func result: ', dual_f_value)
print('\nPoints (canon basis):\n' + "\n".join([f"({', '.join(list(map(str, point)))})" for point in dual_points]))
print()

dual_pf["param"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0])
# np.array(list(map(float, input(f"Введите точку - решение для двойственной задачи({dual_pf['vector_c'].shape[0]} чисел, разделитель - ', '):\n").split(', '))))
# print(f"Get point: {dual_pf['param']}")
# 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0
# 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0

# print_system(direct_pf["matrix_A"].tolist(), direct_pf["vector_b"].tolist(), ["="] * len(direct_pf["vector_b"]),
#              direct_pf["vector_c"].tolist(), [0, 1, 2, 3])
# print(direct_pf["param"])

# print("\nDir solve:", convert_solve(dual_pf))
