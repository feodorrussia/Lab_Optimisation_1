from new_simplex import simplex
from new_funcs import *
from Extreme_points_iterating import corner_dots_method


system, sign, goal_func, idx = read_file("task.txt")

print('---ИСХОДНАЯ ЗАДАЧА---')
print_system(system, sign, goal_func, idx)

print('---ДВОЙСТВЕННАЯ ЗАДАЧА---')
system1, sign1, goal_func1, idx1 = direct_to_dual(system, sign, goal_func, idx)
print_system(system1, sign1, goal_func1, idx1)

print('---КАНОНИЧЕСКАЯ ФОРМА ДВОЙСТВЕННОЙ ЗАДАЧИ---')
system_1, sign_1, goal_func_1, idx_1 = to_canonical(system1, sign1, goal_func1, idx1)
print_system(system_1, sign_1, goal_func_1, idx_1)

# print('---РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ МЕТОДОМ ПЕРЕБОРА ОПОРНЫХ ВЕКТОРОВ---')
# A, b = getAb(system_1)
# solution = corner_dots_method(np.array(goal_func_1), np.array(b), np.array(A), 0)
# print('Вектор решения: ', solution)

print("\n\n")
print('---КАНОНИЧЕСКАЯ ФОРМА---')
system, sign, goal_func, idx = to_canonical(system, sign, goal_func, idx)
print_system(system, sign, goal_func, idx)

# print('---РЕШЕНИЕ ИСХОДНОЙ ЗАДАЧИ МЕТОДОМ ПЕРЕБОРА ОПОРНЫХ ВЕКТОРОВ---')
# A, b = getAb(system)
# solution = corner_dots_method(np.array(goal_func), np.array(b), np.array(A), 0)
# print('Вектор решения: ', solution)

print('\n\n---РЕШЕНИЕ ИСХОДНОЙ ЗАДАЧИ СИМПЛЕКС-МЕТОДОМ---')
A, b = getAb(system)
solution = simplex(goal_func, A, b)
print('Вектор решения: ', solution)
