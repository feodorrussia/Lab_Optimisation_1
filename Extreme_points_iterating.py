from itertools import combinations
import numpy as np
from Problem_preprocessing import mini_gauss

np.seterr(invalid='ignore')


def get_aux(Matrix):
    """
    На основе входной матрицы Matrix генеририруются подматрицы размером m * m и возвращаются через yield
    вместе с индексами соответствующих столбцов матрицы Matrix
    """
    columns_configuration = combinations(
        range(Matrix.shape[1]), Matrix.shape[0]
    )
    for item in columns_configuration:
        A_aux = np.zeros(Matrix.shape[0] * Matrix.shape[0]).reshape(
            Matrix.shape[0], Matrix.shape[0]
        )
        for j in range(len(item)):
            assrt = Matrix[:, item[j]]
            A_aux[:, j] = assrt
        yield A_aux, item


def corner_dots_method(c_vector, b_vector, Matrix, c0=0):
    """
    Метод крайних точек
    Каждая из подматриц решается методом Жордано-Гаусса
    Если решение существует, проверяется его допустимость
    Если решение допустимо, оно добавляется в массив
    Из сформированного массива выбирается решение - наименьший элемент
    Если массив пуст, то допустимого решения не существует
    """
    generator = get_aux(Matrix)
    solutions = {}
    for A_aux, item in generator:
        try:
            tmp_basis = np.arange(A_aux.shape[1])
            A_aux, b_vector_1 = mini_gauss(
                A_aux,
                A_aux.shape[1],
                A_aux.shape[0],
                b_vector.copy(),
                0,
                tmp_basis,
            )

            # проверка
            if all([val >= 0 for val in b_vector_1]):
                result = 0
                point = [0] * c_vector.shape[0]
                for j in range(len(item)):
                    result += c_vector[item[tmp_basis[j]]] * b_vector_1[j]
                    point[item[tmp_basis[j]]] = b_vector_1[j]
                result -= c0

                # print(f'Point: {point}\nFunc_result: {result}')
                if result not in solutions.keys():
                    solutions[result] = [point]
                elif point not in solutions[result]:
                    solutions[result].append(point)
        except Exception:
            continue
            pass
    if len(solutions.keys()) != 0:
        # print(solutions)
        # print('\nExtreme point func result: ', min(solutions.keys()))
        # print('\nPoints: ', solutions[min_result])
        return min(solutions.keys()), solutions[min(solutions.keys())]
    else:
        print('\nExtreme point-log: Problem have no solution\n')
