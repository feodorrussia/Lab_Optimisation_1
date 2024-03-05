import numpy as np


def find_indices(arr):
    indices_dict = {'<=': [], '>=': [], '=': [], '-': []}

    for i, elem in enumerate(arr):
        if elem == '<=':
            indices_dict['<='].append(i)
        elif elem == '>=':
            indices_dict['>='].append(i)
        elif elem == '=':
            indices_dict['='].append(i)
        elif elem == '-':
            indices_dict['-'].append(i)
    return indices_dict


def find_not_zero_and_swap(M, first_row_to_start, k, n, Basis_Indexes):
    """
    if we have the case, when we need to divide on 0 in Gauss method, we need to change columns to find not 0 coef.
    """
    flag = False
    if M[k + first_row_to_start, Basis_Indexes[k]] == 0:
        position_in_basis_indexes = 0
        for w in Basis_Indexes:
            if M[k + first_row_to_start, w] != 0:
                Basis_Indexes[k], Basis_Indexes[position_in_basis_indexes] = (Basis_Indexes[position_in_basis_indexes],
                                                                              Basis_Indexes[k])
                return True
            position_in_basis_indexes += 1
        if not flag:
            return False
    return True


def gauss_one_step(M, b_vector, n, i, t, p):
    """
    Прибавляет к i-й строке t-ю строку с нужным коэффициентом из столбца p (метод Гаусса)
    """
    coef = M[i, p] / M[t, p]
    for j in range(0, n, 1):
        M[i, j] = M[i, j] + (-coef) * M[t, j]
        M[i, j] = '{:.8f}'.format(M[i, j])
    b_vector[i] = b_vector[i] + (-coef) * b_vector[t]
    # print(pd.DataFrame(M))


def mini_gauss(M, n, m, b_vector, first_row_to_start, Basis_Indexes):
    """
    Вспомогательная функция.
    Идет форматирование на 1 знак после запятой.
    Функция приводит подматрицу к единичной матрице по методу Гаусса
    :param Basis_Indexes:
    :param M: Матрица на обработку
    :param n: число столбцов M
    :param m: число строк M
    :param b_vector: вектор-столбец свободных членов (размерность m)
    :param first_row_to_start: номер строки, начиная с которого начинается приведение к диагональному виду (первое исходное уравнение)
    :return: Матрица, у которой в левом нижнем углу стоит единичная матрица
    """
    # print(Basis_Indexes)
    """forward"""
    for k in range(0, m - first_row_to_start, 1):  # 0, 1
        flag = find_not_zero_and_swap(M, first_row_to_start, k, n, Basis_Indexes)
        if flag:
            for i in range(first_row_to_start + k + 1, m, 1):  # 4, 5 -- 5
                gauss_one_step(
                    M, b_vector, n, i, k + first_row_to_start, Basis_Indexes[k]
                )
                # print(f"5:\n{M}")
        else:
            return None
    # print(f"6:\n{M}")
    """back"""
    for k in range(m - first_row_to_start - 1, 0, -1):  # 2, 1
        flag = find_not_zero_and_swap(
            M, first_row_to_start, k, n, Basis_Indexes
        )
        if flag:
            for i in range(
                    first_row_to_start + k - 1, first_row_to_start - 1, -1
            ):  # 4, 3  --  3
                gauss_one_step(
                    M, b_vector, n, i, k + first_row_to_start, Basis_Indexes[k]
                )
        else:
            return None

    """other coefficients"""
    for k in range(0, m - first_row_to_start, 1):  # 0, 1, 2
        for i in range(0, first_row_to_start, 1):  # 0, 1, 2
            gauss_one_step(
                M, b_vector, n, i, k + first_row_to_start, Basis_Indexes[k]
            )

    """Normalize"""
    for i in range(first_row_to_start, m, 1):
        coef = M[i, Basis_Indexes[i - first_row_to_start]]
        for j in range(0, n, 1):
            M[i, j] = M[i, j] / coef
        b_vector[i] = b_vector[i] / coef

    return M, b_vector


def preprocessing_(Input_Matrix, b, Equation_Less: int, Equation_More: int):
    Total_Restrictions = Input_Matrix.shape[1]
    M_General = Input_Matrix.shape[0]
    A = Input_Matrix.copy()
    Basis_Indexes = np.arange(Total_Restrictions)

    out = mini_gauss(
        M=A,
        n=Total_Restrictions,
        m=M_General,
        b_vector=b,
        first_row_to_start=Equation_Less + Equation_More,
        Basis_Indexes=Basis_Indexes,
    )

    if out is not None:
        Ind_Final = list(
            Basis_Indexes[
            Total_Restrictions
            - (Equation_More + Equation_Less): Total_Restrictions
            ]
        ) + list(Basis_Indexes[0: M_General - (Equation_More + Equation_Less)])

        """Normalize"""
        for i in range(0, M_General, 1):
            coef = A[i, Ind_Final[i]]
            A[i, :] = A[i, :] / coef
            b[i] = b[i] / coef

        return A, b, Ind_Final

    else:
        return None


def preproc_make_canon(m, b, signs_x, signs_m, c):
    """Осуществляет переставления строк столбцов матрицы исходя из стратегии представления
    для приведения к канонической форме
    return: x_indexes"""

    indices_x = find_indices(signs_x)
    indices_M = find_indices(signs_m)

    tmp = []
    c_ = []
    b_ = []

    for index in indices_M['<=']:
        tmp.append(m[index, :])
        b_.append(b[index])

    for index in indices_M['>=']:
        tmp.append(m[index, :])
        b_.append(b[index])

    for index in indices_M['=']:
        tmp.append(m[index, :])
        b_.append(b[index])

    for index in indices_x['>=']:
        c_.append(c[index])

    for index in indices_x['<=']:
        tmp[:, index] = [-tmp_ for tmp_ in tmp[:, index]]
        c_.append(c[index])

    for index in indices_x['-']:
        c_.append(c[index])

    return (
        np.array(tmp),
        np.array(b_),
        np.array(c_),
        len(indices_M['<=']),
        len(indices_M['>=']),
        len(indices_x['>=']) + len(indices_x['<=']),
        indices_x['<='] + indices_x['>='] + indices_x['-'],
    )


def make_canon_form(
        input_matrix,
        b,
        c,
        x_positive: int,
        equation_less: int,
        equation_more: int,
        is_max: bool = False,
        is_dual=False,
        perem: [int] = None,
):
    """
    1) Making canon form from input data
    2) Selecting basis: columns, that were added due to inequalities are firstly obtained in basis
       Then we are looking for other basis columns from left to right, trying to make them view like (1, 0, ... 0)

    :param is_dual:
    :param c:
    :param input_matrix: входящая матрица (????equalities);
    :param b: входящий вектор правой части;
    :param is_max: максимизация(True) / минимизация(False);
    :param x_positive: кол-во переменных с ограничением на знак
    :param equation_less: кол-во ограничений с знаком <= / <
    :param equation_more: кол-во ограничений с знаком > / >=
    :param perem:
    :return
    """
    # N, M входящей матрицы
    N_General = input_matrix.shape[1]
    M_General = input_matrix.shape[0]

    param_letter = "y" if is_dual else "x"
    add_param_letter = "q" if is_dual else "z"

    # Определяем сколько нужно добавить переменных, для которых нет ограничения на знак.
    X_Any = N_General - x_positive

    # Определяем, общее количество переменных.
    Total_Restrictions = N_General + X_Any + equation_less + equation_more

    # Создаем новую матрицу из 0.
    A = np.zeros(M_General * Total_Restrictions).reshape((M_General, Total_Restrictions))

    if perem is None:
        perem = [0] * N_General
    perem_ = [f'{param_letter}{item}' for item in perem + np.zeros(X_Any + equation_less + equation_more).tolist()]
    c_ = np.concatenate([c, np.zeros(X_Any + equation_less + equation_more)])
    """
    Вводим в итоговую матрицу столбцы связанные с переменными,
    которые имеют ограничения на знак.
    """

    for i in range(0, x_positive, 1):
        for j in range(0, M_General, 1):
            A[j, i] = input_matrix[j, i]

    # print(f'1) Вкладываем в матрицу столбцы с переменными с ограничением на знак:\n{A}')

    """
    Вкладываем в матрицу столбцы, связанные с переменными без ограничений.
    """

    for i in range(x_positive, N_General, 1):
        for j in range(0, M_General, 1):
            A[j, 2 * i - x_positive] = input_matrix[j, i]
            perem_[2 * i - x_positive] = f's{2 * i - x_positive}'
            A[j, 2 * i - (x_positive - 1)] = -input_matrix[j, i]
            perem_[2 * i - x_positive + 1] = f's{2 * i - x_positive + 1}'
    # print(f'2) Вложили столбцы связанные с переменными без ограничений:\n{A}')

    """
    Неравенства типа '>=' переводятся в неравенства '<=' в случае задачи max;
    иначе - наоборот.
    """

    if is_max:
        for i in range(equation_less, equation_less + equation_more, 1):
            A[i, :] = -A[i, :]
            b[i] = -b[i]
    else:
        for i in range(0, equation_less, 1):
            A[i, :] = -A[i, :]
            b[i] = -b[i]

    # print(f'3) Матрица со строчками, в которым изменился знаки нер-в:\n{A}')

    """
    добавление переменных в неравенства, чтобы сделать их равенствами
    """

    if is_max:
        for i in range(0, equation_less + equation_more, 1):
            A[i, N_General + X_Any + i] = 1
            perem_[N_General + X_Any + i] = f'{add_param_letter}{N_General + X_Any + i}'
    else:
        for i in range(0, equation_less + equation_more, 1):
            A[i, N_General + X_Any + i] = -1
            perem_[N_General + X_Any + i] = f'{add_param_letter}{N_General + X_Any + i}'

    # print(f'4) Добавили переменные в строчки с неравенствами:\n{A}')
    return A, b, perem_, c_


def update_c_without_preproc(c, Equation_Less, Equation_More, X_Any):
    """
    Размер матрицы после приведения в каноническую форму мог быть изменен.
    Приведем в соответствие вектор с
    """
    c_reshaped = list(c) + list(np.zeros(Equation_Less + Equation_More + X_Any))
    X_Positive = len(c) - X_Any

    """представление вектора c через новые переменные"""
    for i in range(0, X_Positive, 1):
        c_reshaped[i] = c[i]
    for i in range(X_Positive, X_Positive + X_Any, 1):
        c_reshaped[2 * i - X_Positive] = c[i]
        c_reshaped[2 * i - (X_Positive - 1)] = -c[i]
    return np.array(c_reshaped)


def update_c(M, b, c, c_free, Basis_Indexes, Equation_Less, Equation_More, X_Any):
    c_reshaped = update_c_without_preproc(c, Equation_Less, Equation_More, X_Any)

    """представление вектора с через небазисные компоненты (обнуление базисных компонент)"""
    for i in range(M.shape[0]):
        """в знаменателе обязана быть 1"""
        coef = c_reshaped[Basis_Indexes[i]] / M[i, Basis_Indexes[i]]
        c_reshaped[:] = c_reshaped[:] - coef * M[i][:]
        c_free = c_free - coef * b[i]
    return c_reshaped, c_free
