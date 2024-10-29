import numpy as np
from scipy.optimize import linprog


def solve(a, b, c):
    # 检查输入有效性
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and isinstance(c, np.ndarray)):
        raise TypeError("Inputs must be NumPy arrays.")

    if a.ndim != 1 or b.ndim != 1 or c.ndim != 2:
        raise ValueError("a and b must be 1D arrays, and c must be a 2D array.")

    if sum(a) != sum(b):
        raise ValueError("Supply and demand must be balanced.")

    m, n = c.shape

    # 定义决策变量
    num_vars = m * n

    # 目标函数系数
    c_flat = c.flatten()

    # 约束条件
    A_eq = []
    b_eq = []

    # 供应约束
    for i in range(m):
        row = np.zeros(num_vars)
        row[i * n:(i + 1) * n] = 1
        A_eq.append(row)
        b_eq.append(a[i])

    # 需求约束
    for j in range(n):
        row = np.zeros(num_vars)
        row[j::n] = 1
        A_eq.append(row)
        b_eq.append(b[j])

    # 转换为 NumPy 数组
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # 线性规划求解
    res = linprog(c=c_flat, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))

    # 将解转换为分配矩阵
    x_matrix = res.x.reshape((m, n))

    return res, x_matrix