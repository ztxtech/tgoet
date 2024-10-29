import numpy as np
from evidence_theory.core.atom import Element
from evidence_theory.core.distribution import Evidence
from evidence_theory.element.combination import powerset

from transfer_distance.src.solve2 import solve


def ele_dis(a, b):
    c = a.value.intersection(b)
    a_p = powerset(a)
    b_p = powerset(b)
    c_p = powerset(c)
    return len(a_p) + len(b_p) - 2 * len(c_p)


def cost_matrix(a, b):
    res = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            res[i][j] = ele_dis(x, y)
    return res


ev1 = Evidence(
    {
        Element({'a'}): 0.3,
        Element({'b', 'c'}): 0.5,
        Element({'a', 'b', 'c'}): 0.2
    }
)

ev2 = Evidence(
    {
        Element({'b'}): 0.1,
        Element({'c'}): 0.2,
        Element({'b', 'c'}): 0.3,
        Element({'a', 'c'}): 0.1,
        Element({'a', 'b', 'c'}): 0.3
    }
)

m = list(ev1.keys())
n = list(ev2.keys())
c = cost_matrix(m, n)

a = np.array([ev1[ele] for ele in m])
b = np.array([ev2[ele] for ele in n])

res, x = solve(a, b, c)
print(res.fun)
