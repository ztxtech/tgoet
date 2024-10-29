def solve(a, b, c):
    # 检查输入有效性
    if sum(a) != sum(b):
        raise ValueError("Supply and demand must be balanced.")

    m = len(a)
    n = len(b)

    # 初始化分配矩阵
    x = [[0 for _ in range(n)] for _ in range(m)]

    # 最小元素法分配
    while True:
        # 找到未耗尽行和列中的最小成本
        min_cost = float('inf')
        row_index, col_index = -1, -1

        for i in range(m):
            for j in range(n):
                if a[i] > 0 and b[j] > 0 and c[i][j] < min_cost:
                    min_cost = c[i][j]
                    row_index, col_index = i, j

        # 如果没有剩余需求或供给，则结束
        if row_index == -1 or col_index == -1:
            break

        # 确定可分配的最大数量
        alloc = min(a[row_index], b[col_index])

        # 进行分配
        x[row_index][col_index] = alloc
        a[row_index] -= alloc
        b[col_index] -= alloc

    return x
