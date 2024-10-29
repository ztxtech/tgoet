import random

N = 50
C = 4
FOD = 5


def basic_exp(N, C, FOD):
    c = [{} for i in range(C)]
    for i in range(N):
        fod = [chr(i) for i in range(ord('a'), ord('z') + 1)][:FOD]
        cur_i = random.choice(fod)
        cur_c = random.randint(0, C - 1)
        if cur_i not in c[cur_c]:
            c[cur_c][cur_i] = 1
        else:
            c[cur_c][cur_i] += 1
    return c


# 初始化容器
c = basic_exp(N, C, FOD)

c_history = []

for i in range(1, N):
    # 计算每个容器的总元素数
    total_counts = [sum(container.values()) for container in c]

    # 根据总元素数计算累积概率分布
    cumulative_probabilities = []
    cumulative_sum = 0
    for count in total_counts:
        cumulative_sum += count
        cumulative_probabilities.append(cumulative_sum)

    # 随机选择一个容器
    rand_num = random.randint(1, cumulative_sum)
    chosen_container_index = next(i for i, cum_prob in enumerate(cumulative_probabilities) if rand_num <= cum_prob)

    # 从选定的容器中随机选择一个元素，根据比例
    chosen_container = c[chosen_container_index]
    total_elements = sum(chosen_container.values())
    rand_num = random.randint(1, total_elements)
    cumulative_probabilities = []
    cumulative_sum = 0
    for key, value in chosen_container.items():
        cumulative_sum += value
        cumulative_probabilities.append(cumulative_sum)
        if rand_num <= cumulative_sum:
            chosen_element = key
            break

    # 抽中的元素数值减一
    chosen_container[chosen_element] -= 1

    # 记录历史
    c_history.append((chosen_container_index, chosen_element))

# 打印结果
print("Containers:", c)
print("History:", c_history)
