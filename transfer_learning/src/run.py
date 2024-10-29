# 53 697 763 143 151d 52 936 45 17 144 544
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# 假设以下导入已经正确
from tl.data import UCIDataset
from tl.exp import Exp
from tl.model import tl


def generate_combinations(elements):
    all_combinations = []
    # 生成从1到len(elements)的所有长度的组合
    for r in [1, 2, 3, 7, 8, 9]:
        combinations = list(itertools.combinations(elements, r))
        all_combinations.extend(combinations)
    return all_combinations


def fun(id, train_index):
    print(id, train_index)
    dataset = UCIDataset(id)
    model = tl(dataset.ps, dataset.target_num, False)
    # e-2 10
    exp = Exp(model=model, dataset=dataset, device='cuda:0', train_index=train_index, batch_size=32, seed=2024, lr=1e-2,
              epoch=10)
    exp.ppt()
    exp.train()

    return id, train_index, exp.accuracy_ppt, exp.accuracy_val[-1]


if __name__ == '__main__':
    # 参数准备 350 850
    ids = [17, 52, 53, 109, 143, 144, 145, 151, 697, 763, 936]

    workers = 16
    res = []

    train_indexes = generate_combinations(range(10))

    if workers > 1:
        # 使用进程池执行 fun 函数
        with ProcessPoolExecutor(workers) as executor:
            # 创建任务列表
            tasks = [(id, train_index) for id in ids for train_index in train_indexes]

            # 提交任务并获取 Future 对象
            futures = [executor.submit(fun, task[0], task[1]) for task in tasks]

            # 按任务完成顺序收集结果
            for future in as_completed(futures):
                result = future.result()
                res.append(result)
    else:
        for id in ids:
            for train_index in train_indexes:
                res.append(fun(id, train_index))

    # 将 myres 转换为 JSON 格式并保存到文件
    with open('./cache/myparams.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
