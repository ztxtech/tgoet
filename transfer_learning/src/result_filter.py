import json
import os

import pandas as pd

# 定义缓存路径变量
CACHE_PATH = './cache/mask'

# 读取JSON文件
json_file_path = os.path.join(CACHE_PATH, 'myparams.json')
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['id', 'train', 'ppt', 'tl'])
df['ppt-tl'] = df['ppt'] - df['tl']
csv_file_path = os.path.join(CACHE_PATH, "myparams.csv")

df.to_csv(csv_file_path)

# 新增一列，表示train列的长度
df['train_len'] = df['train'].apply(len)

# 按照train_len分组，并计算ppt和tl的方差和均值
# grouped = df.groupby(['id'])[['ppt', 'tl']].agg(['mean', 'std'])
grouped = df.groupby(['id', 'train_len'])[['ppt', 'tl']].agg(['mean', 'std'])

# 重置索引以便更好地查看结果
grouped.reset_index(inplace=True)

# 添加新列，表示ppt的均值是否比tl的小
grouped['mean_ppt_tl'] = grouped[('ppt', 'mean')] - grouped[('tl', 'mean')]

# 添加新列，表示ppt的方差是否比tl的大
grouped['var_ppt_tl'] = grouped[('ppt', 'std')] - grouped[('tl', 'std')]

# 显示结果
print(grouped)

output_file_path = os.path.join(CACHE_PATH, "filter.csv")
grouped.to_csv(output_file_path)
