import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tl.data import UCIDataset

# 设置 seaborn 主题
sns.set_theme(style="whitegrid")  # 选择你喜欢的主题

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 设置字体大小
plt.rcParams['font.size'] = 16  # 标签和刻度的字体大小
plt.rcParams['axes.titlesize'] = 18  # 标题的字体大小
plt.rcParams['axes.labelsize'] = 16  # 轴标签的字体大小
plt.rcParams['xtick.labelsize'] = 14  # x轴刻度的字体大小
plt.rcParams['ytick.labelsize'] = 14  # y轴刻度的字体大小

# 加载数据
data = UCIDataset(53)
data = [data.x[i] + [data.y[i]] for i in range(len(data))]

# 将数据转换为 DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Class'])

plt.figure(figsize=(12, 9), dpi=500)  # 增加图片尺寸
# 使用 seaborn 的 pairplot 绘制散点图矩阵
sns.pairplot(df, hue='Class', palette='tab10', markers=['o', 's', 'D'])  # 指定颜色方案和标记样式

plt.savefig("./cache/mask/53.pdf", bbox_inches='tight')
