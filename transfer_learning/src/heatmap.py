import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 读取数据
path = './cache/mask/id_train.csv'
df = pd.read_csv(path)
df.columns = ['Dataset ID', 'Fold Numbers', df.columns[2]]
df[df.columns[2]] = abs(df[df.columns[2]])

# 创建Pivot Table
pivot_table = df.pivot(index=df.columns[0], columns=df.columns[1], values=df.columns[2])

# 绘制热力图
plt.figure(figsize=(12, 9), dpi=400)  # 增加图片尺寸
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5,
            annot_kws={"size": 14})  # 更改颜色映射，格式化注释，增大注释字体大小
plt.title('Increasing Heatmap', fontsize=20)  # 增大标题字体大小
plt.xlabel(df.columns[1], fontsize=18)  # 增大x轴标签字体大小
plt.ylabel(df.columns[0], fontsize=18)  # 增大y轴标签字体大小
plt.xticks(fontsize=16)  # 增大x轴刻度字体大小
plt.yticks(fontsize=16)  # 增大y轴刻度字体大小
plt.savefig("./cache/mask/id_train.pdf", bbox_inches='tight')
