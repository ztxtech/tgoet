import os

import dill
import pandas as pd
from dstz.element.combination import powerset
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo

from tl.utils import Gaussianer


def filter_columns(df):
    # 创建一个新的DataFrame来保存结果
    result_df = pd.DataFrame()

    # 遍历DataFrame中的每一列
    for column in df.columns:
        # 检查当前列是否全部为数值型
        if pd.to_numeric(df[column], errors='coerce').notnull().all():
            # 计算当前列的标准差
            std_dev = df[column].std(ddof=0)  # 使用ddof=0来计算总体标准差

            # 如果标准差大于等于0.001，则保留该列
            if std_dev > 1e-10:
                result_df[column] = df[column]
        else:
            # 如果不是数值型，则直接跳过该列
            continue

    return result_df


class UCIDataset(Dataset):
    def __init__(self, id, path="./cache"):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.filepath = os.path.join(path, f'{id}.pkl')

        if os.path.exists(self.filepath):
            # Load existing dataset
            self.load(self.filepath)
            print(f'UCI Dataset ID:{id} Loaded from {self.filepath}.')
        else:
            dataset = fetch_ucirepo(id=id)
            print(f'UCI Dataset ID:{id} Download Successfully')

            x = dataset.data.features
            self.x_origin = filter_columns(x)
            self.feature = self.x_origin.columns
            self.feature_num = len(self.feature)
            self.x_origin.columns = list(range(self.feature_num))
            self.x = self.x_origin.values.tolist()
            self.feature_index = list(range(self.feature_num))

            self.y_origin = dataset.data.targets
            self.y_origin.columns = ['target']
            self.y = self.y_origin.values.reshape(-1).tolist()
            self.target = list(set(self.y))
            self.target_num = len(self.target)
            self.target_map = {t: idx for idx, t in enumerate(self.target)}
            self.y = [self.target_map[t] for t in self.y]
            self.target_index = list(range(self.target_num))
            self.y_origin.loc[:, 'target'] = self.y_origin['target'].map(self.target_map)
            self.ps = list(powerset(self.target_index))

            # Save the dataset
            self.save(self.filepath)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def save(self, filepath):
        """Save the dataset object to a file using dill."""
        with open(filepath, 'wb') as f:
            dill.dump(self, f)

    def load(self, filepath):
        """Load the dataset object from a file using dill."""
        with open(filepath, 'rb') as f:
            loaded_dataset = dill.load(f)
        self.__dict__.update(loaded_dataset.__dict__)

    def get_gaussianer(self, index):
        full_df = pd.concat([self.x_origin, self.y_origin], axis=1)
        full_df = full_df.iloc[index, :]
        mean_df = full_df.groupby('target').mean().reset_index()
        std_df = full_df.groupby('target').std().reset_index()

        self.gaussian_df = None

        # 遍历每个特征和每个目标值
        for target in mean_df['target']:
            for feature in self.feature_index:
                mean = mean_df.loc[mean_df['target'] == target, feature].values[0]
                std = std_df.loc[std_df['target'] == target, feature].values[0]

                if self.gaussian_df is not None:
                    # 将 Guassianer 对象的信息添加到 DataFrame 中
                    self.gaussian_df = pd.concat(
                        [self.gaussian_df,
                         pd.DataFrame({'target': [target], 'feature': [feature], 'mean': [mean], 'std': [std]})],
                        ignore_index=True)
                else:
                    self.gaussian_df = pd.DataFrame(
                        {'target': [target], 'feature': [feature], 'mean': [mean], 'std': [std]})

        gaussianer = Gaussianer(self.gaussian_df)
        return gaussianer


class ExpDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # 根据索引获取样本
        return self.x[index], self.y[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.x)
