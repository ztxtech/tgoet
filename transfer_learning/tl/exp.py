from random import shuffle

import torch
from dstz.core.atom import Element
from dstz.math.matrix.dual import conjunctive_rule
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tl.data import ExpDataset
from tl.utils import mass_builder, vec_builder, seed_everything


class Exp:
    def __init__(self, model=None, dataset=None, device='cpu', fold=10, train_index=None, batch_size=10, seed=2025,
                 lr=1e-3,
                 epoch=10):

        if train_index is None:
            train_index = [0, 1, 2]
        self.seed = seed
        seed_everything(self.seed)
        self.model = model
        self.dataset = dataset
        self.device = device
        self.is_training = False
        self.fold = fold
        self.train_index = train_index
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr

        self.model.to(self.device)

        self.gaussianer = None

        self.train_dataset = None
        self.test_dataset = None

        self.train_mass = None
        self.test_mass = None

        self.train_mass_f = None
        self.test_mass_f = None

        self.train_vec = None
        self.test_vec = None

        self.train_exp_dataset = None
        self.test_exp_dataset = None

        self.train_data_loader = None
        self.test_data_loader = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
        #                                    patience=max(int(self.epoch * 0.1), 2),
        #                                    )

        self.labels_ppt = None
        self.output_ppt = None
        self.output_labels_ppt = None
        self.total_ppt = None
        self.correct_ppt = None
        self.accuracy_ppt = None

        self.loss_train = None
        self.accuracy_train = None
        self.labels_train = None
        self.output_train = None
        self.output_labels_train = None
        self.correct_train = None  # 用于跟踪每个训练周期中分类正确的样本数
        self.total_train = None

        self.loss_val = None
        self.accuracy_val = None
        self.labels_val = None
        self.output_val = None
        self.output_labels_val = None
        self.correct_val = None  # 用于跟踪每个训练周期中分类正确的样本数
        self.total_val = None  # 用于跟踪每个训练周期中的总样本数

        self.train_test_split()

    def data2mass(self, dataset):
        data_mass = []
        print("DATA2MASS")
        for data_piece in tqdm(dataset):
            data_mass_piece = []
            for f_idx, feature in enumerate(data_piece[0]):
                mass = {}
                for t_idx in self.dataset.target_index:
                    try:
                        mass[t_idx] = self.gaussianer.pdf(feature, f_idx, t_idx)
                    except:
                        continue
                ev = mass_builder(mass)
                data_mass_piece.append(ev)
            data_mass.append(data_mass_piece)
        return data_mass

    # def ds_rule(self, m1, m2):
    #     return ds_rule(m1, m2)

    def ds_rule(self, m1, m2):
        m = conjunctive_rule(m1, m2)
        if Element(set()) in m:
            empty_mass = m.pop(Element(set()))
            for key in m.keys():
                m[key] = m[key] / (1 - empty_mass)
        return m

    def massfusion(self, data_mass):
        mass_f = []
        print("MASS_FUSION")
        for data_mass_piece in tqdm(data_mass):
            tmp = data_mass_piece[0]
            for index in range(1, len(data_mass_piece)):
                tmp = self.ds_rule(tmp, data_mass_piece[index])
            mass_f.append(tmp)
        return mass_f

    def mass2vec(self, mass):
        vec = []
        for m in mass:
            vec.append(vec_builder(m, self.dataset.ps))
        return vec

    def train_test_split(self):

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        shuffle(indices)
        split_size = dataset_size // self.fold
        # 创建 k_folds 个子集
        subsets = [indices[i * split_size:(i + 1) * split_size] for i in range(self.fold)]
        train_subset_indices = []
        test_subset_indices = []
        for idx, subset in enumerate(subsets):
            if idx in self.train_index:
                train_subset_indices.extend(subset)
            else:
                test_subset_indices.extend(subset)

        self.train_dataset = Subset(self.dataset, train_subset_indices)
        self.test_dataset = Subset(self.dataset, test_subset_indices)

        self.gaussianer = self.dataset.get_gaussianer(train_subset_indices)

        self.train_mass = self.data2mass(self.train_dataset)
        self.test_mass = self.data2mass(self.test_dataset)

        self.train_mass_f = self.massfusion(self.train_mass)
        self.test_mass_f = self.massfusion(self.test_mass)

        self.train_vec = self.mass2vec(self.train_mass_f)
        self.test_vec = self.mass2vec(self.test_mass_f)

        self.train_exp_dataset = ExpDataset(self.train_vec,
                                            torch.tensor([self.dataset.y[i] for i in self.train_dataset.indices],
                                                         dtype=torch.long))
        self.test_exp_dataset = ExpDataset(self.test_vec,
                                           torch.tensor([self.dataset.y[i] for i in self.test_dataset.indices],
                                                        dtype=torch.long))

        self.train_data_loader = DataLoader(
            dataset=self.train_exp_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test_data_loader = DataLoader(
            dataset=self.test_exp_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def ppt(self):
        print("-" * 8 + "PPT" + "-" * 8)

        self.correct_ppt = 0  # 用于跟踪每个验证周期中分类正确的样本数
        self.total_ppt = 0  # 用于跟踪每个验证周期中的总样本数
        self.labels_ppt = []
        self.output_ppt = []
        self.output_labels_ppt = []

        data_loader = self.test_data_loader

        self.model.eval()
        self.model.printT()

        with (torch.no_grad()):
            # 遍历训练数据集中的批次
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                b, _, f = outputs.shape
                outputs = outputs.reshape(b, f)  # 通过模型前向传播计算预测
                _, predicted = torch.max(outputs.data, 1)  # 找到每个样本的预测类别

                self.total_ppt += labels.size(0)  # 累积总样本数
                self.correct_ppt += (predicted == labels).sum().item()  # 累积分类正确的样本数
                self.labels_ppt = self.labels_ppt + labels.detach().cpu().tolist()
                self.output_ppt = self.output_ppt + outputs.detach().cpu().tolist()
                self.output_labels_ppt = self.output_labels_ppt + predicted.detach().cpu().tolist()

            self.accuracy_ppt = 100 * self.correct_ppt / self.total_ppt
            print(f"Accuracy: {self.accuracy_ppt:.2f}%")

    def train(self):
        print("-" * 8 + "TRAIN" + "-" * 8)

        self.loss_train = []
        self.accuracy_train = []
        self.labels_train = []
        self.output_train = []
        self.output_labels_train = []
        self.correct_train = []  # 用于跟踪每个训练周期中分类正确的样本数
        self.total_train = []  # 用于跟踪每个训练周期中的总样本数

        self.loss_val = []
        self.accuracy_val = []
        self.labels_val = []
        self.output_val = []
        self.output_labels_val = []
        self.correct_val = []  # 用于跟踪每个训练周期中分类正确的样本数
        self.total_val = []  # 用于跟踪每个训练周期中的总样本数

        for epoch in range(self.epoch):
            print("-" * 8 + f"Epoch: {epoch + 1}" + "-" * 8)
            # 设置模型为训练模式，这将启用梯度计算和参数更新
            self.model.train()
            loss_train_epoch = 0.0  # 用于跟踪每个训练周期的总损失
            total_train_epoch = 0
            correct_train_epoch = 0
            labels_train_epoch = []
            output_train_epoch = []
            output_labels_train_epoch = []

            # 遍历训练数据集中的批次
            for inputs, labels in self.train_data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                b, _, f = outputs.shape
                outputs = outputs.reshape(b, f)  # 通过模型前向传播计算预测

                loss = self.criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播，计算梯度
                self.optimizer.step()  # 根据梯度更新模型参数
                # self.scheduler.step(loss)

                loss_train_epoch += loss.item()  # 累积损失
                _, predicted = torch.max(outputs.data, 1)  # 找到每个样本的预测类别
                total_train_epoch += labels.size(0)  # 累积总样本数
                correct_train_epoch += (predicted == labels).sum().item()  # 累积分类正确的样本数
                labels_train_epoch = labels_train_epoch + labels.detach().cpu().tolist()
                output_train_epoch = output_train_epoch + outputs.detach().cpu().tolist()
                output_labels_train_epoch = output_labels_train_epoch + predicted.detach().cpu().tolist()

            # 计算并记录训练准确率和损失
            accuracy_train_epoch = 100 * correct_train_epoch / total_train_epoch
            self.total_train.append(total_train_epoch)
            self.correct_train.append(correct_train_epoch)
            self.accuracy_train.append(accuracy_train_epoch)
            self.loss_train.append(loss_train_epoch / len(self.train_data_loader))
            self.labels_train.append(labels_train_epoch)
            self.output_train.append(output_train_epoch)
            self.output_labels_train.append(output_labels_train_epoch)
            self.model.printT()
            # print(f"Train Accuracy: {accuracy_train_epoch:.2f}%")

            # print("-" * 4 + f"Epoch Validation: {epoch + 1}" + "-" * 4)

            # 进入验证循环
            self.model.eval()  # 设置模型为评估模式，不进行梯度计算和参数更新
            loss_val_epoch = 0.0  # 用于跟踪每个训练周期的总损失
            total_val_epoch = 0
            correct_val_epoch = 0
            labels_val_epoch = []
            output_val_epoch = []
            output_labels_val_epoch = []

            with torch.no_grad():  # 在验证循环中不进行梯度计算
                # 遍历训练数据集中的批次
                for inputs, labels in self.test_data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    b, _, f = outputs.shape
                    outputs = outputs.reshape(b, f)  # 通过模型前向传播计算预测

                    loss = self.criterion(outputs, labels)  # 计算损失

                    loss_val_epoch += loss.item()  # 累积损失
                    _, predicted = torch.max(outputs.data, 1)  # 找到每个样本的预测类别
                    total_val_epoch += labels.size(0)  # 累积总样本数
                    correct_val_epoch += (predicted == labels).sum().item()  # 累积分类正确的样本数
                    labels_val_epoch = labels_val_epoch + labels.detach().cpu().tolist()
                    output_val_epoch = output_val_epoch + outputs.detach().cpu().tolist()
                    output_labels_val_epoch = output_labels_val_epoch + predicted.detach().cpu().tolist()

            # 计算并记录训练准确率和损失
            accuracy_val_epoch = 100 * correct_val_epoch / total_val_epoch
            self.total_val.append(total_val_epoch)
            self.correct_val.append(correct_val_epoch)
            self.accuracy_val.append(accuracy_val_epoch)
            self.loss_val.append(loss_val_epoch / len(self.test_data_loader))
            self.labels_val.append(labels_val_epoch)
            self.output_val.append(output_val_epoch)
            self.output_labels_val.append(output_labels_val_epoch)
            self.model.printT()
            # print(f"Validation Accuracy: {accuracy_val_epoch:.2f}%")

            # 打印训练和验证结果
            print(
                f"Epoch [{epoch + 1}/{self.epoch}] - Train Loss: {loss_train_epoch:.4f}, "
                f"Train Accuracy: {accuracy_train_epoch:.2f}%, Validation Loss: {loss_val_epoch:.4f}, "
                f"Validation Accuracy: {accuracy_val_epoch:.2f}%")
