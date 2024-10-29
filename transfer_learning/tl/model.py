import torch
from torch import nn


class tl(nn.Module):
    def __init__(self, ps, target_num, mask=True):
        super(tl, self).__init__()
        self.ps = ps
        self.target_num = target_num
        self.T = nn.Parameter(self.ppt_matrix(ps, target_num))
        self.constraint = nn.Parameter(self.mask_matrix(ps, target_num), requires_grad=False)
        # self.act = nn.Sigmoid()
        self.mask = mask
        self.act = nn.ReLU()
        self.initial = False

    def ppt_matrix(self, ps, target_num):
        res = torch.zeros((len(ps), target_num))
        for i, key in enumerate(ps):
            for j in key.value:
                res[i, j] = 1.0 / len(key)
        return res

    def mask_matrix(self, ps, target_num):
        res = torch.zeros((len(ps), target_num))
        for i, key in enumerate(ps):
            for j in range(target_num):
                if j in key:
                    res[i, j] = 1
        return res

    def printT(self):
        if self.initial:
            w = self.act(self.T)
            if self.mask:
                w = w * self.constraint
            w_sum = torch.sum(w, dim=1, keepdim=True)
            w = w / w_sum
        else:
            w = self.T
        print(self.ps)
        print(w)

    def forward(self, x):
        if self.initial:
            w = self.act(self.T)
            if self.mask:
                w = w * self.constraint
            w_sum = torch.sum(w, dim=1, keepdim=True)
            w = w / w_sum
        else:
            w = self.T
            self.initial = True
        res = torch.matmul(x, w)
        return res
