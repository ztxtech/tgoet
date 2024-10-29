import random

import numpy as np
import torch
from dstz.core.atom import Element
from dstz.core.distribution import Evidence
from scipy.stats import norm


def mass_builder(mass_dict):
    res = Evidence()
    if not mass_dict:
        return res

    total_mass = sum(mass_dict.values())
    normalized_mass_dict = {key: value / total_mass for key, value in mass_dict.items()}
    sorted_keys = sorted(normalized_mass_dict, key=normalized_mass_dict.get, reverse=True)

    for idx, key in enumerate(sorted_keys):
        res[Element(set(sorted_keys[:idx + 1]))] = normalized_mass_dict[key]

    return res


def vec_builder(ev, ps):
    res = torch.zeros((1, len(ps)))
    for idx, key in enumerate(ps):
        if key in ev:
            res[0, idx] = ev[key]
    return res


class Gaussianer:
    def __init__(self, gaussian_df):
        self.gaussian_df = gaussian_df

    def pdf(self, x, feature, target):
        params = self.gaussian_df.loc[(self.gaussian_df['feature'] == feature) & (self.gaussian_df['target'] == target)]
        if params.empty:
            raise ValueError(f"No parameters found for feature {feature} and target {target}")

        mean = params['mean'].values[0]
        std = params['std'].values[0]

        if std > 0:
            return norm.pdf(x, loc=mean, scale=std)
        else:
            if x == mean:
                return 1.00
            else:
                return 0.00


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
