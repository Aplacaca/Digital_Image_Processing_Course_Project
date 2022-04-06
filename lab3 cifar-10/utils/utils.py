import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mertix(predict, y):
    index = (np.array(predict) > 0.5).reshape(-1)
    if index.sum() != 0:
        P = y[index].sum() / index.sum()
        R = y[index].sum() / y.sum()
    else:
        P = 0
        R = 0

    return P, R
