import os
import random
import torch
import numpy as np


def seed_fix(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = False  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


