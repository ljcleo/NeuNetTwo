import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from src.config import Config


def fix_seed(config: Config) -> None:
    seed = config.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config.num_gpus > 0:
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
