import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from yaml import safe_load

from src.command import CommandLineArgs
from src.gpu import init_gpu
from src.util import get_path


@dataclass
class Config:
    seed: int
    debug_level: int
    use_subset: bool
    train_ratio: float
    normalize_mean: List[float]
    normalize_std: List[float]
    random_resize_bound: float
    random_erasing_p: float
    mixup_alpha: float
    cutmix_alpha: float
    label_smoothing: float
    num_repeated_aug: int
    model_name: str
    model_type: str
    drop_path_rate: float
    optimizer: str
    lr: float
    weight_decay: float
    num_epochs: int
    train_batch_size: int
    test_batch_size: int
    num_cpus: int
    num_gpus: int
    num_samples: int
    perturbation_interval: int


def init_config(root_path: Path, args: CommandLineArgs) -> Config:
    config_path: Path = get_path(root_path, 'config')

    with (config_path / 'default.yaml').open('r', encoding='utf8') as f:
        main_config: Dict[str, Any] = safe_load(f)
        main_config.update({'model_name': args.model, 'debug_level': args.debug_level,
                            'use_subset': args.use_subset,
                            'num_cpus': min(args.max_cpu_num, multiprocessing.cpu_count()),
                            'num_gpus': init_gpu(args)})

    with (config_path / f'{args.model}.yaml').open('r', encoding='utf8') as f:
        main_config.update(safe_load(f))
    config = Config(**main_config)

    if config.debug_level >= 1:
        config.num_epochs = min(config.num_epochs, 2)
        config.num_samples = min(config.num_samples, 2)
        config.perturbation_interval = 1

    if config.debug_level >= 2:
        config.num_epochs = min(config.num_epochs, 1)

    if config.use_subset:
        config.test_batch_size = min(config.test_batch_size, 64)

    return config
