from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms as tf
from timm.data.distributed_sampler import RepeatAugSampler
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader, default_collate, Dataset, random_split, Subset
from torch.utils.data.dataloader import _collate_fn_t
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
from src.config import Config

from src.util import get_path


def collate_and_transform(batch: List[Tuple[torch.Tensor, int]], use_gpu: bool,
                          image_tf: Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                          batch_tf: Optional[Mixup]=None) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: torch.Tensor
    labels: torch.Tensor
    inputs, labels = default_collate(batch)

    if use_gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()

    if image_tf is not None:
        inputs = image_tf(inputs)
    if batch_tf is not None:
        inputs, labels = batch_tf(inputs, labels)

    return inputs, labels


def init_datasets(root_path: Path, config: Config) -> None:
    data_path: Path = get_path(root_path, 'data')
    path_str: str = data_path.as_posix()
    train_dev_set = CIFAR100(path_str, train=True, transform=tf.PILToTensor(), download=True)
    test_set = CIFAR100(path_str, train=False, transform=tf.PILToTensor(), download=True)

    if config.use_subset:
        train_dev_set = Subset(train_dev_set, list(range(2000)))
        test_set = Subset(test_set, list(range(400)))

    train_size: int = round(len(train_dev_set) * config.train_ratio)
    dev_size: int = len(train_dev_set) - train_size

    train_set: Dataset
    dev_set: Dataset
    train_set, dev_set = random_split(train_dev_set, [train_size, dev_size],
                                      generator=torch.Generator().manual_seed(config.seed))

    torch.save(train_set, data_path / 'train.pt')
    torch.save(dev_set, data_path / 'dev.pt')
    torch.save(test_set, data_path / 'test.pt')


def init_loaders(root_path: Path, config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_path: Path = get_path(root_path, 'data')
    train_set: Dataset = torch.load(data_path / 'train.pt')
    dev_set: Dataset = torch.load(data_path / 'dev.pt')
    test_set: Dataset = torch.load(data_path / 'test.pt')

    train_sampler = RepeatAugSampler(train_set, num_replicas=1, rank=0,
                                     num_repeats=config.num_repeated_aug, selected_round=0)
    dev_sampler = DistributedSampler(dev_set, num_replicas=1, rank=0, seed=config.seed)
    test_sampler = DistributedSampler(test_set, num_replicas=1, rank=0, seed=config.seed)

    use_gpu: bool = config.num_gpus > 0
    normalize = tf.Normalize(config.normalize_mean, config.normalize_std)

    train_tf = tf.Compose([
        tf.RandomResizedCrop(32, scale=(min(0.99, config.random_resize_bound), 1)),
        tf.RandomHorizontalFlip(),
        tf.TrivialAugmentWide(),
        tf.ConvertImageDtype(torch.float),
        normalize,
        tf.RandomErasing(min(1, config.random_erasing_p))
    ])

    test_tf = tf.Compose([
        tf.ConvertImageDtype(torch.float),
        normalize
    ])

    mixup_cutmix = Mixup(mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha,
                         label_smoothing=config.label_smoothing, num_classes=100)

    train_collate: _collate_fn_t = partial(collate_and_transform, use_gpu=use_gpu,
                                           image_tf=train_tf, batch_tf=mixup_cutmix)
    test_collate: _collate_fn_t = partial(collate_and_transform, use_gpu=use_gpu, image_tf=test_tf)

    train_loader = DataLoader(train_set, batch_size=config.train_batch_size,
                              sampler=train_sampler, collate_fn=train_collate)
    dev_loader = DataLoader(dev_set, batch_size=config.test_batch_size,
                             sampler=dev_sampler, collate_fn=test_collate)
    test_loader = DataLoader(test_set, batch_size=config.test_batch_size,
                             sampler=test_sampler, collate_fn=test_collate)

    return train_loader, dev_loader, test_loader
