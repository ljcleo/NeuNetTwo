from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import ray.tune as tune
import torch
from ray.tune.progress_reporter import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from timm.optim import create_optimizer_v2
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from src.config import Config
from src.data import DataLoader, init_datasets, init_loaders
from src.init import fix_seed
from src.model import ImageClsModel
from src.util import clean_path, get_path


@dataclass
class PerDeviceTrainer:
    config: Config

    def prepare(self, root_path: Path, epoch: int = 0,
                model_checkpoint: Optional[OrderedDict[str, torch.Tensor]] = None,
                optimizer_checkpoint: Optional[Dict] = None) -> None:
        print('Preparing for training ...')
        self._train_loader: DataLoader
        self._dev_loader: DataLoader
        self._test_loader: DataLoader

        self._train_loader, self._dev_loader, self._test_loader = init_loaders(
            root_path, self.config
        )

        self._num_train_samples: int = len(self._train_loader.dataset)
        self._num_dev_samples: int = len(self._dev_loader.dataset)
        self._num_test_samples: int = len(self._test_loader.dataset)
        self._num_steps_per_train_epoch: int = len(self._train_loader)
        self._num_steps_per_dev_epoch: int = len(self._dev_loader)
        self._num_steps_per_test_epoch: int = len(self._test_loader)

        if self.config.debug_level >= 2:
            self._num_train_samples = self.config.train_batch_size
            self._num_dev_samples = self.config.test_batch_size
            self._num_test_samples = self.config.test_batch_size
            self._num_steps_per_train_epoch = 1
            self._num_steps_per_dev_epoch = 1
            self._num_steps_per_test_epoch = 1

        self._model = ImageClsModel(self.config)
        if model_checkpoint is not None:
            self._model.load_state_dict(model_checkpoint)

        if self.config.num_gpus > 0:
            self._model.cuda()

        self._optimizer: Optimizer = create_optimizer_v2(
            self._model, opt=self.config.optimizer, lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        if optimizer_checkpoint is not None:
            self._optimizer.load_state_dict(optimizer_checkpoint)
        self.epoch: int = epoch

    def train(self) -> float:
        print(f'Training epoch {self.epoch + 1} ...')
        train_loss: float = 0.0
        progress = tqdm(desc='Training', total=self._num_steps_per_train_epoch,
                        mininterval=5, miniters=10)

        for inputs, labels in self._train_loader:
            inputs: torch.Tensor
            labels: torch.Tensor
            loss: torch.Tensor = self._model.forward(inputs, target=labels)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            train_loss += loss.item() * inputs.size(0)
            progress.update()

            if self.config.debug_level >= 2:
                break

        self.epoch += 1
        progress.close()
        return train_loss / self._num_train_samples

    def test(self, dev: bool) -> Tuple[float, float, float]:
        print(f'Evaluating on {"dev" if dev else "test"} set ...')
        loader: DataLoader = self._dev_loader if dev else self._test_loader
        num_samples: int = self._num_dev_samples if dev else self._num_test_samples

        test_loss: float = 0.0
        test_top1_correct: int = 0
        test_top5_correct: int = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs: torch.Tensor
                labels: torch.Tensor
                cur_loss: torch.Tensor
                cur_top1_correct: torch.Tensor
                cur_top5_correct: torch.Tensor

                cur_loss, cur_top1_correct, cur_top5_correct = self._model.forward(
                    inputs, target=labels, pred_eval=True
                )

                test_loss += cur_loss.item()
                test_top1_correct += cur_top1_correct.item()
                test_top5_correct += cur_top5_correct.item()

                if self.config.debug_level >= 2:
                    break

        return (test_loss / num_samples, test_top1_correct / num_samples,
                test_top5_correct / num_samples)

    def checkpoint(self) -> Tuple[int, OrderedDict[str, torch.Tensor], Dict]:
        print('Exporting checkpoint ...')
        return self.epoch, self._model.state_dict(), self._optimizer.state_dict()


def full_train(config: Dict[str, Any], checkpoint_dir: Optional[Path] = None) -> None:
    print('Generating new trial ...')
    base_config: Config = config.pop('base_config')
    root_path: Path = config.pop('root_path')
    new_config: Config = replace(base_config, **config)

    print(f'Creating trainer ...')
    trainer = PerDeviceTrainer(new_config)

    if checkpoint_dir is not None:
        print(f'Loading existing checkpoint ...')
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        epoch: int = torch.load(checkpoint_dir / 'epoch.pt')
        model_checkpoint: OrderedDict[str, torch.Tensor] = torch.load(checkpoint_dir / 'model.pt')
        optimizer_checkpoint: Dict = torch.load(checkpoint_dir / 'optimizer.pt')

        print(f'Preparing trainer ...')
        trainer.prepare(root_path, epoch=epoch, model_checkpoint=model_checkpoint,
                        optimizer_checkpoint=optimizer_checkpoint)
    else:
        print(f'Preparing trainer ...')
        trainer.prepare(root_path)

    while trainer.epoch < new_config.num_epochs:
        train_loss: float = trainer.train()
        dev_loss: float
        dev_top1_accuracy: float
        dev_top5_accuracy: float
        dev_loss, dev_top1_accuracy, dev_top5_accuracy = trainer.test(True)

        if trainer.epoch == new_config.num_epochs:
            test_loss: float
            test_top1_accuracy: float
            test_top5_accuracy: float
            test_loss, test_top1_accuracy, test_top5_accuracy = trainer.test(False)

        if trainer.epoch % new_config.perturbation_interval == 0:
            epoch, model_checkpoint, optimizer_checkpoint = trainer.checkpoint()

            with tune.checkpoint_dir(trainer.epoch) as checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                torch.save(epoch, checkpoint_dir / 'epoch.pt')
                torch.save(model_checkpoint, checkpoint_dir / 'model.pt')
                torch.save(optimizer_checkpoint, checkpoint_dir / 'optimizer.pt')

        report: Dict[str, Union[int, float]] = {
            'epoch': trainer.epoch,
            'random_resize_bound': trainer.config.random_resize_bound,
            'random_erasing_p': trainer.config.random_erasing_p,
            'drop_path_rate': trainer.config.drop_path_rate,
            'lr': trainer.config.lr,
            'weight_decay': trainer.config.weight_decay,
            'train_batch_size': trainer.config.train_batch_size,
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'dev_top1_accuracy': dev_top1_accuracy,
            'dev_top5_accuracy': dev_top5_accuracy
        }

        if trainer.epoch == new_config.num_epochs:
            report.update({
                'test_loss': test_loss,
                'test_top1_accuracy': test_top1_accuracy,
                'test_top5_accuracy': test_top5_accuracy
            })

        tune.report(**report)


def train(root_path: Path, config: Config) -> str:
    print('Initializing dataset ...')
    fix_seed(config)
    init_datasets(root_path, config)

    log_root_path: Path = get_path(root_path, 'log')
    clean_path(log_root_path / config.model_name)

    scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=config.perturbation_interval,
        hyperparam_mutations={
            'random_resize_bound': tune.uniform(0.1, 0.9),
            'random_erasing_p': tune.uniform(0, 0.5),
            'drop_path_rate': tune.uniform(0, 0.3),
            'lr': tune.loguniform(1e-6, 1e-3),
            'weight_decay': tune.loguniform(1e-5, 1e-2),
            'train_batch_size': [8, 16] if config.use_subset else [32, 64, 128, 256, 384]
        }
    )

    reporter = CLIReporter(
        metric_columns={
            'epoch': 'EP',
            'train_loss': 'TL',
            'dev_loss': 'DL',
            'dev_top1_accuracy': 'DA1',
            'dev_top5_accuracy': 'DA5'
        },
        parameter_columns={
            'random_resize_bound': 'RR',
            'random_erasing_p': 'RE',
            'drop_path_rate': 'SP',
            'lr': 'LR',
            'weight_decay': 'WD',
            'train_batch_size': 'BS'
        },
        max_report_frequency=60,
        sort_by_metric=True
    )

    analysis: tune.ExperimentAnalysis = tune.run(
        full_train,
        metric='dev_top1_accuracy',
        mode='max',
        name=config.model_name,
        config={
            'base_config': config,
            'root_path': root_path,
            'random_resize_bound': tune.uniform(0.1, 0.9),
            'random_erasing_p': tune.uniform(0, 0.5),
            'drop_path_rate': tune.uniform(0, 0.3),
            'lr': tune.loguniform(1e-6, 1e-3),
            'weight_decay': tune.loguniform(1e-5, 1e-2),
            'train_batch_size': tune.choice([8, 16] if config.use_subset else
                                            [32, 64, 128, 256, 384])
        },
        resources_per_trial={'cpu': max(config.num_cpus // max(config.num_gpus, 1), 1),
                             'gpu': min(config.num_gpus, 1)},
        num_samples=config.num_samples,
        local_dir=log_root_path.as_posix(),
        scheduler=scheduler,
        keep_checkpoints_num=4,
        checkpoint_score_attr='dev_top1_accuracy',
        progress_reporter=reporter,
        trial_name_creator=lambda x: f'{config.model_name}_{x.trial_id}',
        trial_dirname_creator=lambda x: f'{config.model_name}_{x.trial_id}',
        resume='AUTO',
        reuse_actors=True
    )

    return analysis.best_logdir
