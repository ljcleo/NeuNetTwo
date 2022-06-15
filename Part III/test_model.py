def main(model_name) -> None:
    from pathlib import Path
    from src.util import get_path
    root_path = Path('.').absolute()
    model_path: Path = get_path(root_path, 'model')

    from src.command import CommandLineArgs, parse_arg
    args: CommandLineArgs = parse_arg([model_name])

    from src.config import Config, init_config
    config: Config = init_config(root_path, args)

    from src.data import init_datasets
    init_datasets(root_path, config)

    import torch
    from typing import OrderedDict
    state_dict: OrderedDict[str, torch.Tensor] = torch.load(model_path / f'{config.model_name}.pt')

    from typing import Tuple
    from src.train import PerDeviceTrainer
    trainer = PerDeviceTrainer(config)
    trainer.prepare(root_path, model_checkpoint=state_dict)
    dev_eval: Tuple[float, float, float] = trainer.test(True)
    test_eval: Tuple[float, float, float] = trainer.test(False)

    from csv import writer

    with (get_path(root_path, 'result') /
          f'{config.model_name}.csv').open('w', encoding='utf8', newline='') as f:
        w = writer(f)
        w.writerow(('loss', 'top1_acc', 'top5_acc'))
        w.writerow(dev_eval)
        w.writerow(test_eval)

if __name__ == '__main__':
    main('resnext')
    main('vit')
    main('vit2')
