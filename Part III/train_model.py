def main() -> None:
    from pathlib import Path
    root_path = Path('.').absolute()

    from src.command import CommandLineArgs, parse_arg
    args: CommandLineArgs = parse_arg()

    from src.config import Config, init_config
    config: Config = init_config(root_path, args)

    from src.train import train
    best_dir: str = train(root_path, config)
    print(f'Best results stored in {best_dir}.')

    from shutil import copy
    from src.util import get_path
    copy(Path(best_dir) / 'checkpoint_000200' / 'model.pt',
         get_path(root_path, 'model') / f'{config.model_name}.pt')
    print('Best model saved.')

if __name__ == '__main__':
    main()
