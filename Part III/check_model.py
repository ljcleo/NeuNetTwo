def main(model_name) -> None:
    from pathlib import Path
    root_path = Path('.').absolute()

    from src.command import CommandLineArgs, parse_arg
    args: CommandLineArgs = parse_arg([model_name])

    from src.config import Config, init_config
    config: Config = init_config(root_path, args)

    from src.model import ImageClsModel
    model = ImageClsModel(config)

    from torchinfo import summary
    summary(model, input_size=(config.train_batch_size, 3, 32, 32), depth=2,
            device='cpu' if config.num_gpus == 0 else 'cuda')

if __name__ == '__main__':
    main('resnext')
    main('vit')
    main('vit2')
