from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter

from src.config import Config
from src.util import clean_path, get_path


def init_tb_writer(root_path: Path, config: Config) -> SummaryWriter:
    writer_path: Path = get_path(root_path, 'img', config.model_name)
    clean_path(writer_path)
    return SummaryWriter(writer_path)
