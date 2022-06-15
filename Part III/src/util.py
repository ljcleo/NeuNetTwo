from pathlib import Path
from typing import Optional


def get_path(root_path: Path, sub_dir: str, name: Optional[str] = None) -> Path:
    result: Path = root_path / sub_dir
    if name is not None:
        result = result / name.split('-')[0]

    result.mkdir(parents=True, exist_ok=True)
    return result


def clean_path(path: Path) -> None:
    if path.exists():
        for sub_path in path.iterdir():
            if sub_path.is_dir():
                clean_path(sub_path)
                sub_path.rmdir()
            else:
                sub_path.unlink()
