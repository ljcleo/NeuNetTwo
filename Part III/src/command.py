from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class CommandLineArgs:
    model: str
    use_subset: bool
    debug_level: int
    max_cpu_num: int
    max_gpu_num: int
    visible_gpu: str
    min_gpu_mem: int


def parse_arg(args: Optional[Sequence[str]] = None) -> CommandLineArgs:
    parser = ArgumentParser(description='Train image classification model on CIFAR-100.')
    parser.add_argument('model', help='model name')

    parser.add_argument('-d', '--debug-level', default=0, type=int, choices=(0, 1, 2),
                        help='debug level')

    parser.add_argument('-p', '--use-subset', action='store_true', help='use subset only')
    parser.add_argument('-g', '--visible-gpu', nargs=1, help='candidate GPU ID')

    parser.add_argument('-c', '--max-cpu-num', default=0x7fffffff, type=int,
                        help='maximum number of CPU')

    parser.add_argument('-n', '--max-gpu-num', default=0x7fffffff, type=int,
                        help='maximum number of GPU (to use CPU set this to 0)')

    parser.add_argument('-m', '--min-gpu-mem', default=10240, type=int,
                        help='minimum free memory per device')


    return parser.parse_args(args, namespace=CommandLineArgs)
