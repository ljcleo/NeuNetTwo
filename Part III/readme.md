# NeuNetTwo --- Part III

Neural Network &amp; Deep Learning Final Project (Part III)

## Introduction

Image classification models for the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), written in Python 3 using PyTorch and `timm`. Two types of model --- ResNeXt and ViT --- are supported.

## Requirements

- Python >= 3.9
- PyTorch >= 1.11
- PyTorch Image Models (`timm`) >= 0.5.4
- Ray Tune
- Tensorboard
- pyyaml
- tqdm
- torchinfo (for model structure summary)

## Usage

1. Run `python check_model.py` to display the structure summary of all available models (`torchinfo` package required). Among all model types, `resnext` denotes ResNeXt-29, `vit` denotes ViT-13, and `vit2` denotes ViT-6.

2. Run `python train_model.py [model]` to train models using the configuration specified by `model`. This can be `resnext`, `vit`, `vit2` or other custom configuration if available in the `config` directory. All checkpoints and logs are stored in the `log/[model]` directory, among which parameters of the best model are copied to the `model` directory.

   Best models trained by PBT can be downloaded [here](https://drive.google.com/drive/folders/12-_e5F52pPr9wQ5wcbIVUXIRrAcpUcAC?usp=sharing). Just put the dumped models into the `model` directory.

3. Run `python test_model.py` to evaluate loss and accuracy of all trained models on both development and test sets. Results are written to `out/[model].csv` for each model. Make sure that all three model files are put in the `model` directory before running this script.

## Project Structure

```{plain}
Part III
├── config
│   ├── default.yaml  # Default configuration
│   ├── resnext.yaml  # ResNeXt-29
│   ├── vit.yaml      # ViT-13
│   └── vit2.yaml     # ViT-6
├── data
├── log               # Training checkpoints and logs
├── model             # Best models
├── out               # Test results
├── src
│   ├── __init__.py
│   ├── command.py    # Command line parser
│   ├── config.py     # Config loader
│   ├── data.py       # Dataset and loader generator
│   ├── gpu.py        # GPU initialization
│   ├── init.py       # Random seed initialization
│   ├── model.py      # Model definition
│   ├── train.py      # Trainer class and training methods
│   ├── util.py       # Utility functions
├── check_model.py    # Model summary script
├── train_model.py    # Model training script
├── test_model.py     # Model evaluation script
└── readme.md         # This file
```

## Author

Jingcong Liang, [18307110286](mailto:18307110286@fudan.edu.cn)
