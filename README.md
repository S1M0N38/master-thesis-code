<div align="center">
  <h1>Master Thesis Code</h1>
  <p><em>Build with <a href="https://github.com/S1M0N38/pytorch-template">[🔥]</a></em></p>

  <a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&amp;logo=python">
  </a>
  <a href="https://github.com/S1M0N38/master-thesis-code/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/S1M0N38/master-thesis-code?style=for-the-badge&amp;color=ff69b4">
  </a>
  <a href="https://discord.com/users/S1M0N38#0317">
    <img alt="Discord" src="https://img.shields.io/static/v1?label=DISCORD&amp;message=DM&amp;color=blueviolet&amp;style=for-the-badge">
  </a>
</div>

-------------------------------------------------------------------------------

This repository contains the code used to produce results for [my master
thesis](https://github.com/S1M0N38/master-thesis). I find useful to have access
to the code to check the implementation describe by scientific papers, so here
it is.

## Installation

1. Clone the repository and its submodule `datasets` [^1]
```bash
git clone --recurse-submodules https://github.com/S1M0N38/master-thesis-code.git
```
2. Enter the repository
```bash
cd master-thesis-code
```
3. Create a folder or symbolic link to store experiments, i.e. training results
```bash
mkdir experiments
# This folder will become heavy by storing training results (checkpoints,
# models' outputs, etc.) so you can create where you have enough space and then
# just create a symbolic link to it:
# ln -s /path/to/experiments experiments
```
4. Create a virtual environment with python 3.10 (check with `python -V`)
```bash
python -m venv .venv
```
5. Activate the virtual environment
```bash
source .venv/bin/activate
```
6. Install the requirements
```bash
python -m pip install -r requirements.txt
```
7. Download the datasets
<!-- ```bash -->
<!-- /bin/bash -c "$(curl -fsSL https://S1M0N38.xyz/scripts/master-thesis-datasets-download.sh)" -->
<!-- ``` -->
8. Create symbolic links to the datasets
```bash
# Symbolic to CIFAR100
ln -s path/to/cifar-100-python datasets/datasets/CIFAR100/inputs/cifar-100-python

# Symbolic to iNaturalist19
# ln -s path/to/iNaturalist19/train datasets/datasets/iNaturalist19/inputs/train
# ln -s path/to/iNaturalist19/val   datasets/datasets/iNaturalist19/inputs/val
# ln -s path/to/iNaturalist19/test  datasets/datasets/iNaturalist19/inputs/test

# Symbolic to tieredImageNet
# ln -s path/to/tieredImageNet/train datasets/datasets/tieredImageNet/inputs/train
# ln -s path/to/tieredImageNet/val   datasets/datasets/tieredImageNet/inputs/val
# ln -s path/to/tieredImageNet/test  datasets/datasets/tieredImageNet/inputs/test
```

[^1]: Alternatively, you can git clone
`https://github.com/S1M0N38/master-thesis-datasets` and create symbolic link


## Usage

The entire pipeline consists of the following steps:
1. Train the model
2. Test the model
3. Evaluate testing results
4. Visualize results

### 1. Train

<!-- TODO: add GPU model -->
Training step require at least one GPU (mine was ...) because do it on CPU
it's unbearably slow.
Assuming that you have activated the virtual environment you can train a model
using a configuration file with:

```bash
python "train.py" "configs/CIFAR100/xe-onehot.toml"
# This train a EfficientNetB0 using CrossEntropy and onehot encoding
# on CIFAR100 dataset. Use other .toml file in configs or define your own
```

Everything about training is define in the `TOML` configuration file, whose
key/values are used to dynamically initialize model, dataloaders, metrics, etc.
(This project is based on [[🔥](https://github.com/S1M0N38/pytorch-template)]
template, so take a look at that to understand how it works under the hood)

If training successfully started, a new directory is created inside
`experiments/CIFAR100` with the following naming scheme:
```
{MONTHDAY}_{HOURMINUTE}_{CONFIGHASH}_{NAME}
```
- The first part it's contains date and time so it's easy to sort various
experiments by creation time.
- `CONFIGHASH` is the hash of the configuration file so it's easy to quickly
group different experiments with exactly the same configuration.
- `NAME` is the name of the experiment define in the TOML file with the key
`name`.

For example

```
0707_1458_8bc6fb3e_xe-onehot
├── checkpoints
│  └── ...
├── runs
│  └── events.out.tfevents.1688741895.hostname.localhost.3233031.0
├── config.toml
└── trainer.log
```

where `config.toml` contains a copy of the configuration file specify
in the previous command.

You can track the training progress by
- following the log file: `tail -f experiments/CIFAR100/*/trainer.log `
- using [TensorBoard](https://www.tensorflow.org/tensorboard): `tensorboard --logdir experiments/CIFAR100/`

Model's checkpoints (model graph and weights) will be save inside `checkpoints`.
In the next step these checkpoints will be used to load the trained model in
memory.

### 2. Test

After trained the model we want to test it, i.e. run the test dataset through
the model and store results. For testing you still need GPU.
The testing script will:
- Save model output
- Save features extracted from penultimate level
- Perform [FGSM](https://arxiv.org/abs/1412.6572#) attack (targeted or untargeted)
- Save model output and features produced by the adversarial inputs

```bash
python "test.py" "configs/CIFAR100/xe-onehot.toml" --epsilon 0.001
```

This will search for all experiments in `experiments` that were trained using
`configs/CIFAR100/xe-onehot.toml` as configuration file and invite the user to
choose one. Then it will ask for the target of adversarial attack (suppose we
choose `apple` as target).

After testing the experiment folder should contains a new directory named
`results`.

```
0707_1458_8bc6fb3e_xe-onehot
├── results
│  ├── apple
│  │  ├── features-0.00100.npy
│  │  └── outputs-0.00100.npy
│  ├── features.npy
│  ├── outputs.npy
│  └── targets.npy
└── ...
```

- `targets.npy` is simply a numpy array containing the `y` of the test dataset
(in the case of onehot encoding its value are simply integer number).
- `outpus.npy` ans `features.npy` respectively contains the model outputs and
features obtained by feeding the model with the images from datasets.
- `{TARGET}/features-{EPSILON}.npy` and `{TARGET}/features-{EPSILON}.npy`
are the model's outputs and features in the case of the adversarial attack.
If the attack was untargeted, `{TARGET}` is `_`.
