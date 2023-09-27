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

# Installation (NOT WORKING YET)

1. Clone the repository and its submodule `datasets`
```bash
git clone --recurse-submodules https://github.com/S1M0N38/master-thesis-code.git
```
2. Enter the repository
```bash
cd master-thesis-code
```
3. Create a folder or symbolic link to store experiments, i.e. training results
```bash
mkdir experiments # or ln -s /path/to/experiments experiments
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
```bash
/bin/bash -c "$(curl -fsSL https://s1m0n38.xyz/scripts/master-thesis-download-dataset.sh)"
```
8. Create symbolic links to the datasets
```bash
# Symbolic to CIFAR100
ls -s path/to/cifar-100-python datasets/CIFAR100/inputs/cifar-100-python

# Symbolic to iNaturalist19
# ln -s path/to/iNaturalist19/train datasets/iNaturalist19/inputs/train
# ln -s path/to/iNaturalist19/val   datasets/iNaturalist19/inputs/val
# ln -s path/to/iNaturalist19/test  datasets/iNaturalist19/inputs/test

# Symbolic to tieredImageNet
# ln -s path/to/tieredImageNet/train datasets/tieredImageNet/inputs/train
# ln -s path/to/tieredImageNet/val   datasets/tieredImageNet/inputs/val
# ln -s path/to/tieredImageNet/test  datasets/tieredImageNet/inputs/test
```
