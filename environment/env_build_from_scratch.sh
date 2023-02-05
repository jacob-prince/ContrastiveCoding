#!/bin/bash

envname=dnffa

# initialize
conda deactivate
pip install --upgrade pip

# # start with packages needed for ffcv
conda create -n $envname python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c conda-forge -c pytorch && conda activate $envname && conda update ffmpeg && pip install pyzmq scipy && pip install ffcv

# downgrade packages to make ffcv work
pip install numba==0.56.4
pip install pytorch-pfn-extras==0.5.8

# packages for testing environment in jupyterlab
pip install ipykernel ipython jupyterlab ipywidgets

# add kernel to jupyterlab
ipython kernel install --user --name=$envname

# download pycortex
pip install -U pycortex

# # some packages for github/jupyter integration
pip install GitPython
pip install nbstripout nbconvert
nbstripout --install

# wandb integration
pip install wandb
wandb login # will need to manually paste in key

# # download this project package
pip install --user -e .

# test imports
python3 test_imports.py


