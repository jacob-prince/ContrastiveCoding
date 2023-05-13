#!/bin/bash

envname=$1

pip install --upgrade pip

conda deactivate

mamba create -y -n $envname python=3.9 cupy=11.6 pkg-config compilers libjpeg-turbo opencv pytorch-gpu cudatoolkit=11.7 torchvision torchaudio pytorch-cuda=11.7 numba -c pytorch -c nvidia -c conda-forge

conda activate $envname

# install ffcv
pip install ffcv

# install pycortex
pip install pycortex
pip install numpy==1.23.5
pip install scipy==1.10.1
pip install numba==0.56.4

# packages for testing environment in jupyterlab
mamba install ipykernel ipython jupyterlab ipywidgets

# add kernel to jupyterlab
ipython kernel install --user --name=$envname

# some packages for github/jupyter integration
mamba install GitPython nbstripout nbconvert
nbstripout --install

# wandb integration
mamba install wandb
wandb login # will need to manually paste in key

# for dealing with COCO dataset
mamba install pycocotools

# for running torchlens
pip install graphviz
pip install git+https://github.com/johnmarktaylor91/torchlens

# other packages for analyses
mamba install torchmetrics seaborn nibabel h5py

# install this project package
pip install --user -e ../

# set directory of the pycortex database
python3 pycortex_database_setup.py

# install circuit pruning tools
mamba install umap-learn=0.5.3 dash=2.7.1 jupyter-dash=0.4.2 pyarrow=10.0.1
pip install lucent==0.1.0 kornia==0.4.1 kaleido==0.2.1

pip install --user -e /home/jovyan/work/DropboxSandbox/circuit_pruner_iccv2023/circuit_pruner_code

# test imports
python3 env_test_imports.py
