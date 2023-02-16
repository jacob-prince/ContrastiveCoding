#!/bin/bash

envname=$1

conda deactivate
conda create --clone ffcv --name $envname

conda activate $envname

# install pycortex and sklearn
pip install pycortex==1.2.6
pip install numpy==1.23.5
pip install numba==0.56.4
pip install scipy==1.10.0

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

# install this project package
pip install --user -e ../

# test imports
python3 env_test_imports.py

# set directory of the pycortex database
python3 pycortex_database_setup.py
