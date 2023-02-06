#!/bin/bash

envname=dnffa

# prepare
conda deactivate

# create env from .yml file
conda env create -n $envname --file environment.yml

# activate env
conda activate $envname

# download this project package
pip install --user -e .

# test imports
python3 env_test_imports.py

