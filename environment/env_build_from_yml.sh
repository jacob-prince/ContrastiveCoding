#!/bin/bash

envname=dnffa_test_2

# prepare
conda deactivate

# create env from .yml file
conda env create -n $envname --file environment.yml

# activate env
conda activate $envname

# test imports
python3 env_test_imports.py


