#!/bin/bash

conda env export > environment.yml

git add environment.yml
git commit -m "updated environment.yml"
git push origin main

