#!/bin/bash

# Create a new Conda environment
conda env create -f test_new.yml --name env_test

# Activate the Conda environment
source activate env_test