#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/conda_env pip python=3.9 -y

source activate /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/conda_env
pip install black flake8 isort numpy nibabel brainspace neuromaps surfplot