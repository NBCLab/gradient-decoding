#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/conda_env pip python=3.9 -y

conda config --append envs_dirs /home/data/nbc/misc-projects/Peraza_GradientDecoding/env

source activate /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/conda_env
pip install black flake8 isort numpy nibabel brainspace neuromaps surfplot scikit-learn ipykernel \
            biopython git+https://github.com/JulioAPeraza/NiMARE.git@gradient-decoding