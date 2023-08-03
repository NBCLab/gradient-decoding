#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/conda_env pip python=3.9 -y

conda config --append envs_dirs /home/data/nbc/misc-projects/Peraza_GradientDecoding/env

source activate /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/conda_env
pip install black flake8 isort 'gradec==0.0.1rc3' brainspace mapalign scikit-learn ipykernel \
    seaborn netneurotools git+https://github.com/amueller/word_cloud.git