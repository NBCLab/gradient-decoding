#!/bin/bash
#SBATCH --job-name=workflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=40gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
# ------------------------------------------

# IB_40C_512G, for running workflow
# investor, for testing
pwd; hostname; date

#==============Shell script==============#
# Load evironment
source /home/data/nbc/misc-projects/Peraza_GradientDecoding/env/activate_env
set -e

PROJECT_DIR="/home/data/nbc/misc-projects/Peraza_GradientDecoding"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/workflow.py \
    --project_dir ${PROJECT_DIR} \
    --n_cores ${SLURM_CPUS_PER_TASK}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date