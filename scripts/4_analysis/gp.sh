#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 81920


cd $SLURM_WORKING_DIR

# point to conda path
. $CONDA_PATH
   
conda activate $CONDA_ENV

echo "batch: Starting job at $(date)"

python src/gp_models.py $MODEL True

echo "batch: Completed job at $(date)"

