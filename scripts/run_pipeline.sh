#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 81920

cd $SLURM_WORKING_DIR

echo "batch: Starting job on $(date)"

# point to conda path
. $CONDA_PATH

# activate environment
conda activate $$CONDA_ENV

#python 1_data_initialize.py &&
#echo $(date) &&

python 2_data_preprocessing.py
echo $(date)

#python 3_chrom_metrics.py
#echo $(date)

#python 4_analysis.py
#echo $(date)

#python 5_figures.py
#echo $(date)

echo "batch: Completed job on $(date)"

