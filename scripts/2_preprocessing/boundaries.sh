#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 81920

cd $SLURM_WORKING_DIR

# point to conda path
. $CONDA_PATH
 
# activate environment
conda activate $CONDA_ENV

echo "batch: Starting job at $(date)"

echo "Transcript boundaries chromosome $CHROM, antisense=$ANTISENSE"
python src/transcript_boundaries.py $CHROM $ANTISENSE

echo "batch: Completed job at $(date)"
