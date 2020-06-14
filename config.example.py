

import sys
sys.path.append('.')

# ------ Configuration --------------

# Use a subset of the data for debugging the pipeline
DEBUG = False
DEBUG_CHROMS = [1] # chromosomes to process on when in debug mode

OUTPUT_DIR = 'output'

# Split pieces of analysis across slurm jobs
# driving script will wait for child jobs to finish before continuing
USE_SLURM = False

# paths for running scripts on SLURM
SLURM_WORKING_DIR = "/path/to/working/dir"
CONDA_PATH = "/path/to/conda.sh"
CONDA_ENV = "conda_env_name"

# Directory to watch for slurm child jobs to complete
WATCH_TMP_DIR = '%s/watch_tmp' % OUTPUT_DIR

# Directories for BAM files
BAM_DIR = 'data/bam/'

# Filepath for logging
LOG_FILE_PATH = '%s/log.txt' % OUTPUT_DIR

# Global definitions
times = [0, 7.5, 15, 30, 60, 120]

# Fixed global parameters
from src.utils import print_fl

def print_preamble():

    if DEBUG: print_fl("\n*** DEBUG MODE, using chromosome(s): %s ****\n"
        % ', '.join([str(c) for c in DEBUG_CHROMS]))
    print_fl("Working with output directory %s" % OUTPUT_DIR)

# -------- Directories ---------

rna_dir = '%s/rna_seq' % OUTPUT_DIR
mnase_dir = '%s/mnase_seq' % OUTPUT_DIR
misc_figures_dir = '%s/other/' % OUTPUT_DIR

# --------- Data paths -------------

pileup_path = '%s/rna_seq_pileup_dm538_543.h5.z' % rna_dir
mnase_seq_path = '%s/mnase_seq_merged_dm498_509_sampled.h5.z' % mnase_dir
rna_seq_path = '%s/rna_seq_dm538_543.h5.z' % rna_dir

# --------- Preprocessing paths -------------

pileup_chrom_dir = '%s/pileup_chr/' % rna_dir

anti_chrom_dir = '%s/antisense_boundaries_chrom/' % rna_dir
sense_chrom_dir = '%s/sense_boundaries_chrom/' % rna_dir
cc_sense_chrom_dir = '%s/sense_cross_correlations_chrom/' % mnase_dir
cc_antisense_chrom_dir = '%s/antisense_cross_correlations_chrom/' % mnase_dir

nuc_kernel_path = '%s/nucleosome_kernel.json' % mnase_dir
sm_kernel_path = '%s/small_kernel.json' % mnase_dir

# nucleosomes chrom save directory
sense_nuc_chrom_dir = '%s/nucleosomes_chrom_sense/' % (mnase_dir)
anti_nuc_chrom_dir = '%s/nucleosomes_chrom_antisense/' % (mnase_dir)

# entropy
sense_entropy_dir = '%s/entropies_sense/' % (mnase_dir)
anti_entropy_dir = '%s/entropies_antisense/' % (mnase_dir)

# cross correlation paths
cross_corr_sense_path = '%s/cross_correlation_sense.h5.z' % (mnase_dir)
cross_corr_antisense_path = '%s/cross_correlation_antisense.h5.z' % (mnase_dir)

# ---------- Analysis ---------------

gp_dir = '%s/gp/' % OUTPUT_DIR

p1_sense_path = '%s/p1_sense.csv' % mnase_dir
p2_sense_path = '%s/p2_sense.csv' % mnase_dir
p3_sense_path = '%s/p3_sense.csv' % mnase_dir

p1_antisense_path = '%s/p1_antisense.csv' % mnase_dir
p2_antisense_path = '%s/p2_antisense.csv' % mnase_dir
p3_antisense_path = '%s/p3_antisense.csv' % mnase_dir

# --------- Global data -----------------

from src.datasets import read_orfs_data
import os

# load paper orfs if possible
paper_orfs_path = "%s/orfs_cd_paper_dataset.csv" % OUTPUT_DIR
paper_orfs = None
if os.path.exists(paper_orfs_path):
    paper_orfs = read_orfs_data(paper_orfs_path)

# load antisense orfs if possible
antisense_orfs_path = '%s/antisense_boundaries_computed.csv' % rna_dir
antisense_orfs = None
if os.path.exists(antisense_orfs_path):
    antisense_orfs = read_orfs_data(antisense_orfs_path)

