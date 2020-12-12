
import pandas as pd
from config import *
from src.utils import mkdirs_safe
from src.utils import print_fl
from src.datasets import read_orfs_data
from src.timer import Timer
from src.reference_data import (read_park_TSS_PAS,
                                read_brogaard_nucleosomes,
                                read_macisaac_abf1_sites,
                                read_sgd_orfs)

# global timer
timer = Timer()

# global inputs
rna_seq = None
all_mnase_data = None
mnase_coverage = None

all_orfs = read_sgd_orfs()
half_lives = read_orfs_data('data/half_life.csv')
TSSs = read_park_TSS_PAS()
orfs = all_orfs.join(TSSs[['TSS', 'PAS']])

if DEBUG:
    orfs = orfs[orfs.chr.isin(DEBUG_CHROMS)]

def read_input_data():

    global rna_seq
    global all_mnase_data

    print_fl("Reading RNA-seq...", end='')
    rna_seq = pd.read_hdf(rna_seq_path, 'rna_seq_data')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Reading MNase-seq...", end='')
    all_mnase_data = pd.read_hdf(mnase_seq_path, 
        'mnase_data')
    print_fl("Done.")
    timer.print_time()
    print_fl()


def determine_transcript_boundaries():

    print_fl("Reading RNA-seq pileup...", end='')
    rna_seq_pileup = pd.read_hdf(pileup_path, 'pileup')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    from src.transcript_boundaries import compute_boundaries, load_park_boundaries

    park_boundaries = load_park_boundaries()

    mkdirs_safe([anti_chrom_dir, sense_chrom_dir])

    # ------------- Antisense transcript boundaries ------------------

    print_fl("Determining antisense boundaries...", end='')
    antisense_boundaries = compute_boundaries(park_boundaries, rna_seq_pileup,
        save_dir=anti_chrom_dir, pileup_path=pileup_path,
        find_antisense=True, log=True, timer=timer)

    # all to compare with Park
    path = '%s/antisense_boundaries_computed_all.csv' % rna_dir
    antisense_boundaries.to_csv(path)

    # antisense paper data set
    path = '%s/antisense_boundaries_computed.csv' % rna_dir
    antisense_boundaries = antisense_boundaries[['TSS', 'strand', 
        'start', 'stop']].join(paper_orfs[['name', 'chr', 'orf_class']],
        how='inner')
    antisense_boundaries.to_csv(antisense_orfs_path)

    print_fl("Done.")
    print_fl("Wrote to %s" % path)
    timer.print_time()
    print_fl()

    # ------------- Sense transcript boundaries ------------------

    # compute sense boundaries
    # TODO: currently unused, only for check against Park boundaries
    print_fl("Determining sense boundaries...", end='')
    sense_boundaries = compute_boundaries(park_boundaries, rna_seq_pileup,
        save_dir=sense_chrom_dir, pileup_path=pileup_path,
        find_antisense=False, log=True, timer=timer)

    # all to compare with Park
    path = '%s/sense_boundaries_computed_all.csv' % rna_dir
    sense_boundaries.to_csv(path)

    # paper data set
    path = '%s/sense_boundaries_computed.csv' % rna_dir
    sense_boundaries.join(paper_orfs[[]], how='inner').to_csv(path)

    print_fl("Done.")
    print_fl("Wrote to %s" % path)
    timer.print_time()
    print_fl()

def compute_TPM_and_rate():

    from src.transcription import calculate_reads_TPM
    from src.transformations import log2_fold_change
    from src.reference_data import read_sgd_orf_introns

    # exclude introns in read counts on sense strand
    if not INCLUDE_SENSE_INTRONS:
        CDS_introns = read_sgd_orf_introns()
    else:
        CDS_introns = None

    # ------- Compute TPM --------------

    # Antisense

    # compute read counts and TPM and save to disk as csv
    print_fl("Calculating antisense TPM...", end='')
    anti_read_counts, anti_TPM, anti_RPK = calculate_reads_TPM(all_orfs, rna_seq, antisense=True)
    anti_read_counts.to_csv('%s/antisense_read_counts.csv' % rna_dir)
    anti_TPM.to_csv('%s/antisense_TPM.csv' % rna_dir)

    # Compute TPM logfold change
    antisense_TPM_log2fold = log2_fold_change(anti_TPM, pseudo_count=1)
    antisense_TPM_log2fold.to_csv('%s/antisense_TPM_log2fold.csv' % rna_dir)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # compute read counts and TPM and save to disk as csv
    print_fl("Calculating sense TPM...", end='')
    read_counts, TPM, RPK = calculate_reads_TPM(all_orfs, rna_seq, 
        include_introns=INCLUDE_SENSE_INTRONS, CDS_introns=CDS_introns)
    read_counts.to_csv('%s/sense_read_counts.csv' % rna_dir)
    TPM.to_csv('%s/sense_TPM.csv' % rna_dir)

    # Compute TPM logfold change
    sense_TPM_log2fold = log2_fold_change(TPM, pseudo_count=1)
    sense_TPM_log2fold.to_csv('%s/sense_TPM_log2fold.csv' % rna_dir)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # -------- Compute TPM/min -------------

    from src.transcription import calculate_xrates
    from src import transformations

    print_fl("Calculating TPM rate...", end='')

    # calculate the transcription rates for each ORF
    orf_xrates = calculate_xrates(TPM, half_lives)

    # Set transcription rates <0 to 1, for fold-change calculation
    times = [0.0, 7.5, 15.0, 30.0, 60.0, 120.0]
    print_fl("Truncating %d genes which have a transcription rate less than 0"
             " at some time" % 
             len(orf_xrates[orf_xrates[times].min(axis=1) < 0]))
    orf_xrates_g0 = orf_xrates[times].copy()
    for time in times:
        orf_xrates_g0.loc[orf_xrates[time] <= 0, time] = 1

    # fold change and log transform
    orf_xrates_log2_fold_change = transformations\
        .log2_fold_change(orf_xrates_g0, pseudo_count=0.01)

    # Write to disk
    orf_xrates_g0.to_csv('%s/orf_xrates.csv' % rna_dir)
    orf_xrates_log2_fold_change.to_csv('%s/orf_xrates_log2fold.csv' % rna_dir)

    print_fl("Done.")
    timer.print_time()
    print_fl()

def compute_mnase_seq_coverage():
    from src.occupancy import calculate_coverage

    global mnase_coverage

    print_fl("Calculating MNase-seq coverage...", end='')
    mnase_coverage = calculate_coverage(all_mnase_data, orfs, window=2000)
    mnase_coverage.to_csv('%s/coverage_2000.csv' % mnase_dir)
    print_fl("Done.")
    timer.print_time()
    print_fl()


def compute_orfs_set():

    global paper_orfs

    print_fl("Determining paper ORFs set...")
    from src.orfs import determine_paper_set
    paper_orfs = determine_paper_set(orfs, TSSs, half_lives, mnase_coverage)
    paper_orfs.TSS = paper_orfs.TSS.astype(int)
    paper_orfs.to_csv('%s/orfs_cd_paper_dataset.csv' % OUTPUT_DIR)
    print_fl("Done.")
    timer.print_time()
    print_fl()

def calculate_kernels():

    brogaard = read_brogaard_nucleosomes()
    brogaard = brogaard.sort_values('NCP_score/noise_ratio', ascending=False)\
        .reset_index(drop=True)
    abf1_sites = read_macisaac_abf1_sites()

    if DEBUG:
        brogaard = brogaard[brogaard['chromosome'].isin(DEBUG_CHROMS)]
        abf1_sites = abf1_sites[abf1_sites['chr'].isin(DEBUG_CHROMS)]

    from src.kernel_fitter import compute_nuc_kernel, compute_sm_kernel, compute_triple_kernel

    print_fl("Computing nucleosome kernel...", end='')
    nuc_kernel = compute_nuc_kernel(all_mnase_data, brogaard)
    nuc_kernel.save_kernel('%s/nucleosome_kernel.json' % mnase_dir)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Computing small fragments kernel...", end='')
    sm_kernel = compute_sm_kernel(all_mnase_data, abf1_sites)
    sm_kernel.save_kernel('%s/small_kernel.json' % mnase_dir)
    print_fl("Done.")
    timer.print_time()
    print_fl()


def main():
    """
    Processing of RNA-seq and MNase-seq for analyses

    1. Compute antisense boundaries
    2. Compute measures of TPM and TPM rate
    3. Compute MNase-seq coverage 
    4. Curate set of ORFs for analyses
    5. Compute cross correlation kernels for chromatin metrics

    Inputs:
    - RNA-seq data frame
    - RNA-seq pileup data frame
    - MNase-seq data frame
    - Canonical ORFs
    - TSS/PAS annotations
    - Half life annotations
    - Brogaard nucleosome positions
    - MacIsaac Abf1 positions

    Output:

    - Antisense transcript TSS/TESs dataframe
    - Sense transcript TSS/TESs dataframe

    - TPM data frame
    - TPM/rate data frame

    - MNase-seq coverage data frame

    - Set of ORFs used in paper

    - Nucleosome cross correlation kernel
    - Small factors cross correlation kernel

    """

    print_fl("*******************************")
    print_fl("* 2      Preprocessing        *")
    print_fl("*******************************")

    print_preamble()

    print_fl("\n------- Read inputs ----------\n")
    read_input_data()

    print_fl("\n------- MNase-seq coverage ----------\n")
    compute_mnase_seq_coverage()

    print_fl("\n------ Paper ORFs set ---------\n")
    compute_orfs_set()

    print_fl("\n------- TPM and TPM rate ----------\n")    
    compute_TPM_and_rate() 

    print_fl("\n------- Transcript boundaries ----------\n")
    determine_transcript_boundaries()

    print_fl("\n--------- Cross correlation kernels ------------\n")
    calculate_kernels()


if __name__ == '__main__':
    main()
