
import pandas as pd
from config import *
from src.utils import mkdirs_safe
from src.utils import print_fl
from src.datasets import read_orfs_data
from src.timer import Timer
from src import nucleosome_linkages
from src.kernel_fitter import compute_triple_kernel

# global timer
timer = Timer()


# global timer
timer = Timer()

# global inputs
all_mnase_data = None

def read_input_data():

    global all_mnase_data

    print_fl("Reading MNase-seq...", end='')
    all_mnase_data = pd.read_hdf('%s/mnase_seq_merged_dm498_509_sampled.h5.z'
        % mnase_dir, 
        'mnase_data')
    print_fl("Done.")
    timer.print_time()
    print_fl()


def compute_occupancies(strand='sense'):

    from src.occupancy import calculate_occupancies_all_chromosomes

    occ_orfs = paper_orfs

    # use antisense TSSs
    if strand == 'antisense': 
        occ_orfs = antisense_orfs

    print_fl("Calculating occupancies...", end='')
    occupancies = calculate_occupancies_all_chromosomes(all_mnase_data, 
        occ_orfs)
    occupancies.to_csv('%s/occupancies_%s.csv' % (mnase_dir, strand))
    print_fl("Done.")
    timer.print_time()
    print_fl()


def compute_cross_correlations(strand='sense'):

    from src.cross_correlation_kernel import MNaseSeqDensityKernel
    from src.cross_correlation import calculate_cross_correlation_all_chromosomes

    cc_orfs = paper_orfs
    cc_dir = cc_sense_chrom_dir
    cross_corr_path = cross_corr_sense_path
    if strand == 'antisense': 
        cc_orfs = antisense_orfs
        cc_dir = cc_antisense_chrom_dir
        cross_corr_path = cross_corr_antisense_path

    mkdirs_safe([cc_dir])

    nuc_kernel = MNaseSeqDensityKernel(filepath=nuc_kernel_path)
    sm_kernel = MNaseSeqDensityKernel(filepath=sm_kernel_path)
    triple_kernel = compute_triple_kernel(nuc_kernel)

    print_fl("Cross correlating %d ORFs..." % len(cc_orfs))
    
    cross, summary_cross = calculate_cross_correlation_all_chromosomes(
        all_mnase_data, cc_orfs, nuc_kernel, sm_kernel, triple_kernel,
        save_chrom_dir=cc_dir, timer=timer, log=True,
        find_antisense=(strand == 'antisense'))
    
    cross.to_hdf(cross_corr_path,
        'cross_correlation', mode='w', complevel=9, complib='zlib')
    summary_cross.to_csv('%s/cross_correlation_summary_%s.csv' % 
        (mnase_dir, strand))

    print_fl("Done.")
    timer.print_time()
    print_fl()


def compute_organization_measures(strand='sense'):

    from src.entropy import calculate_cc_summary_measure

    orfs = paper_orfs
    if strand == 'antisense':
        orfs = antisense_orfs

    mkdirs_safe([sense_entropy_dir, anti_entropy_dir]) 

    print_fl("Loading cross correlation")
    cross = pd.read_hdf('%s/cross_correlation_%s.h5.z' % 
        (mnase_dir, strand), 'cross_correlation')
    
    print_fl("Calculating entropy %d ORFS..." % len(orfs))
    entropies = calculate_cc_summary_measure(orfs, cross, strand,
        timer)
    entropies = entropies.round(3)
    entropies.to_csv('%s/orf_%s_entropies.csv' % (mnase_dir, strand))
    timer.print_time()
    print_fl()

def call_p123_nucleosomes(strand='sense'):

    from src.nucleosome_linkages import call_all_nucleosome_p123

    # relevant cross correlation directory

    p123_orfs = paper_orfs
    save_chrom_dir = sense_nuc_chrom_dir
    cc_dir = cc_sense_chrom_dir
    p1_path, p2_path, p3_path = (
        p1_sense_path,
        p2_sense_path,
        p3_sense_path
        )

    if strand == 'antisense': 
        p123_orfs = antisense_orfs
        save_chrom_dir = anti_nuc_chrom_dir
        cc_dir = cc_antisense_chrom_dir
        p1_path, p2_path, p3_path = (
            p1_antisense_path,
            p2_antisense_path,
            p3_antisense_path
            )

    mkdirs_safe([save_chrom_dir])

    print_fl("Calling +1, +2, and +3 nucleosomes...", end='\n')

    linkages, p123_orfs = call_all_nucleosome_p123(p123_orfs, 
        (strand=='antisense'), cc_dir, save_chrom_dir, timer)

    linkages.to_csv('%s/called_orf_nucleosomes_%s.csv' % (mnase_dir, strand))
    p123_orfs.to_csv('%s/called_orf_p123_nucleosomes_%s.csv' % (mnase_dir, strand))

    p1 = nucleosome_linkages.convert_to_pos_time_df(p123_orfs, linkages, '+1')
    p2 = nucleosome_linkages.convert_to_pos_time_df(p123_orfs, linkages, '+2')
    p3 = nucleosome_linkages.convert_to_pos_time_df(p123_orfs, linkages, '+3')

    p1.to_csv(p1_path)
    p2.to_csv(p2_path)
    p3.to_csv(p3_path)

    print_fl('Done.')
    timer.print_time()
    print_fl()


def main():
    """
    Computation of chromatin metrics against sense and antisense strand
    for analysis.

    1. * For the sense strand for each ORF
    2. Compute occupancies
    3. Compute cross correlation and save these per chromosome to disk
    4. Compute cross correlation summaries
    5. Call nucleosomes in cross correlation window
    6. Call +1, +2, and +3 nucleosomes relative to TSS
    7. * Repeat 2-6 for identified for antisense TSSs

    Inputs:
    - MNase-seq data
    - RNA-seq data
    - Antisense transcript boundaries

    Output:
    - MNase-seq occupancy summaries for each ORF
    - Per bp cross correlation scores for each ORF
    - Cross correlation summary scores for each ORF
    - Called nucleosomes local to each ORF
    - Called +1, +2, +3 nucleosomes to each ORF

        * for Sense and Antisense strands

    """

    print_fl("***********************")
    print_fl("* 3      Metrics      *")
    print_fl("***********************")

    print_preamble()

    # paths to save cross correlations per chromosome
    mkdirs_safe([cc_sense_chrom_dir, cc_antisense_chrom_dir])
    
    if USE_SLURM: mkdirs_safe([WATCH_TMP_DIR])
    
    print_fl("\n------- Read inputs ----------\n")
    read_input_data()

    print_fl("\n------- Calculate occupancies (Sense) ----------\n")
    compute_occupancies()

    print_fl("\n------- Calculate cross correlation (Sense) ----------\n")
    compute_cross_correlations()  

    print_fl("\n------- Calculate nucleosome shift (Sense) ----------\n")
    call_p123_nucleosomes()

    print_fl("\n------- Calculate entropy (Sense) ----------\n")
    compute_organization_measures(strand='sense')

    print_fl("\n------- Calculate occupancies (Antisense) ----------\n")
    compute_occupancies(strand='antisense')

    print_fl("\n------- Calculate cross correlation (Antisense) ----------\n")
    compute_cross_correlations(strand='antisense')

    print_fl("\n------- Calculate nucleosome shift (Antisense) ----------\n")
    call_p123_nucleosomes(strand='antisense')

    print_fl("\n------- Calculate entropy (Antisense) ----------\n")
    compute_organization_measures(strand='antisense')

if __name__ == '__main__':
    main()
