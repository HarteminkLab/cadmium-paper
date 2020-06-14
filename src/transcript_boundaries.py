
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.rna_seq_plotter import get_smoothing_kernel
from src.transcription import filter_rna_seq_pileup
from src.utils import flip_strand, print_fl
from src.tasks import TaskDriver, child_done
from config import WATCH_TMP_DIR, pileup_path, anti_chrom_dir, sense_chrom_dir, USE_SLURM
from src.timer import Timer
from src.utils import run_cmd
from src.datasets import read_orfs_data
from config import DEBUG, DEBUG_CHROMS
from src.slurm import submit_sbatch


def task_name(find_antisense):
    name = ('transcript_boundaries_%s' % 
           ('sense' if not find_antisense else 'antisense'))
    return name


def compute_boundaries(orfs, pileup, save_dir, find_antisense=False, 
    log=False, timer=None, pileup_path=None):

    name = task_name(find_antisense)

    driver = TaskDriver(name, WATCH_TMP_DIR, 16, timer=timer)
    driver.print_driver()

    # find antisense transcript boundaries if possible
    # start from sense TSS and grow outward if an antisense peak is high enough
    for chrom in range(1, 17):

        if not USE_SLURM:
            compute_boundaries_chrom(orfs, pileup, chrom, save_dir, find_antisense,
               log=log, timer=timer)
            child_done(name, WATCH_TMP_DIR, chrom)
        else:
            exports = ("CHROM=%d,ANTISENSE=%s,SLURM_WORKING_DIR=%s,CONDA_PATH=%s,CONDA_ENV=%s"
                       % (chrom, str(find_antisense), SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV))
            script = 'scripts/2_preprocessing/boundaries.sh'
            submit_sbatch(exports, script, WATCH_TMP_DIR)

    # wait for all chromosomes to finish
    # superfluous if not in SLURM mode
    driver.wait_for_tasks()

    # merge completed transcript boundary files
    transcript_boundaries = pd.DataFrame()
    for chrom in range(1, 17):
        cur_boundaries = pd.read_csv(boundary_file_name(save_dir, chrom, 
            find_antisense)).set_index('orf_name')
        transcript_boundaries = transcript_boundaries.append(cur_boundaries)

    return transcript_boundaries

def boundary_file_name(directory, chrom, find_antisense):
    return '%s/%s_boundaries_computed_chr%s.csv' % \
        (directory, ('antisense' if find_antisense else 'sense'),
         chrom)

def compute_boundaries_chrom(orfs, pileup, chrom, save_dir, find_antisense,
    log=False, timer=None):

    chrom_genes = orfs[orfs.chr == chrom]

    transcript_boundaries = chrom_genes[[]].copy()
    transcript_boundaries['start'] = None
    transcript_boundaries['stop'] = None

    search_window = 2000
    search_2 = search_window/2

    if log and timer is not None: 
        print_fl("Chromosome %d - %s. %d genes" % 
            (chrom, timer.get_time(), len(chrom_genes)))

    chrom_rna_seq = filter_rna_seq_pileup(pileup, chrom == chrom)

    i = 0
    for orf_name, gene in chrom_genes.iterrows():

        if log and timer is not None and i % 100 == 0: 
            print_fl("%d/%d - %s" % (i, len(chrom_genes),
             timer.get_time()))

        i += 1

        span = gene.transcript_start-search_2, gene.transcript_stop+search_2
        gene_pileup = filter_rna_seq_pileup(chrom_rna_seq, 
            span[0], span[1], gene.chr)

        try:
            start, stop = find_transcript_boundaries(gene_pileup, span, gene,
                find_antisense=find_antisense)
        except ValueError:
            # skip if issues finding boundaries
            continue

        transcript_boundaries.loc[orf_name, 'start'] = start
        transcript_boundaries.loc[orf_name, 'stop'] = stop

        TSS, TES = start, stop
        if ((gene.strand == '-' and not find_antisense) or
            (gene.strand == '+' and find_antisense)):
            TSS, TES = stop, start

        strand = gene.strand
        if find_antisense: strand = flip_strand(gene.strand) 

        transcript_boundaries.loc[orf_name, 'TSS'] = TSS
        transcript_boundaries.loc[orf_name, 'TES'] = TES
        transcript_boundaries.loc[orf_name, 'strand'] = strand
        transcript_boundaries.loc[orf_name, 'chr'] = gene.chr

    transcript_boundaries = transcript_boundaries.dropna()
    transcript_boundaries.to_csv(boundary_file_name(save_dir, chrom, 
        find_antisense))

    return transcript_boundaries


def find_boundaries(gene, span, sense_TSS, x, y_smoothed):
    
    df = pd.DataFrame({'x': x, 'y': y_smoothed})
    df = df.set_index('x')

    highest_pos = df.loc[sense_TSS-200:sense_TSS+200].idxmax().y
    highest_val = df.loc[highest_pos].y

    if highest_val < 5: return None, None

    start = None

    def _get_search_boundary(highest_val, arange, default_val):
        """find left boundary, this highest value to first time on the left
           side it reachest this 1.1*value (or boundary), or the value is 10% 
           larger than the previous position"""
        boundary = default_val
        prev = highest_val
        for i in arange:

            # skip if out of range
            if i not in df.index: continue

            cur = df.loc[i].y
            if cur >= highest_val*1.1 or cur > prev*1.1:
                boundary = i
                break
        return boundary

    left_boundary = _get_search_boundary(highest_val, 
        np.arange(highest_pos-1, span[0], -1), span[0])
    right_boundary = _get_search_boundary(highest_val,
        np.arange(highest_pos+1, span[1]), span[1])

    start = df.loc[left_boundary:highest_pos].idxmin().y
    stop = df.loc[highest_pos:right_boundary].idxmin().y

    return start, stop

def plot_boundaries(orf_rna_seq, span, gene, anti_start, anti_stop):
    smooth_kernel = get_smoothing_kernel(100, 10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    sense_strand = '+' if gene.strand == '+' else '-'
    antisense_strand = '+' if sense_strand == '-' else '-'

    pos = orf_rna_seq[orf_rna_seq.strand == sense_strand]\
        .groupby('position').sum()

    y = np.log2(pos.pileup+1)
    y_smoothed = np.convolve(y, smooth_kernel, mode='same')

    ax1.plot(pos.index, y_smoothed)
    ax1.axvline(gene.transcript_start)
    ax1.axvline(gene.transcript_stop)
    ax1.set_xlim(*span)
    ax1.set_title('Sense')

    pos = orf_rna_seq[orf_rna_seq.strand == antisense_strand]\
        .groupby('position').sum()
    x = pos.index
    y = np.log2(pos.pileup+1)

    # kernel
    y_smoothed = np.convolve(y, smooth_kernel, mode='same')

    ax2.plot(pos.index, y_smoothed)
    ax2.axvline(anti_start)
    ax2.axvline(anti_stop)
    ax2.set_xlim(*span)
    ax2.set_title('Antisense')


def find_transcript_boundaries(orf_pileup, span, gene, find_antisense=False):
    smooth_kernel = get_smoothing_kernel(100, 10)

    sense_strand = '+' if gene.strand == '+' else '-'
    antisense_strand = '+' if sense_strand == '-' else '-'

    if find_antisense: strand = antisense_strand
    else: strand = sense_strand

    # Start of transcript (either start of ORF or Park's TSS)
    if sense_strand == '+':
        start_search_loc = gene.transcript_start
    else:
        start_search_loc = gene.transcript_stop

    pos = orf_pileup[orf_pileup.strand == strand].groupby('position').sum()
    x = pos.index
    y = np.log2(pos.pileup+1)
    y_smoothed = np.convolve(y, smooth_kernel, mode='same')
    start, end = find_boundaries(gene, span, start_search_loc, x, y_smoothed)
    return start, end

def load_park_boundaries():

    from src.reference_data import load_park_orf_transcript_boundaries
    park_boundaries = load_park_orf_transcript_boundaries()
    
    if DEBUG: park_boundaries = park_boundaries[
        park_boundaries.chr.isin(DEBUG_CHROMS)]
    return park_boundaries

def main():
    """
    Usage:
        python transcript_boundaries <chrom> <save_dir> 
            <antisense> <pileup_path>

        e.g.

        python src/transcript_boundaries.py 1 True
    """
    
    (_, chrom, antisense) = \
        tuple(sys.argv)

    orfs = load_park_boundaries()

    # path loaded from config
    pileup = pd.read_hdf(pileup_path, 'pileup')

    chrom = int(chrom)
    antisense = antisense.lower() == 'true'

    print_fl("Running transcript boundaries on chromosome %d, antisense: %s" % 
        (chrom, str(antisense)))

    save_dir = anti_chrom_dir if antisense else \
        sense_chrom_dir

    name = task_name(antisense)

    timer = Timer()
    compute_boundaries_chrom(orfs, pileup, chrom, 
        save_dir, antisense,
        log=True, timer=timer)

    child_done(name, WATCH_TMP_DIR, chrom)

if __name__ == '__main__':
    main()

