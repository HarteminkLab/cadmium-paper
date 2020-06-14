
import sys
sys.path.append('.')

import sys
import pandas as pd
import numpy as np
from timer import Timer
from transcription import filter_rna_seq
from src.tasks import TaskDriver, child_done
from config import *
from src.slurm import submit_sbatch


def calculate_rna_seq_pileup(rna_seq, 
    timer, times=[0, 7.5, 15, 30, 60, 120], log=False):
    """
    Calculate the pileup for each position in the genome
    """

    timer = Timer()
    pileup = pd.DataFrame()
    name = 'pileup'

    driver = TaskDriver(name, WATCH_TMP_DIR, 16, timer=timer)
    driver.print_driver()

    # each chromosome
    for chrom in range(1, 17):
        if not USE_SLURM:
            calculate_pileup_chr(rna_seq, chrom, timer)
            child_done(name, WATCH_TMP_DIR, chrom)
        else:
            exports = ("CHROM=%d,SLURM_WORKING_DIR=%s,CONDA_PATH=%s,CONDA_ENV=%s"
                       % (chrom, SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV))
            script = 'scripts/1_data_initialize/pileup.sh'
            submit_sbatch(exports, script, WATCH_TMP_DIR)

    driver.wait_for_tasks()
    print_fl()

    # merge
    pileup = pd.DataFrame()
    for chrom in range(1, 17):
        if DEBUG and chrom not in DEBUG_CHROMS: continue
        chr_pileup = pd.read_hdf(pileup_chr_path(chrom), 
            'pileup')
        if len(chr_pileup) == 0: continue
        pileup = pileup.append(chr_pileup)

    return pileup


def calculate_pileup_chr(rna_seq, chrom, timer, times=[0, 7.5, 15, 30, 60, 120]):

    print_fl("Pileup for chromosome %d..." % chrom)

    if DEBUG and chrom not in DEBUG_CHROMS: return None

    chrom_rna_seq = filter_rna_seq(rna_seq, chrom=chrom)
    start, end = 1, chrom_rna_seq.stop.max()
    pileup = pd.DataFrame()

    for time in times:

        print_fl("%s - %s" % (str(time), timer.get_time()))

        time_rna_seq = filter_rna_seq(chrom_rna_seq, time=time)

        # each strand
        for strand in ['+', '-']:
            cur_rna_seq = filter_rna_seq(time_rna_seq, strand=strand)

            # get pileup for strand, time, chromosome
            start, end = 1, end
            cur_pileup = get_pileup(cur_rna_seq, start, end)
            cur_pileup['strand'] = strand
            cur_pileup['chr'] = chrom
            cur_pileup['time'] = time

            # append to pileup
            pileup = pileup.append(cur_pileup)

    pileup = pileup.reset_index(drop=True)
    pileup.to_hdf(pileup_chr_path(chrom), 'pileup',  mode='w', complevel=9,
        complib='zlib')

    return pileup


def pileup_chr_path(chrom):
    return "%s/pileup_chr%d.h5.z" % (pileup_chrom_dir, chrom)


def get_pileup(data, start, end):
    """
    Given RNA-seq reads, calculate the pileup by creating rows for each
    position along a fixed fragment length as a pivot table, then unstack
    into unique position reads to count.
    """

    n = len(data)
    frag_len = data['length'].values[0]

    # create a matrix of start positions for each read as a row
    # create a column for each subsequent position along the fragment length
    # and increment the position value, each element in the matrix will
    # correspond to a unique pileup position as a pivoted table
    start_pos = np.array(data[['start']])

    start_pos_repeat = np.concatenate([start_pos]*frag_len, axis=1)
    pos_increment = np.array([np.array(np.arange(0, frag_len))]*n)
    pivoted_positions = np.add(start_pos_repeat, pos_increment)
    positions_df = pd.DataFrame(pivoted_positions)

    # unstack pivoted tabled, each position counts as a pileup value
    counts = positions_df.unstack().reset_index().rename(columns={0:'position'})[['position']]\
        .sort_values('position').reset_index(drop=True)
    counts['pileup'] = 1
    counts = counts.groupby('position').count()
    counts = counts.join(pd.DataFrame(index=np.arange(start, end)), how='outer')
    counts = counts.reset_index().fillna(0)
    counts.pileup = counts.pileup.astype(int)
    counts = counts.rename(columns={'index':'position'})
    
    return counts


def main():

    (_, chrom) = tuple(sys.argv)

    chrom = int(chrom)
    print_fl("Running cross correlation on chromosome %d" % (chrom))

    name = 'pileup'
    timer = Timer()

    print_fl("Reading RNA-seq...", end='')
    rna_seq_data = pd.read_hdf(rna_seq_path, 'rna_seq_data')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    calculate_pileup_chr(rna_seq_data, chrom, timer)

    child_done(name, WATCH_TMP_DIR, chrom)

if __name__ == '__main__':
    main()

