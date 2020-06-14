
import sys
sys.path.append('.')

import pandas as pd
from scipy.stats import entropy
import numpy as np
from src.tasks import TaskDriver, child_done
from config import *
from src.timer import Timer
from src.slurm import submit_sbatch


def bin_reads(nuc_poss, xlim=(0, 500), ylim=(0, 250)):
    """Bin reads into a histogram of counts within a window
    TODO: this is done elsewhere as pivot table, may need to switch to using
    this method if its faster
    """

    # add one to include last bp and one for histogram
    x_extent = np.arange(xlim[0], xlim[1] + 2)
    y_extent = np.arange(ylim[0], ylim[1] + 2)

    H, xedges, yedges = np.histogram2d(nuc_poss[:, 1], nuc_poss[:, 0],
        bins=(y_extent, x_extent))
    return H


def calc_entropy(cross_cor, base=2):
    # compute estimated probability distribution
    total_cc = np.sum(cross_cor)
    prob_est = 1.* cross_cor / (total_cc+0.01) # avoid divide by 0
    return entropy(prob_est, base=base)


def calc_entropy_2d(nuc_poss, base=2):
    """Calculate entropy from raw center reads, requires converting
    into a histogram of counts"""

    counts = bin_reads(nuc_poss).flatten() # treat each pos,len as independent
    
    # compute estimated probability distribution
    # using counts
    prob_est = counts / np.sum(counts)
    return entropy(prob_est, base=base)



def load_orf_entropies_by_cc_type(cc_type, strand_name):

    path = '%s/orf_%s_entropies.csv' % (mnase_dir, strand_name)
    cc_summary = pd.read_csv(path)
    cc_summary = cc_summary.set_index(['cc_type', 'key', 'orf_name', 'time'])

    cc_nuc = cc_summary.loc[cc_type][['entropy']]

    data = None
    for cc_type in ['-200_0', '0_150']:
        cur_cc = cc_nuc.loc[cc_type].rename(columns={'entropy': cc_type})
        if data is None:
            data = cur_cc
        else:
            data = pd.concat([data, cur_cc], axis=1)
            
    return data.copy()


def load_orf_entropies(key, cc_type, strand):

    orf_entropies = pd.read_csv('%s/orf_%s_entropies.csv' % (mnase_dir, strand))
    orf_entropies = orf_entropies.set_index(['key', 'cc_type', 'orf_name', 'time'])
    entropy_pivot = orf_entropies.loc[key].loc[cc_type]
    entropy_pivot = entropy_pivot.reset_index().pivot(index='orf_name', columns='time', values='entropy')

    return entropy_pivot.copy()


def get_name(select_range, cc_type):
    key = "%d_%d" % select_range
    child_name = "%s_%s" % (key, cc_type)
    return child_name


def calculate_cc_summary_measure(orfs, cross_correlation, strand, timer, measure='entropy',
    times=[0.0, 7.5, 15, 30, 60, 120]):

    data = pd.DataFrame()

    name = 'entropy_%s' % strand
    kernel_types = ['nucleosomal', 'small', 'triple']
    select_ranges = [(-200, 0), (0, 500), (0, 100), (0, 150)]
    n = len(kernel_types) * len(select_ranges)

    driver = TaskDriver(name, WATCH_TMP_DIR, n, timer=timer)
    driver.print_driver()

    for select_range in select_ranges:
        for cc_type in kernel_types:
            key = "%d_%d" % select_range
            child_name = get_name(select_range, cc_type)

            if not USE_SLURM:
                calculate_cc_summary_measure_range_type(orfs, cross_correlation,
                    cc_type, select_range, strand, timer, measure)
                child_done(name, WATCH_TMP_DIR, child_name)
            else:
                exports = ("SELECT_RANGE=%s,CC_TYPE=%s,ANTISENSE=%s,SLURM_WORKING_DIR=%s,CONDA_PATH=%s,CONDA_ENV=%s" % \
                          (key, cc_type, str(strand == 'antisense',
                           SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV)))
                script = 'scripts/3_chrom_metrics/entropy.sh'
                submit_sbatch(exports, script, WATCH_TMP_DIR)

    driver.wait_for_tasks()
    print_fl()

    # merge
    data = pd.DataFrame()
    for select_range in select_ranges:
        for cc_type in kernel_types:
            child_name = get_name(select_range, cc_type)
            cur_entropy = pd.read_csv(save_path(strand, select_range, cc_type))
            if len(cur_entropy) == 0: continue
            data = data.append(cur_entropy)

    return data


def calculate_cc_summary_measure_range_type(orfs, cross_correlation,
    cc_type, select_range, strand, timer, 
    measure='entropy', times=[0.0, 7.5, 15, 30, 60, 120]):

    data = pd.DataFrame()
    key = "%d_%d" % select_range

    for orf_name, orf in orfs.iterrows():
        orf_cc = cross_correlation.loc[cc_type].loc[orf_name]\
            [np.arange(select_range[0], select_range[1]+1)]

        for time in times: 
        
            cur_cc = orf_cc.loc[time].values

            if measure == 'entropy':
                value = calc_entropy(cur_cc)

            elif measure == 'peak_to_trough':
                # avoid divide by 0
                value = np.max(cur_cc) / (np.min(cur_cc) + 0.1)

            elif measure == 'peak':

                value = np.max(cur_cc)

            row = pd.Series({'orf_name':orf_name, 'time':time, 'key':key,
                'cc_type': cc_type, measure: value})
            data = data.append(row, ignore_index=True)

    data = data.set_index(['cc_type', 'orf_name', 'time'])
    data.to_csv(save_path(strand, select_range, cc_type))

    return data

def save_path(strand, select_range, cc_type):

    name = get_name(select_range, cc_type)

    if strand == 'sense': save_dir = sense_entropy_dir
    else: save_dir = anti_entropy_dir

    return "%s/orf_%s_entropies_%s.csv" % (save_dir, strand, name)


def main():

    (_, select_range, cc_type, antisense) = \
        tuple(sys.argv)

    antisense = antisense.lower() == 'true'
    strand = 'sense' if not antisense else 'antisense'
    select_range = select_range.split('_')
    select_range = int(select_range[0]), int(select_range[1])

    cross_corr_path = cross_corr_sense_path
    orfs = paper_orfs
    if antisense:
        cross_corr_path = cross_corr_antisense_path
        orfs = antisense_orfs

    child_name = get_name(select_range, cc_type)

    print_fl("Running entropy for %d-%d, %s, %s" % 
        (select_range[0], select_range[1], cc_type, strand))

    name = 'entropy_%s' % strand
    timer = Timer()

    print_fl("Reading Cross correlation...", end='')
    cross_correlation = pd.read_hdf(cross_corr_path, 'cross_correlation')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Computing entropy")
    calculate_cc_summary_measure_range_type(orfs, cross_correlation,
        cc_type, select_range, strand, timer)

    child_done(name, WATCH_TMP_DIR, child_name)


if __name__ == '__main__':
    main()