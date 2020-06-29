

import sys
sys.path.append('.')

import numpy as np
from src.datasets import create_orfs_time_df
from scipy.signal import correlate2d
from src.cross_correlation_kernel import MNaseSeqDensityKernel
from src.datasets import create_orfs_time_df
import pandas as pd
from src.chromatin import filter_mnase
from src.utils import print_fl
from src.transformations import exhaustive_counts
from src.tasks import TaskDriver, child_done
from config import *
from src.timer import Timer
from src.slurm import submit_sbatch


def cross_filename(directory, chrom):
    return ('%s/cross_correlation_chr%d.h5.z' % (directory, chrom))


def summary_filename(directory, chrom):
    return ('%s/cross_correlation_summary_chr%d.csv' % (directory, chrom))


def task_name(find_antisense):
    return 'cross_correlation_%s' % ('antisense' if find_antisense 
        else 'sense')


def calculate_cross_correlation_all_chromosomes(mnase_seq, TSSs, 
    nuc_kernel, sm_kernel, triple_kernel, log=True, save_chrom_dir=None, timer=None, 
    find_antisense=False):

    name = task_name(find_antisense)
    driver = TaskDriver(name, WATCH_TMP_DIR, 16, timer=timer)
    driver.print_driver()

    # for all chromosomes calculate occupancies per orf
    for chrom in range(1, 17):

        if not USE_SLURM:
            calculate_cross_correlation_chr(mnase_seq, TSSs, chrom, 
                find_antisense, nuc_kernel, sm_kernel, triple_kernel, save_chrom_dir, 
                log, timer)
            child_done(name, WATCH_TMP_DIR, chrom)
        else:
            exports = ("CHROM=%d,ANTISENSE=%s,SLURM_WORKING_DIR=%s,CONDA_PATH=%s,CONDA_ENV=%s"
                       % (chrom, str(find_antisense), SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV))
            script = 'scripts/2_preprocessing/cross_correlation.sh'
            submit_sbatch(exports, script, WATCH_TMP_DIR)

    # wait for all chromosomes to finish
    # superfluous if not in SLURM mode
    driver.wait_for_tasks()
    print_fl()

    # merge
    summary_cross = pd.DataFrame()
    cross = pd.DataFrame()
    for chrom in range(1, 17):
        chrom_cross = pd.read_hdf(cross_filename(save_chrom_dir, chrom), 
            'cross_correlation')

        if len(chrom_cross) == 0: continue

        chrom_summary = pd.read_csv(summary_filename(save_chrom_dir, chrom))\
            .set_index('orf_name')
        cross = cross.append(chrom_cross)
        summary_cross = summary_cross.append(chrom_summary)

    summary_cross = np.round(summary_cross, 5)
    cross = np.round(cross, 5)

    return cross, summary_cross


def calculate_cross_correlation_chr(mnase_seq, TSSs, chrom, find_antisense,
    nuc_kernel, sm_kernel, triple_kernel, save_chrom_dir, log=False, timer=None):

    # cross correlation window
    window = 2800
    win_2 = window/2
    
    chrom_orfs = TSSs[TSSs.chr == chrom]

    if len(chrom_orfs) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if log: 
        print_fl("Chromosome %d. %d genes" % (chrom, len(chrom_orfs)))
        if timer is not None:
            timer.print_time()

    chrom_cross_correlations = pd.DataFrame()
    chrom_mnase = filter_mnase(mnase_seq, chrom=chrom)

    i = 0

    for idx, orf in chrom_orfs.iterrows():

        if log and i % 200 == 0: 
            print_fl("  %d/%d - %s" % (i, len(chrom_orfs), timer.get_time()))

        i += 1

        cur_wide_counts_df, cur_cc = calculate_cross_correlation_orf(
            orf, chrom_mnase, window, nuc_kernel, sm_kernel, triple_kernel)

        # round to save memory. 
        # TODO: number of digits to round to may need to be modified
        chrom_cross_correlations = chrom_cross_correlations.append(cur_cc)

    chrom_cross_correlations = np.round(chrom_cross_correlations, 5) 

    # save chrom cross correlation
    chrom_cross_correlations.to_hdf(
        cross_filename(save_chrom_dir, chrom),
        'cross_correlation',  mode='w', complevel=9,
        complib='zlib')
    
    if len(chrom_cross_correlations) == 0:
        chrom_summary = pd.DataFrame()
    else:
        chrom_summary = summary_cross_correlation_cc_class(\
            chrom_cross_correlations)
        # round to save memory. 
        # TODO: number of digits to round to may need to be modified
        chrom_summary = np.round(chrom_summary, 5)

    # save summary
    chrom_summary.to_csv(summary_filename(save_chrom_dir, chrom))

    return chrom_summary, chrom_cross_correlations


def calculate_cross_correlation_orf(orf, chrom_mnase, window, 
                                    nuc_kernel, sm_kernel, triple_kernel):
    win_2 = window/2

    # wide enough span for all occupancy measures
    orf_span = orf.TSS - win_2, orf.TSS + win_2
    orf_mnase = filter_mnase(chrom_mnase, start=orf_span[0], 
        end=orf_span[1], chrom=orf.chr)
    orf_mnase.loc[:, 'orf_name'] = orf.name

    # convert to centered on TSS
    orf_mnase.loc[:, 'mid'] = orf_mnase.mid - orf.TSS

    # flip if crick
    if orf.strand == '-':
        orf_mnase.loc[:, 'mid'] = orf_mnase.mid*-1

    # exhaustive data frame with all positions (as columns) 
    # and lengths (as rows). Inserting 0s if needed
    # to allow for sliding kernel cross correlation/convolution
    cur_wide_counts_df = exhaustive_counts((-win_2, win_2),
        (0, 250), 'mid', 'length', parent_keys=['orf_name', 'time'], 
        data=orf_mnase, returns='wide', log=False)

    cur_cc = compute_cross_correlation_metrics(cur_wide_counts_df, nuc_kernel,
        sm_kernel, triple_kernel)

    return cur_wide_counts_df, cur_cc


def compute_cross_correlation_metrics(wide_counts_df, nuc_kernel,
    sm_kernel, triple_kernel, times=[0.0, 7.5, 15, 30, 60, 120]):
    """Calculate the three cross correlation values from the nucleosomal
    and small factors kernels"""

    nuc_cc = get_cross_correlation(wide_counts_df, nuc_kernel, times=times)
    small_cc = get_cross_correlation(wide_counts_df, sm_kernel, times=times)
    triple_cc = get_cross_correlation(wide_counts_df, triple_kernel, times=times)

    pos_span = max(nuc_cc.columns.min(), small_cc.columns.min()), \
               min(nuc_cc.columns.max(), small_cc.columns.max())
    pos_range = np.arange(pos_span[0], pos_span[1]+1)
    nuc_cc = nuc_cc[pos_range]
    small_cc = small_cc[pos_range]
    diff_cc = nuc_cc[pos_range] - small_cc[pos_range]

    small_cc['cc_class'] = 'small'
    nuc_cc['cc_class'] = 'nucleosomal'
    diff_cc['cc_class'] = 'diff'
    triple_cc['cc_class'] = 'triple'

    nuc_cc = nuc_cc.set_index('cc_class', append=True)
    small_cc = small_cc.set_index('cc_class', append=True)
    diff_cc = diff_cc.set_index('cc_class', append=True)

    triple_cc = triple_cc.set_index('cc_class', append=True)

    # ensure that the cc's have the same columns
    same_cols = triple_cc.columns

    nuc_cc = nuc_cc[same_cols]
    small_cc = small_cc[same_cols]
    diff_cc = diff_cc[same_cols]

    cross_correlation_df = nuc_cc.append(small_cc).append(diff_cc).append(triple_cc)
    cross_correlation_df = cross_correlation_df.reorder_levels(['cc_class', 
        'orf_name', 'time'])

    return cross_correlation_df


def get_cross_correlation(wide_counts_df, kernel, 
    times=[0.0, 7.5, 15, 30, 60, 120]):
    """
    Assumes ndarray of (orf, time, length, position)
    """

    # calculate indices to be create the resulting dataframe
    kernel_span = kernel.extent[0], kernel.extent[1]
    positions = wide_counts_df.columns.values
    pos_span = positions.min(), positions.max()
    kernel_width_2 = (kernel_span[1]-kernel_span[0])/2
    result_span = pos_span[0]+kernel_width_2, pos_span[1]-kernel_width_2
    result_len = result_span[1]-result_span[0]+1

    orf_idxs = wide_counts_df.index.levels[0]
    n = 1
    num_times = len(times)

    kern_mat = kernel.kernel_mat

    conv_df = create_orfs_time_df(orf_idxs, columns=np.arange(result_span[0], result_span[1]+1))

    for orf_name in orf_idxs:
        for time in times:

            try:
                cur_arr = wide_counts_df.loc[orf_name].loc[time].values
            except Exception as e:
                print_fl("Exception thrown for ORF %s.\n%s" % (orf_name, str(e)))
                continue

            cur_conv_score = correlate2d(cur_arr, kern_mat, mode='valid')
            conv_df.loc[orf_name].loc[time] = cur_conv_score

    return conv_df.astype(float)


def summary_cross_correlation_cc_class(cross_cor_df):
    df = pd.DataFrame()
    for cc_class in ['nucleosomal', 'small', 'diff' , 'triple']:
        cc_summary = summary_cross_correlation(cross_cor_df.loc[cc_class])
        cc_summary['cc_class'] = cc_class
        df = df.append(cc_summary)

    df_summary = df.set_index('cc_class', append=True).reorder_levels(
        ['cc_class', 'orf_name', 'time'])
    return df_summary


def summary_cross_correlation(cross_cor_df):
    """Summarize cross correlation in specified windows"""

    orf_idxs = cross_cor_df.index.levels[0]
    spans = [(-200, 0), (0, 500), (130, 630)]
    cc_summary_df = create_orfs_time_df(orf_idxs)

    for span in spans:
        key = "%d_%d" % span
        cc_summary_df[key] = cross_cor_df[np.arange(*span)].sum(axis=1)

    return cc_summary_df


def main():

    from src.kernel_fitter import compute_triple_kernel

    (_, chrom, antisense) = \
        tuple(sys.argv)
    antisense = antisense.lower() == 'true'

    chrom = int(chrom)
    print_fl("Running cross correlation on chromosome %d, antisense: %s" % 
        (chrom, str(antisense)))

    name = task_name(antisense)
    timer = Timer()

    nuc_kernel = MNaseSeqDensityKernel(filepath=nuc_kernel_path)
    sm_kernel = MNaseSeqDensityKernel(filepath=sm_kernel_path)
    triple_kernel = compute_triple_kernel(nuc_kernel)

    print_fl("Reading MNase-seq...", end='')
    all_mnase_data = pd.read_hdf(mnase_seq_path, 'mnase_data')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    if not antisense:
        cc_dir = cc_sense_chrom_dir
        cc_orfs = read_orfs_data("%s/orfs_cd_paper_dataset.csv" % OUTPUT_DIR)
    else:
        cc_dir = cc_antisense_chrom_dir
        cc_orfs = antisense_orfs

    calculate_cross_correlation_chr(all_mnase_data, cc_orfs, chrom, 
        antisense, nuc_kernel, sm_kernel, triple_kernel, cc_dir, log=True, 
        timer=timer)

    child_done(name, WATCH_TMP_DIR, chrom)


if __name__ == '__main__':
    main()
