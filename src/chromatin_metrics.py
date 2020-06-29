

import sys
sys.path.append('.')

from src.timer import Timer, TimingContext
from src.transformations import exhaustive_counts
from src.occupancy import calculate_occupancies, calculate_coverage
from src.cross_correlation import compute_cross_correlation_metrics, summary_cross_correlation_cc_class
from src.datasets import read_orfs_data
from src.utils import get_link, print_fl
import pandas as pd
from src.cross_correlation import get_cross_correlation


def main():

    run_wide = False
    run_merge = False
    run_occupancy = False
    run_cross = False
    run_antisense = False

    # parse arguments
    if len(sys.argv) < 3: 
        print_fl("Usage: python chromatin_metrics.py "
                 "<all/merge/wide/occupancy/cross> <chr/all>")
        sys.exit(1)

    commands = sys.argv[1].split(';')
    arg_chrom = sys.argv[2]
    if arg_chrom == 'all': arg_chrom = None
    else: arg_chrom = int(arg_chrom)

    print_fl("Running '%s' for chromosome %s" % (','.join(commands), str(arg_chrom)))

    if 'all' in commands:
        run_wide = True
        run_occupancy = True
        run_cross = True 

    if 'antisense' in commands:
        print_fl("Running antisense")
        run_antisense = True

    if 'merge' in commands:
        run_merge = True
    elif 'occupancy' in commands:
        run_occupancy = True

    if run_antisense:
        write_dir = 'output/2020-03_14_antisense_chromatin_metrics_chrom/'
    else:
        write_dir = 'output/2020-03_14_chromatin_metrics_chrom/'

    print_fl("Using write directory %s" % write_dir)

    # create wide df
    if run_wide:
        print_fl("Computing wide format counts...")
        wide_counts_df = compute_orf_wide_dfs(arg_chrom, write_dir, run_antisense)
    else: wide_counts_df = None

    # run metrics
    if run_occupancy or run_cross:

        if wide_counts_df is None:
            print_fl("Loading wide counts...")
            path = "%s/wide_counts_chr%d.h5.z" % (write_dir, arg_chrom)
            wide_counts_df = pd.read_hdf(path, 'counts')

            print_fl("Checking validity of wide counts...")
            wide_counts_df = check_wide_df(wide_counts_df, write_dir, arg_chrom)

        print_fl("Computing metrics...")
        calculate_and_write_metrics(wide_counts_df, write_dir, 
            arg_chrom, run_occupancy=run_occupancy, run_cross=run_cross)

    # run merge
    if run_merge:
        print_fl("Running merge...")
        create_merged_metrics_df(write_dir, 
            occupancy=('occupancy' in commands),
            cross=('cross' in commands))

def check_wide_df(wide_df, write_dir, arg_chrom):
    """Check that the number of items in the data frame is correct.
    Remove ORFs with missing data"""

    orf_idxs = wide_df.index.levels[0]
    updated_orf_idxs = []

    for orf in orf_idxs:
        if len(wide_df.loc[orf]) != 6*251: 
            print_fl("Omitting orf %s" % orf)
        else: updated_orf_idxs.append(orf)

    print_fl("Creating updated df")
    updated_wide_df = wide_df.loc[updated_orf_idxs]

    if len(updated_orf_idxs) == len(orf_idxs):
        updated_wide_df = wide_df
    else:
        print_fl("Reseting index %d ORFs" % len(updated_orf_idxs))

        if len(updated_orf_idxs) > 500:
            reseted_df = pd.DataFrame()
            for i in range(0, len(updated_orf_idxs), 500):
                cur_orfs = updated_orf_idxs[i:i+500]
                print_fl("  %d/%d" % (i, len(updated_orf_idxs)))
                cur_df = updated_wide_df.loc[cur_orfs].reset_index().set_index(
                    ['parent', 'time', 'length'])
                reseted_df = reseted_df.append(cur_df)
            updated_wide_df = reseted_df
        else: 
            updated_wide_df = updated_wide_df.reset_index().set_index(
                ['parent', 'time', 'length'])

    path = "%s/wide_counts_chr%d.h5.z" % (write_dir, arg_chrom)
    updated_wide_df.to_hdf(path, 'counts', 
        mode='w', complevel=9, complib='zlib')

    return updated_wide_df

def create_merged_metrics_df(write_dir, occupancy, cross):

    if cross:
        # merge cross correlation dataframes
        print_fl("Merging cross correlation...")
        cross_correlation = pd.DataFrame()
        for chrom in range(1, 17):
            path = "%s/orf_cross_correlation_chr%d.h5.z" % (write_dir, chrom)
            cur_df = pd.read_hdf(path, 'cross_correlation')
            cross_correlation = cross_correlation.append(cur_df)
        # write to disk
        cross_correlation.to_hdf('%s/orf_cross_correlation.h5.z' % write_dir, 
            'cross_correlation', mode='w', complevel=9, complib='zlib')

        # merge cross correlation summary
        print_fl("Merging cross correlation summary...")
        orf_cross_correlation_summary = pd.DataFrame()
        for chrom in range(1, 17):
            path = "%s/orf_cross_correlation_summary_chr%d.csv" % (write_dir, chrom)
            cur_df = pd.read_csv(path)
            orf_cross_correlation_summary = orf_cross_correlation_summary.append(cur_df)
        # write to disk
        orf_cross_correlation_summary.to_csv('%s/orf_cross_correlation_summary.csv' %
                write_dir, index=False)

    # merge coverage
    if occupancy:
        print_fl("Merging coverage...")
        coverage = pd.DataFrame()
        for chrom in range(1, 17):
            path = "%s/coverage_chr%d.csv" % (write_dir, chrom)
            cur_df = pd.read_csv(path)
            coverage = coverage.append(cur_df)
        # write to disk
        coverage.to_csv('%s/coverage.csv' %
            write_dir, index=False)

        # merge coverage
        print_fl("Merging occupancy...")
        occupancy = pd.DataFrame()
        for chrom in range(1, 17):
            path = "%s/occupancy_chr%d.csv" % (write_dir, chrom)
            cur_df = pd.read_csv(path)
            occupancy = occupancy.append(cur_df)
        # write to disk
        occupancy.to_csv('%s/occupancy.csv' %
            write_dir, index=False)


def calculate_and_write_metrics(wide_counts_df, write_dir, arg_chrom=None,
    run_cross=True, run_occupancy=True):

    print_fl("Calculating chromatin metrics and writing to %s...\n" % write_dir)

    orf_idxs = wide_counts_df.index.levels[0]
    times = [0.0, 7.5, 15, 30, 60, 120]
    n = len(orf_idxs)
    num_times = len(times)

    if arg_chrom is None: suffix  = ''
    else: suffix = '_chr%d' % arg_chrom

    if run_cross:
        with TimingContext() as timing:
            print_fl("\nCalculating cross correlation...")
            cross_cor_df = compute_cross_correlation_metrics(wide_counts_df)
            write_path = '%s/orf_cross_correlation%s.h5.z' % (write_dir, suffix)
            cross_cor_df.to_hdf(write_path, 'cross_correlation', 
                mode='w', complevel=9, complib='zlib')
            print_fl("  " + timing.get_time())
            print_fl("\nSummarizing cross correlation...")

        with TimingContext() as timing:
            cross_cor_summary = summary_cross_correlation_cc_class(cross_cor_df)
            write_path = "%s/orf_cross_correlation_summary%s.csv" % (write_dir, suffix)
            cross_cor_summary.to_csv(write_path, float_format='%.3f')

            print_fl("  " + timing.get_time())
            print_fl("\nCalculating occupancy metrics...")

        
            print_fl("\nCalculating occupancy...")
            occupancy = calculate_occupancies(wide_counts_df)
            write_path = "%s/occupancy%s.csv" % (write_dir, suffix)
            occupancy.to_csv(write_path, float_format='%.3f')
            print_fl("  " + timing.get_time())


if __name__ == '__main__':
    with TimingContext() as main_timing:
        main()
        print_fl("\nDone. ")
        print_fl("  " + main_timing.get_time())

