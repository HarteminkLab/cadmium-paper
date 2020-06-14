
import sys
sys.path.append('.')


from timer import Timer
import pandas as pd
import numpy as np

from src.datasets import read_orfs_data
from src.chromatin import filter_mnase, transform_mnase, filter_mnase_counts_wide
from src.datasets import create_orfs_time_df




def calculate_occupancies_all_chromosomes(mnase_seq, TSSs, log=False):

    occupancies = pd.DataFrame()

    # for all chromosomes calculate occupancies per orf
    for chrom in range(1, 17):

        chrom_orfs = TSSs[TSSs.chr == chrom]

        if len(chrom_orfs) == 0: continue

        chrom_mnase = filter_mnase(mnase_seq, chrom=chrom)

        for idx, orf in chrom_orfs.iterrows():
            
            # wide enough span for all occupancy measures
            orf_span = orf.TSS - 1000, orf.TSS + 1000
            orf_mnase = filter_mnase(chrom_mnase, start=orf_span[0], 
                end=orf_span[1])

            cur_occupancies = calculate_occupancies(orf_mnase, orf)
            occupancies = occupancies.append(cur_occupancies)

    return occupancies


def calculate_occupancies(orf_mnase, orf, times=[0, 7.5, 15, 30, 60, 120]):
    """
    Calculate occupancies for each region given a wide format dataframe
    indexed by ORF, length; positional columns; and count values
    """

    TSS = orf.TSS
    num_times = len(times)

    ranges = [(-200,0), (-400,0), 
              (0, 500), (0, 1000)]
    len_ranges = [(0, 100), 
                  (159-15, 159+15), 
                  (0, 250)]

    def _key_name(len_span, pos_span):
        return '%d_%d_len_%d_%d' % tuple(list(pos_span) + list(len_span))

    # store occupancies as data frame
    # keyed by orf name and time, columns are partition names
    orf_occupancies = pd.DataFrame()

    # for every length and range combination
    for len_span in len_ranges:
        for pos_span in ranges:
            key = _key_name(len_span, pos_span)

            cur_mnase = filter_mnase(orf_mnase, 
                start=pos_span[0]+TSS, end=pos_span[1]+TSS,
                length_select=len_span)

            counts = cur_mnase[['time', 'chr']].groupby('time').count()\
                .rename(columns={'chr':'count'})
            counts = counts.join(pd.DataFrame(index=times), how='outer')
            counts = counts.fillna(0)
            counts['key'] = key
            counts = counts.reset_index().rename(columns={'index':'time'})

            orf_occupancies = orf_occupancies.append(counts)

    orf_occupancies = orf_occupancies.pivot(index='time', columns='key', 
        values='count').reset_index()
    orf_occupancies.loc[:, 'orf_name'] = orf.name
    orf_occupancies = orf_occupancies.set_index(['orf_name', 'time'])
    orf_occupancies.loc[:] = orf_occupancies.astype(int)

    return orf_occupancies


def calculate_coverage(mnase_seq, TSSs, window=2000):
    """
    Calculate the coverage for each ORF as the proportion of local positions (
    `window` around its TSS) with at least one MNase-seq fragment center
    at any time point.
    """

    win_2 = window/2
    coverage = TSSs[[]].copy()
    coverage['coverage'] = 0.0

    for chrom in range(1, 17):

        chrom_orfs = TSSs[TSSs.chr == chrom]
        if len(chrom_orfs) == 0: continue
        chrom_mnase = filter_mnase(mnase_seq, chrom=chrom)

        for idx, orf in chrom_orfs.iterrows():
            cur_mnase = filter_mnase(chrom_mnase, start=(orf.TSS - win_2), 
                                     end=(orf.TSS + win_2))

            # unique read centers for any time in the time course
            unique_positions = cur_mnase.mid.unique()

            # normalized by the window size
            cur_coverage = float(len(unique_positions)) / float(window)
            coverage.loc[idx] = cur_coverage

    return coverage


if __name__ == '__main__':
    main()
