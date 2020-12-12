
import sys
sys.path.append('.')

from src.timer import Timer
import pandas as pd
import numpy as np
from src.utils import print_fl
from src.datasets import read_orfs_data


def filter_mnase_counts_wide(wide_counts_df, len_span, pos_span):
    """
    Query an MNase-seq dataframe indexed by length; columned by position, and
    values are counts.
    """
    position_df = wide_counts_df[np.arange(pos_span[0], pos_span[1])]
    len_query = 'length >= %d & length < %d' % len_span
    queried_df = position_df.query(len_query)
    return queried_df

# for caching calls to filter MNase seq data
# TODO: fix cache
cached_mnase_span = None
cached_chrom = None
cached_mnase = None

def filter_mnase(mnase, start=None, end=None, chrom=None, 
    sample=None, time=None, length_select=(0, 250), use_cache=False, 
    flip=False, translate_origin=None, sample_key='time'):
    """
    Filter MNase-seq data given the argument parameters, do not filter if
    not specified
    """

    global cached_mnase_span
    global cached_chrom
    global cached_mnase

    if time is not None:
        sample = time
        sample_key = 'time'

    if use_cache and end is not None and start is not None:

        # for determining to reset cache and pad search window
        cache_padding = 500

        if (cached_mnase is None or end > cached_mnase_span[1]-cache_padding
             or start < cached_mnase_span[0]+cache_padding or cached_chrom != chrom):

            # set cache to 100000 window from start
            # next genes should be nearest this window greater than this
            # assumes start and end are less than nearest_win
            nearest_win = 100000

            if end - start > nearest_win: raise ValueError("Error setting cache")

            start_cache = start - cache_padding
            cached_mnase_span = start_cache, start_cache+nearest_win
            cached_chrom = chrom

            # set cache MNase
            cached_mnase = filter_mnase(mnase, start=cached_mnase_span[0], 
                chrom=cached_chrom, end=cached_mnase_span[1], use_cache=False)
            mnase = cached_mnase
        else:
            mnase = cached_mnase

    select = ((mnase.length >= length_select[0]) &
              (mnase.length < length_select[1]))

    if start is not None and end is not None:
        select = ((mnase.mid < end) & 
                  (mnase.mid >= start)) & select

    if chrom is not None:
        select = (mnase.chr == chrom) & select
 
    if sample is not None:
        select = select & (mnase[sample_key] == sample)

    ret_mnase = mnase[select].copy()

    # translade mid positions to translated point
    if translate_origin is not None:
        ret_mnase.mid = ret_mnase.mid - translate_origin
        ret_mnase.start = ret_mnase.start - translate_origin
        ret_mnase.stop = ret_mnase.stop - translate_origin

        # flip across origin if needed
        if flip: 
            ret_mnase.mid = -ret_mnase.mid
            old_start = ret_mnase.start.values.copy()
            ret_mnase.start = -ret_mnase.stop.copy()
            ret_mnase.stop = -old_start

    return ret_mnase


def transform_mnase(mnase, center, strand):
    """
    Transform MNase-seq to center on provided point and flip if on crick strand
    """
    mnase = mnase.copy()

    # translate mids to center on TSS, 5' to 3'
    if strand == '+': mnase['mid'] = mnase.mid - center
    else: mnase['mid'] = center - mnase.mid
    mnase['mid'] = mnase.mid.astype(int)

    return mnase

def collect_mnase(mnase_seq, window, pos_chr_df, 
                  pos_key='position', chrom_key='chromosome',
                  strand=None, set_index=False, log=False):

    collected_mnase_eq = pd.DataFrame()
    win_2 = window/2

    timer = Timer()

    if log:
        print_fl("Collecting MNase-seq fragments for %d entries" % len(pos_chr_df))
        print_fl("around a %d window" % window)

    i = 0
    for chrom in range(1, 17):
        
        # get chromosome specific nucleosoems and MNase-seq
        chrom_entries = pos_chr_df[pos_chr_df[chrom_key] == chrom]    
        if len(chrom_entries) == 0: continue
        chrom_mnase = filter_mnase(mnase_seq, chrom=chrom)
        
        # for each element in the dataset
        for idx, entry in chrom_entries.iterrows():
            
            # get MNase-seq fragments at pos_chr_df
            # and 0 center 
            center = entry[pos_key]
            nuc_mnase = filter_mnase(chrom_mnase, start=center-win_2, end=center+win_2)

            # orient properly left to right (upstream to downstream)
            if strand is None or entry[strand] == '+':
                nuc_mnase.loc[:, 'mid'] = nuc_mnase.mid - center
            # crick strand, flip 
            else:
                nuc_mnase.loc[:, 'mid'] = center - nuc_mnase.mid

            select_columns = ['chr', 'length', 'mid', 'time']
            if set_index:
                nuc_mnase['parent'] = idx
                select_columns.append('parent')

            # append to MNase-seq
            collected_mnase_eq = collected_mnase_eq.append(nuc_mnase[select_columns])

            # print_fl progress
            if log and i % 200 == 0: print_fl("%d/%d - %s" % (i, len(pos_chr_df), 
                timer.get_time()))
            i += 1

    if log: timer.print_time()

    return collected_mnase_eq


# sample merged datasets so each time has `min_depth` number of reads
def sample_mnase(mnase_data, sample_depth, times=[0, 7.5, 15, 30, 60, 120]):
    """Sample MNase-seq"""
    np.random.seed(123)
    sampled_mnase = pd.DataFrame()
    for time in times:
        cur_mnase = mnase_data[mnase_data.time == time].sample(sample_depth)
        sampled_mnase = sampled_mnase.append(cur_mnase)
    return sampled_mnase


