
import numpy as np
import itertools
import pandas as pd
import sys
from src.timer import TimingContext
from src.utils import print_fl


def fold_change(data, times=[0, 7.5, 15, 30, 60, 120], 
        pseudo_count=0.1,
        neg_vals=0):
    """
    Calculate fold ratio change compared to the first time point, adding pseudo
    count in cases of 0 value.
    """

    data = data.copy()
    data += pseudo_count
    time_zero = data[times[0]].copy()

    # fold ratio
    data.loc[:, times] = data[times].divide(time_zero, axis=0)

    return data


def log2(data, times=[0, 7.5, 15, 30, 60, 120], fc_floor=1):
    """
    Scale data to log2 add pseudo count if data is <= 0.
    """
    data = data.copy()

    # use floor value for values <= 0
    for time in times:
        data.loc[data[time] <= 0, time] = fc_floor

    data.loc[:, times] = np.log2(data)
    return data

def arcsinh_fold_change(data, times=[0, 7.5, 15, 30, 60, 120], pseudo_count=0.1):
    fc_data = fold_change(data, times, pseudo_count=pseudo_count)
    # return fc_data
    asinh_data = np.arcsinh(fc_data) - np.arcsinh(1)
    return asinh_data

def log2_fold_change(data, times=[0, 7.5, 15, 30, 60, 120],
    pseudo_count=1,
    fc_floor=1.):
    """log2 fold change wrt to 0 time"""
    fc_data = fold_change(data, times, pseudo_count=pseudo_count)
    log2_data = log2(fc_data, times, fc_floor=fc_floor)
    return log2_data

def difference(data, times=[0, 7.5, 15, 30, 60, 120]):
    """
    Calculate difference to the first time point
    """
    data = data.copy()
    time_zero = data[times[0]].copy()

    # difference
    data.loc[:, times] = data[times].subtract(time_zero, axis=0)

    return data


def normalize_by_time(data, how='z-score'):
    """
    Scale each sample to sum over all ORFs to the target sum
    """
    if how == 'z-score':
        scaled = (data - data.mean()) / data.std()
    else: ValueError("Undefined normalization methods")

    return scaled


def exhaustive_counts(x_span, y_span, x_key='mid', y_key='length', data=None, 
    returns='both', parent_keys=None, log=False):
    """
    Create exhaustive dataframe of all xs and y values for a dataframe, counting
    existing of x, y combination. Returns narrow list or pivoted
    """

    exhaustive_values = [np.arange(x_span[0], x_span[1]+1), 
                         np.arange(y_span[0], y_span[1]+1)]

    if parent_keys is not None:
        for parent_key in parent_keys:
            exhaustive_values.append(data[parent_key].unique())

    with TimingContext() as timing:
        if log: print_fl("  Creating full range list...")

        # create dataframe of full range of values to join in case fragment doesnt exist
        # at every position
        full_range_list = list(itertools.product(*exhaustive_values))
        xs = [e[0] for e in full_range_list]
        ys = [e[1] for e in full_range_list]

        full_range = pd.DataFrame()
        full_range[y_key] = ys
        full_range[x_key] = xs

        if parent_keys is not None: 

            # add each parent key's exhaustive values
            for i in range(len(parent_keys)):
                parent_key = parent_keys[i]
                full_range[parent_key] = [e[2+i] for e in full_range_list]

        # set parent key to group on if applicable
        if parent_keys is None: index = [x_key, y_key]
        else: index = parent_keys + [x_key, y_key]
        full_range = full_range.set_index(index)

        if log:
            print_fl("  " + timing.get_time())
            sys.stdout.flush()
            print_fl("  Joining data to full range...")

    with TimingContext() as timing:

        # pivot MNase-seq data into a count histogram
        if data is None:
            full_range['count'] = 0
            narrow_counts = full_range.reset_index()
        else:
            narrow_counts = data[index].copy()
            narrow_counts['count'] = 1
            narrow_counts = narrow_counts.groupby(index).count()

            # join with full range of values
            narrow_counts = narrow_counts.reset_index().merge(full_range.reset_index(), 
                how='outer').fillna(0)
            if parent_keys is not None:
                narrow_counts = narrow_counts.set_index(parent_keys + ['length'])
            else:
                narrow_counts = narrow_counts.set_index(['length'])

        if parent_keys is None:
            pivot_idx = 'length'
        else:
            pivot_idx = parent_keys + [y_key]

        if log:
            print_fl("  " + timing.get_time())
            print_fl("  Pivoting narrow table...")
            sys.stdout.flush()

    with TimingContext() as timing:
        # pivot into length x position matrix
        wide_counts = narrow_counts.pivot_table(index=pivot_idx, columns=x_key, values='count')
        wide_counts = wide_counts.fillna(0).astype(int)
        
        if log:
            print_fl("  " + timing.get_time())
            sys.stdout.flush()

    if returns == 'both':
        return narrow_counts, wide_counts
    elif returns == 'wide':
        del narrow_counts
        return wide_counts
    else:
        raise ValueError("Unspecified parameter")

def z_score_norm(data):
    """
    Normalize each row by z-score for the row. For normalizing correlated genes
    so they can be compared
    """
    data = data.copy()
    prom_cols = data.columns
    mu = data[prom_cols].mean(axis=1)
    std = data[prom_cols].std(axis=1)
    for i in range(len(data.columns)):
        data.loc[:, prom_cols[i]] = (data[prom_cols[i]] - 0)/std
    return data


