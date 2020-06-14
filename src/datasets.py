
import pandas as pd
import numpy as np

def create_orfs_time_df(orf_idxs, times=[0.0, 7.5, 15, 30, 60, 120], columns=[]):
    """Create a dataframe indexed by orf names and times, 
    each metric will be in this format"""
    index = pd.MultiIndex.from_product([orf_idxs, times], 
        names=['orf_name', 'time'])
    df = pd.DataFrame(index=index, columns=columns)
    return df


def read_orfs_data(filename=None):
    """
    Load ORFs data set with time columns converting time columns from string
    to integers if necessary. Assumes file is a csv with columns. Assumes there
    is an orf_name column that can be set as the index.
    """

    data = pd.read_csv(filename).set_index('orf_name')
    times = [0, 7.5, 15, 30, 60, 120]
    
    non_time_cols = list(data.columns)
    int_time_cols = []
    for time in times:

        # check if time column exists
        str_time_col = "%.1f" % time
        if not str_time_col in data.columns:
            str_time_col = str(int(time))
            if not str_time_col in data.columns:
                continue

        # rename column to numeric
        data = data.rename(columns={str_time_col:time})

    return data


def sort_orf_data(data, func=np.mean):
    """
    Sort orfs by aggregating function over time
    """
    sorted_data = data.copy()
    sorted_data['sort_val'] = func(sorted_data, axis=1)
    sorted_data = sorted_data.sort_values('sort_val', ascending=False)
    return data.loc[sorted_data.index]
