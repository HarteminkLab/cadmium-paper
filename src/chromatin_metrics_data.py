

import pandas as pd
import numpy as np
from src import plot_utils
from src.transformations import log2
from src.transformations import log2_fold_change
from src.transformations import difference
from src.transformations import normalize_by_time
from src.datasets import read_orfs_data
from matplotlib import collections as mc
from sklearn.preprocessing import scale
from config import *


class ChromatinDataStore:

    def __init__(self, is_antisense=False, 
        z_score_scale=True):

        self.is_antisense = is_antisense

        if is_antisense: strand_name = 'antisense'
        else: strand_name = 'sense'

        orfs = read_orfs_data('%s/orfs_cd_paper_dataset.csv' % OUTPUT_DIR)
        orfs_idx = orfs.index.values

        antisense_path = '%s/antisense_boundaries_computed.csv' % rna_dir
        antisense_TSS = read_orfs_data(antisense_path)

        if is_antisense:
            orfs_idx = antisense_TSS.dropna().index.values

        xrate = read_orfs_data('%s/orf_xrates.csv' % rna_dir)
        xrate_logfold = read_orfs_data('%s/orf_xrates_log2fold.csv' % rna_dir)

        path = '%s/occupancies_%s.csv' % (mnase_dir, strand_name)
        occupancy = pd.read_csv(path)\
            .set_index(['orf_name', 'time'])
        self.occupancy = occupancy

        from src.entropy import load_orf_entropies

        p1_positions = read_orfs_data('%s/p1_%s.csv' % (mnase_dir, strand_name))
        p2_positions = read_orfs_data('%s/p2_%s.csv' % (mnase_dir, strand_name))
        p3_positions = read_orfs_data('%s/p3_%s.csv' % (mnase_dir, strand_name))

        self.p1_shift = difference(p1_positions)
        self.p2_shift = difference(p2_positions)
        self.p3_shift = difference(p3_positions)

        self.p1 = p1_positions
        self.p2 = p2_positions
        self.p3 = p3_positions

        self.N = len(orfs_idx)

        # promoter occupancy (scale by length of 'promoter')
        self.promoter_sm_occupancy = pivot_metric(occupancy.loc[orfs_idx], 
            '-200_0_len_0_100') / 200.
        self.promoter_sm_occupancy = normalize_by_time(self.promoter_sm_occupancy)

        # scale by length of 'gene body'
        gene_body_organization = load_orf_entropies('0_150', 'triple', strand_name)

        self.gene_body_organization = gene_body_organization.copy().loc[orfs_idx]
        self.gene_body_organization = normalize_by_time(self.gene_body_organization)

        # scale by length of 'promoter'
        promoter_organization = load_orf_entropies('-200_0', 'triple', strand_name)
        self.promoter_organization = promoter_organization.loc[orfs_idx]
        self.promoter_organization = normalize_by_time(self.promoter_organization)

        self.transcript_rate = xrate.copy().loc[orfs_idx]
        self.transcript_rate_logfold = xrate_logfold.loc[orfs_idx]

        self.promoter_sm_occupancy_delta = \
            difference(self.promoter_sm_occupancy.loc[orfs_idx])
        self.gene_body_disorganization_delta = \
            difference(self.gene_body_organization.loc[orfs_idx])
        self.promoter_disorganization_delta = \
            difference(self.promoter_organization.loc[orfs_idx])

        self.orfs = orfs

        self.z_score_scale = z_score_scale

        if self.z_score_scale:

            self.gene_body_disorganization_delta = \
                self.z_scale_chromatin(self.gene_body_disorganization_delta)
            self.promoter_sm_occupancy_delta = \
                self.z_scale_chromatin(self.promoter_sm_occupancy_delta)

        self.chromatin_data = self.promoter_sm_occupancy_delta.join(
            self.gene_body_disorganization_delta, 
            lsuffix='_promoter', rsuffix='_gene')

        self.data = self.chromatin_data.join(self.transcript_rate_logfold, how='inner')
        self.xlabels = [
            'Promoter\noccupancy',
            'Nucleosome\ndisorganization',
            'Transcription\nrate'
        ]

        self.sort_data()


    def z_scale_chromatin(self, data):
        # scale each feature so they are comparable in sorting
        data = data.copy()
        cols = data.columns

        # scale all times by the same amount
        mean = data[cols].values.flatten().mean()
        std = data[cols].values.flatten().std()

        data.loc[:, cols] = (data[cols] - mean) / std
        return data

    def sort_data(self):

        self.combined_chromatin_score = self.chromatin_data.mean(axis=1).\
            sort_values(ascending=False)
        sorted_idx = self.combined_chromatin_score.index.values
        self.chromatin_data = self.chromatin_data.loc[sorted_idx]
        self.data = self.data.loc[sorted_idx]


def add_suffix(pivot_df, suffix):
    times = pivot_df.columns
    mapping = {}
    for t in times:
        mapping[t] = '%.1f%s' % (t, suffix) 
    return pivot_df.rename(columns=mapping)

def pivot_metric(data, value_key, suffix=None):
    pivot_df = data[[value_key]].reset_index().pivot(index='orf_name', 
        columns='time', values=value_key)

    if suffix is not None:
        pivot_df = add_suffix(pivot_df, suffix)

    return pivot_df
