

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

    def __init__(self, is_antisense=False, output_dir=None):

        self.is_antisense = is_antisense

        if is_antisense: strand_name = 'antisense'
        else: strand_name = 'sense'

        if output_dir is None:
            out_dir = OUTPUT_DIR
        else:
            out_dir = output_dir

        rna_dir = '%s/rna_seq' % out_dir
        mnase_dir = '%s/mnase_seq' % out_dir

        orfs = read_orfs_data('%s/orfs_cd_paper_dataset.csv' % out_dir)
        orfs_idx = orfs.index.values

        antisense_path = '%s/antisense_boundaries_computed.csv' % rna_dir
        antisense_TSS = read_orfs_data(antisense_path)

        if is_antisense:
            orfs_idx = antisense_TSS.dropna().index.values
            antisense_TPM_logfold = read_orfs_data('%s/antisense_TPM_log2fold.csv' % rna_dir)
            self.antisense_TPM_logfold = antisense_TPM_logfold.loc[orfs_idx]
        else:
            sense_TPM_logfold = read_orfs_data('%s/sense_TPM_log2fold.csv' % rna_dir)
            self.sense_TPM_logfold = sense_TPM_logfold.loc[orfs_idx]

        xrate = read_orfs_data('%s/orf_xrates.csv' % rna_dir)
        xrate_logfold = read_orfs_data('%s/orf_xrates_log2fold.csv' % rna_dir)

        path = '%s/occupancies_%s.csv' % (mnase_dir, strand_name)
        occupancy = pd.read_csv(path)\
            .set_index(['orf_name', 'time'])
        self.occupancy = occupancy

        from src.entropy import load_orf_entropies
        from src.nucleosome_calling import load_p123

        (self.p1, self.p2, self.p3, self.p1_shift, 
         self.p2_shift, self.p3_shift) = load_p123(strand_name)

        TPM = read_orfs_data('%s/sense_TPM.csv' % rna_dir)
        self.sense_TPM = TPM
        self.sense_log2_TPM = np.log2(TPM+1)

        self.N = len(orfs_idx)

        # promoter occupancy (scale by length of 'promoter')
        self.promoter_sm_occupancy_raw = pivot_metric(occupancy.loc[orfs_idx], '-200_0_len_0_100')
        self.promoter_sm_occupancy = pivot_metric(occupancy.loc[orfs_idx], 
            '-200_0_len_0_100') / 200.
        self.promoter_sm_occupancy = normalize_by_time(self.promoter_sm_occupancy)

        # promoter nucleosome occupancy (scale by length of 'promoter')
        self.promoter_nuc_occupancy_raw = pivot_metric(occupancy.loc[orfs_idx], '-200_0_len_144_174')
        self.promoter_nuc_occupancy = self.promoter_nuc_occupancy_raw / 200.
        self.promoter_nuc_occupancy = normalize_by_time(self.promoter_nuc_occupancy)

        # gene body nucleosome occupancy (scale by length of 'gene body')
        self.gene_body_nuc_occupancy_raw = pivot_metric(occupancy.loc[orfs_idx], '0_500_len_144_174')
        self.gene_body_nuc_occupancy = self.gene_body_nuc_occupancy_raw / 200.
        self.gene_body_nuc_occupancy = normalize_by_time(self.gene_body_nuc_occupancy)

        # gene body organization
        gene_body_organization = load_orf_entropies('0_150', 'triple', 
            strand_name, mnase_seq_dir=mnase_dir)
        self.gene_body_organization = gene_body_organization.copy().loc[orfs_idx]
        self.gene_body_organization_raw = self.gene_body_organization
        self.gene_body_organization = normalize_by_time(self.gene_body_organization)

        # scale by length of 'promoter'
        promoter_organization = load_orf_entropies('-200_0', 'triple', strand_name, 
            mnase_seq_dir=mnase_dir)
        self.promoter_organization = promoter_organization.loc[orfs_idx]
        self.promoter_organization = normalize_by_time(self.promoter_organization)

        self.transcript_rate = xrate.copy().loc[orfs_idx]
        self.transcript_rate_logfold = xrate_logfold.loc[orfs_idx]

        self.promoter_sm_occupancy_delta = \
            difference(self.promoter_sm_occupancy)
        self.gene_body_disorganization_delta = \
            difference(self.gene_body_organization)
        self.promoter_disorganization_delta = \
            difference(self.promoter_organization)

        # other deltas
        self.promoter_nuc_occ_delta = \
            difference(self.promoter_nuc_occupancy)
        self.gene_body_nuc_occ_delta = \
            difference(self.gene_body_nuc_occupancy)

        self.orfs = orfs

        self.chromatin_data = self.promoter_sm_occupancy_delta.join(
            self.gene_body_disorganization_delta, 
            lsuffix='_promoter', rsuffix='_gene')

        self.data = self.chromatin_data.join(self.transcript_rate_logfold, how='inner')
        self.xlabels = [
            'Small fragment\noccupancy',
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

