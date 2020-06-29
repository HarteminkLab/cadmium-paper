
from src.read_bam import _fromRoman
import pandas as pd
import numpy as np



def all_orfs_TSS_PAS():
    all_orfs = read_sgd_orfs()
    all_orfs = all_orfs.join(read_park_TSS_PAS()[['TSS', 'PAS']])
    return all_orfs


def parse_roman_chr(series):
    return series.str.replace('chr', '').apply(_fromRoman)


def read_park_TSS_PAS():

    TSS_filename = 'data/Park_2014_TSS_V64.gff'
    PAS_filename = 'data/Park_2014_PAS_V64.gff'

    def _read_park_gff(filename, key):
        park_data = pd.read_csv(filename, sep='\t', skiprows=3)
        columns = park_data.columns[[0, 2, 3]] # relevant columns
        park_data = park_data[columns].copy()

        # cleanup
        park_data.columns = ['chr', 'orf_name', key]
        park_data.chr = parse_roman_chr(park_data.chr)
        park_data.orf_name = park_data.orf_name.str.replace('_%s' % key, '')
        park_data[key] = park_data[key].astype(int)

        return park_data.set_index('orf_name')

    TSS = _read_park_gff(TSS_filename, 'TSS')

    # manually annotated/adjusted TSSs for vignettes
    TSS.loc['YBR072W', 'TSS'] = 381753
    TSS.loc['YDR253C', 'TSS'] = 964767
    TSS.loc['YBR294W', 'TSS'] = 789000
    TSS.loc['YLR092W', 'TSS'] = 323500

    PAS = _read_park_gff(PAS_filename, 'PAS')
    data = TSS[['TSS']].join(PAS[['PAS']])

    data['manually_curated'] = False
    data.loc['YBR072W', 'manually_curated'] = True
    data.loc['YDR253C', 'manually_curated'] = True
    data.loc['YBR294W', 'manually_curated'] = True
    data.loc['YLR092W', 'manually_curated'] = True

    return data


def read_brogaard_nucleosomes():
    brogaard = pd.read_csv('data/Brogaard_nuc_positions.sacCer3.top2000.tsv', sep='\t',
        names=['chromosome',  'position', 'NCP_score', 'NCP_score/noise_ratio'])
    brogaard.chromosome = parse_roman_chr(brogaard.chromosome)
    return brogaard


def read_macisaac_abf1_sites():
    sites = pd.read_csv('data/p005_c2.sacCer3.gff.txt', sep='\t',
               names=range(9))
    sites = sites[sites.columns[[0, 3, 4, 6, 8]]].copy()
    sites.columns = ['chr','start','stop','strand','TF']

    sites.chr = parse_roman_chr(sites.chr)
    sites.TF = sites.TF.str.replace(';', '').str.replace('Site ', '')
    sites = sites[sites.TF == 'ABF1'].copy()
    sites['mid'] = ((sites['stop'] + sites['start'])/2).astype(int)

    return sites



def read_sgd():
    """Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

    filename = 'data/saccharomyces_cerevisiae_R64-1-1_20110208_no_fasta.gff'
    data = pd.read_csv(filename, sep='\t', skiprows=19, 
                              names=["chr", "source", "cat", "start", "stop", ".", 
                              "strand", "", "desc"])
    data = data[data.columns[[0, 2, 3, 4, 6, 8]]]
    data.columns = ["chr", "cat", "start", "stop", "strand", "desc"]
    data.chr = parse_roman_chr(data.chr)

    return data


def extract_desc_val(data, key):
    """Extract the values in the description field of SGD"""
    def _extract_desc_val_row(orf_row, key):
        """Parse the description of a orf_row in sgd gff, extract `keys` and return 
        as a series. Can be used to return as a df when called with apply"""

        des_map = {}
        for entry in orf_row.desc.split(';'):
            k, val = tuple(entry.split('='))
            if k == key:
                return val

        return None

    # parse description column to extract relevant columns
    vals = data.apply(lambda row: _extract_desc_val_row(row, key=key),
        axis=1)

    return vals


def read_sgd_orfs():

    from pandas import Series

    data = read_sgd()
    orfs = data[data['cat'] == 'gene'].copy()
    orfs['orf_name'] = None
    orfs['orf_class'] = None

    # chromosomal orfs
    orfs = orfs[orfs.chr > 0]

    orf_names = extract_desc_val(orfs, 'ID') 
    orf_classes = extract_desc_val(orfs, 'orf_classification')
    names = extract_desc_val(orfs, 'gene') 
    ontology = extract_desc_val(orfs, 'Ontology_term') 

    orfs['orf_name'] = orf_names
    orfs['orf_class'] = orf_classes

    # set name if it exists, or orf_name by default
    orfs['name'] = names
    orfs.loc[orfs.name.isna(), 'name'] = orfs[orfs.name.isna()]['orf_name']
    orfs['ontology'] = ontology

    orfs['length'] = orfs['stop'] - orfs['start'] + 1
    orfs = orfs.set_index('orf_name')[[
        'name','chr','start','stop','length','strand','orf_class', 'ontology']]

    return orfs


def read_sgd_orf_introns():
    """Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

    genes = read_sgd_orfs()
    data = read_sgd()

    keep_classes = ['intron', 'five_prime_UTR_intron', 'CDS']
    introns_CDSs = data[data.cat.isin(keep_classes)].copy()

    # rename 5' class to intron
    introns_CDSs.loc[(introns_CDSs['cat'] == 'five_prime_UTR_intron'), 'cat'] = 'intron'
    introns_CDSs['parent'] = extract_desc_val(introns_CDSs, 'Parent') 

    # relevant genes
    introns_CDSs = introns_CDSs[introns_CDSs.parent.isin(genes.index.values)]

    return introns_CDSs[['cat', 'start', 'stop', 'parent']].copy().reset_index(drop=True)


def load_park_orf_transcript_boundaries():

    orfs = read_sgd_orfs()
    park_TSS_PAS = read_park_TSS_PAS()

    annot_tr_bounds = park_TSS_PAS.join(orfs[['chr', 'strand', 'start', 'stop']], how='outer')

    watson = annot_tr_bounds.strand == '+'
    has_TSS = ~annot_tr_bounds.TSS.isna()
    has_PAS = ~annot_tr_bounds.PAS.isna()

    annot_tr_bounds.loc[(watson & has_TSS), 'transcript_start'] = annot_tr_bounds.TSS
    annot_tr_bounds.loc[(watson & ~has_TSS), 'transcript_start'] = annot_tr_bounds.start
    annot_tr_bounds.loc[(watson & has_PAS), 'transcript_stop'] = annot_tr_bounds.PAS
    annot_tr_bounds.loc[(watson & ~has_PAS), 'transcript_stop'] = annot_tr_bounds.stop

    annot_tr_bounds.loc[(~watson & has_TSS), 'transcript_stop'] = annot_tr_bounds.TSS
    annot_tr_bounds.loc[(~watson & ~has_TSS), 'transcript_stop'] = annot_tr_bounds.stop
    annot_tr_bounds.loc[(~watson & has_PAS), 'transcript_start'] = annot_tr_bounds.PAS
    annot_tr_bounds.loc[(~watson & ~has_PAS), 'transcript_start'] = annot_tr_bounds.start

    annot_tr_bounds.transcript_start = annot_tr_bounds.transcript_start.astype(int)
    annot_tr_bounds.transcript_stop = annot_tr_bounds.transcript_stop.astype(int)

    return annot_tr_bounds[['chr', 'strand', 'transcript_start', 'transcript_stop']]
