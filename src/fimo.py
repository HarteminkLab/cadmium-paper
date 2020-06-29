
from peaks_sequences import peaks_sequences, create_fna
import pandas as pd
import numpy as np
from utils import run_cmd


class FIMO:

    def __init__(self, out_dir=None, fna_file=None):
        self.config_fimo()

        if out_dir is not None:
            self.OUT_DIR = out_dir

        if fna_file is not None:
            self.FNA_FILE = fna_file

    def config_fimo(self):
        
        from config import FIMO_PATH
        from config import FIMO_GENOME_FSA
        from config import MACISAAC_MEME_PATH

        self.FNA_FILE = 'output/temp_fna.fna'
        self.FIMO = FIMO_PATH
        self.genome = read_yeast_genome(FIMO_GENOME_FSA)
        self.macisaac_path = MACISAAC_MEME_PATH
        self.OUT_DIR = 'output/temp_fimo/'
        self.search_width = 0
        self.search_width_2 = self.search_width/2

    def find_motif_matches(self, peaks, name_key='bin', db_path=None):

        FNA_FILE = self.FNA_FILE
        peaks = peaks.copy()

        # get genome sequences for each peak
        sequences = peaks_sequences(peaks, self.genome)
        peaks['sequence'] = sequences['sequence'].values

        # create FNA file for FIMO
        create_fna(peaks, FNA_FILE, name_key=name_key)

        # run FIMO
        res_m = self.run_fimo('macisaac_yeast', db_path=db_path) 
        res = res_m

        # cleanup fna file
        out, err = run_cmd("rm -rf {}".format(self.FNA_FILE))

        return res


    def run_fimo(self, tb_db='macisaac_yeast', db_path=None):

        OUT_DIR = self.OUT_DIR

        if db_path is not None:
            TF_DB = db_path
        elif tb_db == 'YEASTRACT_20130918':
            TF_DB = self.YEASTRACT_path
        elif tb_db == 'macisaac_yeast':
            TF_DB = self.macisaac_path
        else:
            raise ValueError('Unknown TF DB: {}'.format(tb_db))

        # Run FIMO
        out, err = run_cmd("rm -rf {}".format(OUT_DIR))
        out, err = run_cmd("{} --verbosity 1 --o {} {} {}".format(self.FIMO, 
            OUT_DIR, TF_DB, self.FNA_FILE))

        # Read result and clean up
        res = pd.read_csv('{}/fimo.tsv'.format(OUT_DIR), delimiter='\t')
        res = res[~pd.isna(res['sequence_name'])].copy().reset_index(drop=True)

        out, err = run_cmd("rm -rf {}".format(OUT_DIR))

        res['tf'] = res['motif_id']

        self.matches = res

        return res.drop_duplicates().reset_index(drop=True)


def read_yeast_genome(input_path):
    """
    Read yeast genome data for looking up sequences at various locations in genome.
    """

    from Bio import SeqIO
    from src.read_bam import _fromRoman

    NUM_CHROM = 16

    sequences = {}
    
    with open(input_path, 'rb') as input_file:
        fasta_sequences = SeqIO.parse(input_file,'fasta')

        for fasta in fasta_sequences:
            chrom = fasta.description.split(' ')[-1].replace('[chromosome=', '').replace(']', '')
            chrom = _fromRoman(chrom)

            name, sequence = fasta.id, fasta.seq
            if chrom > 0:
                sequences[chrom] = sequence

    return sequences



def find_motif(fimo, tf, chrom, window):

    peaks = pd.DataFrame({'name': ['temp']})
    peaks['bin'] = '%d_%d' % window
    peaks['start'] = window[0]
    peaks['end'] = window[1]
    peaks['sequence_name'] = '%d_%d' % (chrom, window[0])
    peaks['chr'] = chrom

    results = fimo.find_motif_matches(peaks)
    results['motif_mid'] = ((results.start+results.stop)/2. + window[0])
    
    if tf is not None:
        results = results[results.tf == tf]

    return results


def find_reg_motifs(fimo, orf, where='TSS', window=(-200, 0)):

    orf = orf.copy()

    if where == 'TSS':
        TSS = orf.TSS 

        if orf.strand == '+':
            if np.isnan(TSS): TSS = orf.start
            search_window = TSS + window[0], TSS + window[1]
        else:
            if np.isnan(TSS): TSS = orf.stop
            search_window = TSS - window[1], TSS - window[0]
    elif where == 'PAS':

        PAS = orf.PAS

        if orf.strand == '+':
            search_window = PAS + window[0], PAS + window[1]
        else:
            search_window = PAS - window[1], PAS - window[0]
            
    search_window = search_window
    met32_motif_loc = find_motif(fimo, None, orf.chr, search_window)
    met32_motif_loc = met32_motif_loc#[['tf', 'motif_mid']]
    met32_motif_loc['target'] = orf.name
    met32_motif_loc['chr'] = orf.chr
    met32_motif_loc['where'] = where
    
    return met32_motif_loc


def find_relevant_gene_motifs(fimo, orfs):
    from src.met4 import all_genes
    from src.gene_list import get_gene_list
    from src.utils import get_gene_named
    from config import mnase_dir
    from src.fimo import find_reg_motifs

    TSS_genes = all_genes() + get_gene_list()
    PAS_genes = ['MET31']

    found_motifs = pd.DataFrame()

    for gene_name in PAS_genes:
        orf = get_gene_named(gene_name, orfs)
        cur_found = find_reg_motifs(fimo, orf, where='PAS', window=(0, 400))
        found_motifs = found_motifs.append(cur_found)
        
    for gene_name in TSS_genes:
        orf = get_gene_named(gene_name, orfs)
        cur_found = find_reg_motifs(fimo, orf, where='TSS', window=(-400, 0))
        found_motifs = found_motifs.append(cur_found)

    found_motifs = found_motifs[found_motifs.tf != 'AZF1'].reset_index(drop=True)
    return found_motifs
