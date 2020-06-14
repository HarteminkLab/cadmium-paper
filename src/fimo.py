
from peaks_sequences import peaks_sequences, create_fna
import pandas
import numpy as np
from utils import run_cmd


class FIMO:

    def __init__(self, out_dir=None, fna_file=None):
        self.configure_local()
        # self.configure_cluster()

        if out_dir is not None:
            self.OUT_DIR = out_dir

        if fna_file is not None:
            self.FNA_FILE = fna_file

    def configure_local(self):
        self.FNA_FILE = 'output/temp_fna.fna'
        self.FIMO = '/Users/trung/meme/bin/fimo'
        self.genome = read_yeast_genome("/Users/trung/Documents/projects/data/sacCer2_reference/genome/")
        self.YEASTRACT_path = '/Users/trung/Documents/data/motif_databases/YEAST/YEASTRACT_20130918.meme'
        self.macisaac_path = '/Users/trung/Documents/data/motif_databases/YEAST/macisaac_yeast.v1.meme'
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
        # res_y = self.run_fimo('YEASTRACT_20130918', db_path=db_path) 
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
        res = pandas.read_csv('{}/fimo.tsv'.format(OUT_DIR), delimiter='\t')
        res = res[~pandas.isna(res['sequence_name'])].copy().reset_index(drop=True)

        out, err = run_cmd("rm -rf {}".format(OUT_DIR))

        res['tf'] = res['motif_id']

        self.matches = res

        return res.drop_duplicates().reset_index(drop=True)


def read_yeast_genome(data_dir):
    """
    Read yeast genome data for looking up sequences at various locations in genome.
    """

    from Bio import SeqIO

    NUM_CHROM = 16

    sequences = {}
    for chrom in range(1, NUM_CHROM+1):

        input_path = data_dir + "/chr{:02d}.fsa".format(chrom)

        with open(input_path, 'rb') as input_file:
            fasta_sequences = SeqIO.parse(input_file,'fasta')

            for fasta in fasta_sequences:
                name, sequence = fasta.id, fasta.seq

            sequences[chrom] = sequence

    return sequences
