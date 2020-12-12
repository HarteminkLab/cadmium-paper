
import pandas as pd
import numpy as np
import pysam
from timer import Timer
import os
import sys
from transcription import filter_rna_seq
from config import DEBUG_CHROMS, SRA_BIN_DIR
from src.utils import print_fl, run_cmd


def sra_download_convert_bam(write_dir, sra_id, filename):

    prefetch = "%s/prefetch" % SRA_BIN_DIR 
    sam_dump = "%s/sam-dump" % SRA_BIN_DIR

    # prefetch SRA ID
    print_fl("Prefetching %s" % sra_id)
    sra_write_path = "%s/%s.sra" % (write_dir, filename)
    run_cmd("%s %s --output-file %s" % (prefetch, sra_id,  sra_write_path))

    # dump to sam
    sam_write_path = "%s/%s.sam" % (write_dir, filename)
    print_fl("Dumping SAM %s" % sam_write_path)
    run_cmd("%s %s" % (sam_dump, sra_write_path), stdout_file=sam_write_path)

    # convert to bam
    bam_write_path = "%s/%s.bam" % (write_dir, filename)
    print_fl("Converting to BAM %s" % bam_write_path)
    run_cmd("samtools view -b -S %s" % (sam_write_path), stdout_file=bam_write_path)

    # index
    bam_write_path = "%s/%s.bam" % (write_dir, filename)
    bam_index_path = "%s/%s.bam.bai" % (write_dir, filename)
    print_fl("Indexing BAM %s" % bam_index_path)
    run_cmd("samtools index %s %s" % (bam_write_path, bam_index_path))

    # remove large SAM file
    os.remove(sam_write_path)

def read_mnase_set(directory, filenames, 
    source, times = [0, 7.5, 15, 30, 60, 120], debug=False,
    log=False):
    """Read set of MNase-seq data from directory and return pandas DataFrame"""
    
    timer = Timer()
    mnase_seq = pd.DataFrame()
    for i in range(len(times)):
        time = times[i]
        filename = filenames[i]

        bam_path = directory + '/' + filename
        if log: print_fl("Reading %s for time %.1f" % (bam_path, time))
        time_mnase = read_mnase_bam(bam_path, time, debug=debug)
        mnase_seq = mnase_seq.append(time_mnase)

        if log: timer.print_time()

    mnase_seq['source'] = source
    return mnase_seq


def read_mnase_bam(filename, time, debug=False):
    """
    Read mnase data from bam file. Return a pandas dataframe of x start coordinate, 
    x end coordinate, chromosome, fragment length, and sequence. BAM File
    """
    samfile = pysam.AlignmentFile(filename, "rb")

    data = {'start':[], 'length': [], 'stop': [], 'mid': [], 'chr': [], 'time': []}
    for chrom in range(1, 17):

        # debug mode only reads chromosome 1
        if debug and chrom not in DEBUG_CHROMS: continue

        # get chromosome reads
        try:
            itr = samfile.fetch(str(chrom))
        except ValueError:
            itr = samfile.fetch("chr{}".format(_toRoman(chrom)))

        for read in itr:

            # skip second read in pair
            # equivalent to filtering to include only "-f 32" 
            # flag in samtools
            if not read.mate_is_reverse: continue

            length = read.template_length
            start = read.pos+1
            stop = start + length - 1

            data['start'].append(start)
            data['length'].append(length)
            data['stop'].append(stop)
            data['mid'].append(start + length/2)
            data['chr'].append(chrom)
            data['time'].append(time)

    samfile.close()
    df = pd.DataFrame(data=data)


    return df


def read_rna_seq(filename, time, debug=False):
    """Load an individual RNA-seq file and return a dataframe"""

    samfile = pysam.AlignmentFile(filename, "rb")

    data = {'start':[], 'strand': [], 'length': [],
            'chr': [], 'stop': [], 'time': []}
    for chrom in range(1, 17):

        if debug and chrom not in DEBUG_CHROMS: continue

        # get chromosome reads
        try:
            itr = samfile.fetch(str(chrom))
        except ValueError:
            itr = samfile.fetch("chr{}".format(_toRoman(chrom)))

        for read in itr:

            # skip unmapped reads, i.e. -F 4
            if read.is_unmapped: continue

            length = read.reference_length
            position = read.pos+1 # first base begins at 1
            strand = '-'
            if read.is_reverse: strand = '+'

            data['start'].append(position)
            data['strand'].append(strand)
            data['chr'].append(chrom)
            data['time'].append(time)
            data['length'].append(length)
            data['stop'].append(position + length) # inclusive stop nucleotide

    samfile.close()
    df = pd.DataFrame(data=data)
    return df


def read_rna_seq_set(directory, 
    filenames, source, times=[0, 7.5, 15, 30, 60, 120],
    debug=False):
    """Load RNA-seq set of files and return a dataframe"""

    rna_seq = pd.DataFrame()

    for i in range(len(times)):
        filename = filenames[i]
        time = times[i]
        cur_rna_seq = read_rna_seq(directory + '/' + filename, time, debug=debug)
        rna_seq = rna_seq.append(cur_rna_seq)

    rna_seq['source'] = source

    return rna_seq

def _toRoman(number):
    try:
        return {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII',
         8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII', 13: 'XIII', 
         14: 'XIV', 15: 'XV', 16: 'XVI'}[number]
    except KeyError:
        return -1
    

def _fromRoman(roman):
    try:
        return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, 
            "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, 
            "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, 
            "XV": 15, "XVI": 16}[roman]
    except KeyError:
        return -1
