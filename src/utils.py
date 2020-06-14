
import sys
import os
import numpy as np
import subprocess
import contextlib


def print_fl(val='', end='\n', log=False):

    contents = str(val) + end

    sys.stdout.write(contents)
    sys.stdout.flush()

    if log:
        from config import LOG_FILE_PATH
        with open(LOG_FILE_PATH, 'a') as logfile:
            logfile.write(contents)

def get_size_obj(obj):

    KB = 1024
    MB = KB*1024
    GB = MB*1024

    size = sys.getsizeof(obj)
    units = "B"

    if size > GB: 
        size = size / GB
        units = "GB"
    elif size > MB: 
        size = size / MB
        units = "MB"
    elif size > KB: 
        size = size / KB
        units = "KB"

    return ("%d %s" % (size, units))


def get_gene_named(gene_name, genes=None):
    from src.reference_data import read_sgd_orfs

    if genes is None:
        genes = read_sgd_orfs()
    orf = genes[genes['name'] == gene_name].index.values[0]
    return genes.loc[orf]


def get_gene(orf_name):
    from src.reference_data import read_sgd_orfs
    return read_sgd_orfs().loc[orf_name]


def get_gene_name(orf_name):
    return get_gene(orf_name)['name']
    
def get_orfs(gene_names):
    from src.reference_data import read_sgd_orfs

    orfs = read_sgd_orfs()
    
    # TODO: may need to be optimized
    idxs = []
    for gene_name in gene_names:
        idx = orfs[orfs['name'] == gene_name].index.values[0]
        idxs.append(idx)

    return orfs.loc[idxs]


def get_orf_names(gene_name):
    return get_orfs(gene_name).index.values


def get_orf(gene_name, orfs=None):
    from src.reference_data import read_sgd_orfs
    if orfs is None:
        orfs = read_sgd_orfs()
    orf_name = orfs[(orfs.index == gene_name) | (
        orfs['name'] == gene_name)].index.values[0]
    return orfs.loc[orf_name]

def get_orf_name(gene_name):
    return get_orf(gene_name).name

def get_std_cutoff(data, std_cutoff=1.5):
    """Filter data by standard deviation 
       above or below the mean"""
    mean, std = np.mean(data), np.std(data)
    cutoff = mean+std*std_cutoff
    if std_cutoff > 0: select = data > cutoff
    else: select = data < cutoff
    return data[select]


def run_cmd(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def mkdirs_safe(directories, log=True):
    for directory in directories:
        mkdir_safe(directory, log=log)

def mkdir_safe(directory, log=True):
    if log: print_fl("Creating directory: %s..." % directory, end='')

    if not os.path.exists(directory):
        os.makedirs(directory)
    elif log:
        print_fl("Directory exists. Skipping.", end='')

    if log: print_fl()

def flip_strand(strand):
    if strand == '+': return '-'
    else: return '+'
