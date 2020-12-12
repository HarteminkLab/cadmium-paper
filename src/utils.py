
import sys
import os
import numpy as np
import subprocess
import contextlib

utils_orfs = None

def time_to_index(time):
    times = [0, 7.5, 15, 30, 60, 120]
    indices = {}
    for i in range(len(times)):
        indices[times[i]] = i
    return indices[time]

def print_fl(val='', end='\n', log=True):

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


def get_util_orfs():

    from src.reference_data import read_sgd_orfs

    global utils_orfs

    if utils_orfs is None:
        utils_orfs = read_sgd_orfs()

    return utils_orfs

def get_gene_named(gene_name, genes=None):

    if genes is None:
        genes = get_util_orfs()

    orf = genes[genes['name'] == gene_name].index.values[0]
    return genes.loc[orf]


def get_gene(orf_name):
    return get_util_orfs().loc[orf_name]


def get_gene_name(orf_name):
    return get_gene(orf_name)['name']
    
def get_orfs(gene_names):

    orfs = get_util_orfs()
    
    # TODO: may need to be optimized
    idxs = []
    for gene_name in gene_names:
        idx = orfs[orfs['name'] == gene_name].index.values[0]
        idxs.append(idx)

    return orfs.loc[idxs]


def get_orf_names(gene_name):
    return get_orfs(gene_name).index.values


def get_orf(gene_name, orfs=None):
    if orfs is None:
        orfs = get_util_orfs()
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


def run_cmd(bashCommand, stdout_file=None):
    if stdout_file is not None:
        with open(stdout_file, 'w') as output:
            process = subprocess.Popen(bashCommand.split(), stdout=output) 
            output, error = process.communicate()
    else:
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


def write_str_to_path(string, path):
    with open(path, 'w') as file:
        file.write(string)

