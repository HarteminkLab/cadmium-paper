
from scipy.stats import norm, uniform
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy.stats import norm, uniform
from scipy.optimize import curve_fit
from src.cross_correlation_kernel import MNaseSeqDensityKernel
from src.transformations import exhaustive_counts
from src.plot_utils import hide_spines    
from src.chromatin import filter_mnase, collect_mnase
from src.utils import print_fl


class KernelFitter:

    def __init__(self, mnase_seq, len_mean, window, kernel_type):
        self.mnase_seq = mnase_seq
        self.kernel_type = kernel_type
        self.window = window
        self.window_2 = self.window/2

        self.extent = [-self.window_2, self.window_2,
                    0, 250]

        (self.narrow_counts, 
         self.pivoted_data) = exhaustive_counts((-self.window_2, self.window_2),
            (self.extent[2], self.extent[3]), data=self.mnase_seq, 
            x_key='mid', y_key='length')

        # normalize by number of reads
        self.narrow_counts.loc[:, 'count'] = self.narrow_counts['count']\
            .astype(float) / len(mnase_seq) * 5e3
        self.pivoted_data.loc[:] = self.pivoted_data\
            .astype(float) / len(mnase_seq) * 2e3

        # length means determined by data
        self.len_mean = len_mean

    def fit(self):

        np.random.seed(123)
        print_fl("Fitting positional distribution...")
        self.fit_position()

        print_fl("\nFitting length distribution...")
        self.fit_length()

    def fit_position(self):
        """Fit positional distribution to a normal with uniform background"""
        Y = self.pivoted_data.sum(axis=0)
            
        # fixed parameters, mean of normal and uniform range
        mean_position = 0

        if self.kernel_type == 'small':
            self.pos_std = np.std(Y)
        else:
            self.pos_std = np.std(Y)

        self.pos_mean = mean_position
        print_fl("Kernel positional mean %.2f and std %.2f" % 
              (self.pos_mean, self.pos_std))

    def fit_length(self):
        Y = self.pivoted_data.sum(axis=1)

        if self.kernel_type == 'small':
            # mirror small fragments distribution from 0-mode to compute standard deviation in length
            data = self.pivoted_data.sum(axis=1).loc[0:self.len_mean]
            mirrored_data = data.sort_index(ascending=False)
            data = np.concatenate([data, mirrored_data])
            self.len_std = np.std(data)#/50.

        elif self.kernel_type == 'nucleosome':
            data = self.pivoted_data.sum(axis=1).loc[self.len_mean:]
            mirrored_data = data.sort_index(ascending=False)
            data = np.concatenate([mirrored_data, data])
            self.len_std = np.std(data)#/500.

        print_fl("Kernel %s length mean %.2f and std %.2f" % 
              (self.kernel_type, self.len_mean, self.len_std), log=True)

    def plot_data(self, ax=plt):
        ax.imshow(self.pivoted_data, origin='lower', cmap='RdYlBu_r',
            extent=self.extent, aspect=1.2)

    def generate_kernel(self):

        len_var = self.len_std**2
        len_mean = self.len_mean
        pos_var = self.pos_std**2

        self.kernel = MNaseSeqDensityKernel(
            mean_length=len_mean, var_length=len_var,
            mean_pos=self.pos_mean, var_pos=pos_var, 
            extent=self.extent)


def compute_triple_kernel(nuc_kernel):

    triple_peak_kernel = MNaseSeqDensityKernel(extent=[-400, 400, 0, 250])

    var = nuc_kernel.var_pos*2

    # smooth gaussian kernel along position axis
    x = np.arange(triple_peak_kernel.extent[0], triple_peak_kernel.extent[1]+1, 1)

    pos_dens = np.array(norm.pdf(x, 0, var**0.5)) + \
               np.array(norm.pdf(x, 170, var**0.5)) + \
               np.array(norm.pdf(x, 340, var**0.5))

    lengths = np.arange(0, 251)
    len_dens = norm.pdf(lengths, nuc_kernel.mean_length, 
        nuc_kernel.var_length**.5)

    # multiply two vectors
    new_mat = len_dens.reshape((len(lengths), 1)) * pos_dens.reshape((1, len(pos_dens)))
    new_mat = new_mat / new_mat.flatten().sum()

    triple_peak_kernel.kernel_mat = new_mat

    return triple_peak_kernel


def compute_nuc_kernel(all_mnase_data, brogaard):

    # Get MNase-seq data @ 0 min for top 1000 nucleosomes
    mnase_seq_0 = filter_mnase(all_mnase_data, time=0.0)
    top_brogaard = brogaard.head(2500)

    brogaard_mnase = collect_mnase(mnase_seq_0, window=200, pos_chr_df=top_brogaard)

    nuc_length_mode = int(brogaard_mnase.length.mode())
    print_fl("Nucleosome length mode: %d" % nuc_length_mode)

    nuc_fitter = KernelFitter(brogaard_mnase, len_mean=nuc_length_mode, window=200, kernel_type='nucleosome')
    nuc_fitter.fit()
    nuc_fitter.generate_kernel()

    return nuc_fitter.kernel

def compute_sm_kernel(all_mnase_data, abf1_sites):

    mnase_seq_0 = filter_mnase(all_mnase_data, time=0.0)
    abf1_mnase = collect_mnase(mnase_seq_0, window=150, 
                                   pos_chr_df=abf1_sites, chrom_key='chr', 
                                   pos_key='mid',
                                   strand='strand')
    
    abf1_length_mode = int(abf1_mnase.length.mode())
    print_fl("Abf1 fragment length mode: %d" % abf1_length_mode)

    sm_fitter = KernelFitter(abf1_mnase, len_mean=abf1_length_mode, window=150, 
        kernel_type='small')
    sm_fitter.fit()
    sm_fitter.generate_kernel()
    return sm_fitter.kernel
