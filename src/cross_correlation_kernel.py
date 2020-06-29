

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json
from scipy.stats import multivariate_normal, norm
from src.transformations import exhaustive_counts
from src.plot_utils import hide_spines

class MNaseSeqDensityKernel:

    def __init__(self, extent=[-100, 100, 0, 250], 
        mean_length=163, var_length=500,
        mean_pos=0, var_pos=75, filepath=None):

        if filepath is not None:
            with open(filepath, 'r') as fil:
                json_kernel = json.loads(fil.read())
                mean_pos = json_kernel['mean_pos']
                var_pos = json_kernel['var_pos']
                mean_length = json_kernel['mean_length']
                var_length = json_kernel['var_length']
                extent = json_kernel['extent']

        self.mean_pos = mean_pos
        self.mean_length = mean_length
        self.var_pos = var_pos
        self.var_length = var_length
        self.extent = extent
        self.compute_kernel()

    def save_kernel(self, filepath):
        """save kernel to disk"""
        json_kernel = {'mean_pos': self.mean_pos,
            'var_pos': self.var_pos,
            'mean_length': self.mean_length,
            'var_length': self.var_length,
            'extent': self.extent}
        with open(filepath, 'w') as fil:
            fil.write(json.dumps(json_kernel))

    def compute_kernel(self):
        
        self.kernel_dist = multivariate_normal([self.mean_pos, self.mean_length],
            np.matrix([[self.var_pos, 0], [0, self.var_length]]))

        extent = self.extent
        self.k_width_2 = (extent[1] - extent[0])/2
        

        (self.narrow_counts, 
         self.pivoted_data) = exhaustive_counts( 
            (extent[0], extent[1]),
            (extent[2], extent[3]), x_key='mid', y_key='length')

        # create n, m, 2 matrix of position and length values
        lengths = self.pivoted_data.index.values
        pos = self.pivoted_data.columns.values
        data = np.array(self.pivoted_data)
        data = np.stack([data, data], axis=2)
        data[:, :, 1] = np.vstack([lengths]*len(pos)).T
        data[:, :, 0] = np.vstack([pos]*len(lengths))
        self.pos_len_dfs = data

        # to feed into pdf
        self.kernel_mat = self.kernel_dist.pdf(data)

    def plot(self, ax=None):
        if ax is None: 
            fig, ax = plt.subplots(1, 1, figsize=(3, 5))

        ax.imshow(self.kernel_mat, extent=self.extent, 
                origin='lower', cmap='RdYlBu_r', aspect=1.2)

    def plot_kernel(self, kernel_type):

        if kernel_type == 'nucleosome':
            fig, axs = plt.subplots(2, 2, figsize=(4, 5.35))
        elif kernel_type == 'triple':
            fig, axs = plt.subplots(2, 2, figsize=(8.65, 5.35))
        elif kernel_type == 'small':
            fig, axs = plt.subplots(2, 2, figsize=(4,6.9))
        else:
            raise ValueError("Invalid kernel_type")

        fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.95])
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        ax1, ax2, ax3, ax4 = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
        ax2.set_xticks([])
        ax2.set_yticks([])

        hide_spines(ax2)
        hide_spines(ax4)
        hide_spines(ax1)

        self.plot_position(ax1, kernel_type)
        self.plot(ax3)
        self.plot_length(ax4, kernel_type)

        ax3.set_xlabel("Position (bp)")
        ax3.set_ylabel("Length (bp)")

        # asymmetric kernel, plot only relevant region
        if kernel_type == 'triple':
            ax3.set_xlim(-100, 400)
            ax1.set_xlim(-100, 400)

    def plot_position(self, ax, kernel_type):

        window_2 = (self.extent[1] - self.extent[0])/2.
        X = np.arange(-window_2, window_2+1)
        Y = self.kernel_mat.sum(axis=0)
        # Y = Y / np.sum(Y)*5.

        # components
        # norm_Y = norm.pdf(X, loc=0, scale=self.var_pos**2)
        ax.fill(X, Y, color='#a0a0a0')

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(X.min(), X.max())

        ax.set_ylim(0, Y.max() * 3.)


    def plot_length(self, ax, kernel_type):

        X = np.arange(0, 251)
        Y = self.kernel_mat.sum(axis=1)

        ax.fill_betweenx(X, Y, x2=0, where=X>0, color='#a0a0a0')

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 250)

        ax.set_xlim(0, Y.max() * 3.0)


