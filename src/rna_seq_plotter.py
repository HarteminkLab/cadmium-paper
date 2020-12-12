
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm
from src import plot_utils


class RNASeqPlotter:

    def __init__(self, pileup, times=[0, 7.5, 15, 30, 60, 120]):
        self.times = times
        self.pileup = pileup
        self.span = None
        self.smooth_kernel = get_smoothing_kernel(100, 10)
        self.set_colors()
        self.custom_times = None

    def set_span_chrom(self, span, chrom):

        pileup = self.pileup

        if not self.span == span or not chrom == self.chrom:
            self.span = int(span[0]), int(span[1])
            self.chrom = chrom
            self.cur_pileup = pileup[(pileup.position >= span[0]) & 
                         (pileup.position < span[1]) & 
                         (pileup.chr == chrom)].sort_values('position')

    def set_colors(self):
        self.reds, self.blues = get_strand_colors(len(self.times))

    def plot(self, ax=None):
        """Plot RNA-seq"""

        xticks_interval = (500, 100) 

        if ax is None:
            fig, ax = plt.subplots(figsize=(24, 2))

        def _plot_pileup(ax, data, color, strand='+'):
            X = data.position
            Y = np.log2(data['pileup']+1)
            Y = np.convolve(Y, self.smooth_kernel, mode='same')
            if strand == '-': Y = Y*-1
            ax.plot(X, Y, color=color, linewidth=2, rasterized=True)

        times = self.times
        selected_pileup = self.cur_pileup

        for i in range(len(times)):

            time = times[i]

            if self.custom_times is not None:
                if time not in self.custom_times: continue

            time_pileup = selected_pileup[selected_pileup.time == time]
            data = time_pileup[(time_pileup.strand == '+')]
            if len(data) > 0:
                _plot_pileup(ax, data, self.blues[i], '+')

            data = time_pileup[(time_pileup.strand == '-')]
            if len(data) > 0:
                _plot_pileup(ax, data, self.reds[i], '-')

        ax.set_xlim(*self.span)

        ax.set_xticks(range(self.span[0], self.span[1], xticks_interval[1]), 
            minor=True)

        xticks = range(self.span[0], self.span[1]+xticks_interval[0], xticks_interval[0])
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(['' for x in xticks])

        ax.set_yticks([-10, 0, 10])
        ax.set_yticklabels([10, 0, 10])

        ax.set_yticks(np.arange(-20, 20, 5), minor=True)
        ax.set_ylabel('Log$_2$ pileup+1', fontsize=24)

        ax.set_ylim(-16, 16)

        plot_utils.format_ticks_font(ax, fontsize=13)


def get_smoothing_kernel(window, std):
    """Set RNA-seq smoothing kernel"""
    # smoothing
    X = np.arange(-1*window/2.0, window/2)
    return norm.pdf(X, 0, std)

def get_strand_colors(num_times=6):
    # colors
    color_offset = 2
    N = num_times + color_offset
    blues = map(lambda x: cm.Blues(float(x)/(N), 1), range(color_offset, N))
    reds = map(lambda x: cm.Reds(float(x)/(N), 1), range(color_offset, N))
    return reds, blues
