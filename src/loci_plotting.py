
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ViolinPlotter:

    def __init__(self, orfs, cross_correlation):
        self.orfs = orfs
        self.cross_correlation = cross_correlation
        self.times = [0.0, 7.5, 15, 30, 60, 120]

    def set_gene(self, gene_name):
        gene = self.orfs[self.orfs.name == gene_name].reset_index().loc[0]
        self.orf_cc = self.cross_correlation[self.cross_correlation.gene == gene.orf_name]
        self.gene = gene

    def plot_chrom_violins(self):
        nuc_scale = 5
        sm_scale = 3
        times = self.times
        data = self.orf_cc

        fig = plt.figure(figsize=(8, 4))
        for i in range(len(times)):
            time = times[i]
            d = data[data.time == time]
            zeros = np.zeros(len(d)) - i
            y_upper = d.cross*nuc_scale - i
            y_lower = -d.cross*nuc_scale - i
            plt.fill_between(d.translated_mid, y_lower, y_upper, where=y_upper>zeros, 
                             facecolor='#04B3BC', linewidth=1, edgecolor='white')
            
            y_upper = d.cross*sm_scale - i
            y_lower = -d.cross*sm_scale - i
            plt.fill_between(d.translated_mid, y_lower, y_upper, where=y_upper<=zeros, 
                             facecolor='#F4A658', linewidth=1, edgecolor='white')

        plt.yticks(np.arange(0, -6, -1), times)
        plt.xlim(-1000, 1000)
        plt.axvline(x=0, c='black', linestyle='dotted', linewidth=0.5)
        plt.ylabel('Time (min)')

        return fig
