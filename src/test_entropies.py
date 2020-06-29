
from config import *
from src.entropy import calc_entropy
from src.utils import get_orf
from src.cross_correlation_kernel import MNaseSeqDensityKernel
from src.cross_correlation import calculate_cross_correlation_orf
import numpy as np 
from src.typhoon import TyphoonPlotter
from src.reference_data import read_park_TSS_PAS
import matplotlib.pyplot as plt
from src.entropy import calc_entropy
from src.kernel_fitter import compute_triple_kernel


class TestEntropyKernels:

    def __init__(self):

        park_TSS_PAS = read_park_TSS_PAS()
        orfs = paper_orfs#.join(park_TSS_PAS[['PAS']])
        self.typhoon_plotter = TyphoonPlotter(mnase_path=mnase_seq_path,
                                 rna_seq_pileup_path=pileup_path,
                                 orfs=orfs)
        self.sm_kernel = MNaseSeqDensityKernel(filepath=sm_kernel_path)
        self.nuc_kernel = MNaseSeqDensityKernel(filepath=nuc_kernel_path)
        self.triple_kernel = compute_triple_kernel(self.nuc_kernel)
        self.kernel_type = 'nucleosomal'

    def set_erange(self, e_range):
        self.e_range = e_range

    def set_gene(self, gene_name):

        orf = get_orf(gene_name, paper_orfs)
        self.orf = orf
        self.gene_name = gene_name
        mnase = self.typhoon_plotter.all_mnase_data

        # default cross correlation
        cur_wide_counts_df, cur_cc = calculate_cross_correlation_orf(self.orf, 
            mnase, 3000, self.nuc_kernel, self.sm_kernel, self.triple_kernel)
        self.cur_cc = cur_cc
        self.cur_wide_counts_df = cur_wide_counts_df

        times = self.typhoon_plotter.times
        orf_entropies = []
        for i in range(len(times)):
            time = times[i]
            orf_cc = cur_cc.loc[self.kernel_type].loc[orf.name].loc[time]
            cur_e = calc_entropy(orf_cc[np.arange(self.e_range[0], self.e_range[1]+1)])
            orf_entropies.append(cur_e)
        self.orf_entropies = orf_entropies

    def plot_typhoon(self, write_path):

        orf = self.orf
        if orf.strand == '+':
            custom_highlight_regions = [
                (orf.TSS + self.e_range[0], orf.TSS + self.e_range[1])]
        else:
            custom_highlight_regions = [
                (orf.TSS - self.e_range[1], orf.TSS - self.e_range[0])]

        fig, time_axs, _ = self.typhoon_plotter.plot_gene(self.gene_name, 
            custom_highlight_regions=custom_highlight_regions)
        times = self.typhoon_plotter.times
        
        for i in range(len(times)):
            ax = time_axs[i]
            time = times[i]
            orf_cc = self.cur_cc.loc[self.kernel_type].loc[orf.name]
            ax.fill_between(orf_cc.columns + orf.TSS, orf_cc.loc[time]*250, 
                color='#33a89b')
            ax.set_ylim(-10, 250)

        plt.savefig(write_path, dpi=150)
        plt.close(fig)

    def plot_entropies(self, write_path):
        
        fig = plt.figure(figsize=(4, 3))
        mnase = self.typhoon_plotter.all_mnase_data
        times = self.typhoon_plotter.times
        orf = self.orf

        plt.plot(self.orf_entropies)
        _ = plt.xticks(np.arange(6), times)
        plt.ylim(7.5, 9)
        plt.title(self.gene_name)

        plt.savefig(write_path)
        plt.close(fig)
