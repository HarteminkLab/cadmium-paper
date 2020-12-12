
import sys
sys.path.append('.')

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.chromatin import filter_mnase
from scipy.stats import norm
from src.plot_utils import apply_global_settings
from config import *
from src.reference_data import read_sgd_orfs, read_park_TSS_PAS
from src.timer import Timer


def plot_ends_heatmap(
    orf_0_nuc_mid_counts, orf_120_nuc_mid_counts,
    orf_0_nuc_start_counts, orf_120_nuc_start_counts,
    orf_0_nuc_stop_counts, orf_120_nuc_stop_counts, 
    head=None, tail=None):

    apply_global_settings(titlepad=10)

    mids = [orf_0_nuc_mid_counts, orf_120_nuc_mid_counts]
    starts = [orf_0_nuc_start_counts, orf_120_nuc_start_counts]
    ends = [orf_0_nuc_stop_counts, orf_120_nuc_stop_counts]
    nuc_groups = [starts, mids, ends]
    names = ['Left', 'Middle', 'Right']

    fig = plt.figure(figsize=(6, 5))

    grid_size = (3, 3)
    rows, cols = 3, 3

    ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=1, rowspan=2)
    ax1 = plt.subplot2grid(grid_size, (0, 1), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid(grid_size, (0, 2), colspan=1, rowspan=2)
    axs = [ax0, ax1, ax2]
    origins = [-50, 0, 50]

    ax0 = plt.subplot2grid(grid_size, (2, 0), colspan=1, rowspan=1)
    ax1 = plt.subplot2grid(grid_size, (2, 1), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid(grid_size, (2, 2), colspan=1, rowspan=1)
    axs2 = [ax0, ax1, ax2]

    fig.tight_layout(rect=[0.075, 0.1, 0.95, 0.945])
    plt.subplots_adjust(hspace=0.1, wspace=0.3)

    (ax1, ax2, ax3) = axs

    for i in range(len(axs)):
        ax = axs[i]

        group_120 = nuc_groups[i][1]
        group_0 = nuc_groups[i][0]
        
        data = group_120 - group_0
        
        if head is not None:
            data = data.head(head)
        elif tail is not None:
            data = data.tail(tail)

        ax.imshow(data, vmin=-5, vmax=5, aspect=300./len(data),
                   cmap='RdBu_r', extent=[-500, 500, 0, len(data)])
        ax.set_xlim(-50+origins[i], 150+origins[i])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(names[i])
        ax.axvline(0, color='black', linestyle='dashed', linewidth=1)

    plot_ends_comparison(axs2,
        orf_0_nuc_mid_counts, orf_120_nuc_mid_counts,
        orf_0_nuc_start_counts, orf_120_nuc_start_counts,
        orf_0_nuc_stop_counts, orf_120_nuc_stop_counts, head=head, tail=tail)

    if head is not None:
        topbot = "downstream"
        headtail = head
    else:
        topbot = "upstream"
        headtail = tail

    plt.suptitle("Greatest %d %s nucleosome\nfragments shift, 0-120 min" % 
        (headtail, topbot))


def plot_ends_comparison(axs,
    orf_0_nuc_mid_counts, orf_120_nuc_mid_counts,
    orf_0_nuc_start_counts, orf_120_nuc_start_counts,
    orf_0_nuc_stop_counts, orf_120_nuc_stop_counts, 
    head=None, tail=None):
   
    (ax1, ax2, ax3) = axs
    origins = [-50, 0, 50]

    colors = [plt.get_cmap('tab10')(0), plt.get_cmap('tab10')(3)]

    def plot_ax_nucs(ax, orf_nuc_mid_counts, color):
        data = orf_nuc_mid_counts.sum()
        data = data[np.arange(data.index.min(), data.index.max(), 5)]
        
        x = np.arange(-5, 5)
        kernel = norm.pdf(x, loc=0, scale=0.5)
        kernel = kernel/kernel.sum()
        y = np.convolve(kernel, data.values, mode='same')
        
        ax.plot(data.index, y, color=color)

    mids = [orf_0_nuc_mid_counts, orf_120_nuc_mid_counts]
    starts = [orf_0_nuc_start_counts, orf_120_nuc_start_counts]
    ends = [orf_0_nuc_stop_counts, orf_120_nuc_stop_counts]

    nuc_groups = [starts, mids, ends]

    for i in range(3):
        ax = axs[i]
        
        for j in range(2):
            data = nuc_groups[i][j]
            if head is not None:
                data = data.head(head)
            elif tail is not None:
                data = data.tail(tail)

            plot_ax_nucs(ax, data, colors[j])
        
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_xticks(np.arange(-200, 250, 50))
        ax.set_xlim(-50+origins[i], 150+origins[i])
        ax.set_ylim(0, 1800)
        ax.set_yticks([])
        ax.axvline(0, color='black', linestyle='dashed', linewidth=1)


def get_binned_counts(orf_mnase, key):
    mid_counts = orf_mnase.groupby(key).count()[['chr']].rename(columns={'chr':'count'})

    all_pos = np.arange(-500, 501)
    all_pos_df = pd.DataFrame({'position':all_pos}).set_index('position')

    mid_counts =all_pos_df.join(mid_counts, how='left').fillna(0)
    return mid_counts


def get_p1_mnase_by_TSS(mnase_data, p1_shift, orfs, time):
    timer = Timer()

    p1_shift = p1_shift.loc[p1_shift.index.isin(orfs.index.values)]
    
    # sort by chromosome and start, for MNase-seq caching speedup
    p1_shift_sorted_idx = p1_shift[[]].join(orfs[['chr', 'start']]).sort_values(['chr', 
        'start']).index.values

    all_pos = np.arange(-500, 501)

    orf_nuc_mid_counts = p1_shift[[]].copy()
    orf_nuc_start_counts = p1_shift[[]].copy()
    orf_nuc_stop_counts = p1_shift[[]].copy()

    for pos in all_pos:
        orf_nuc_mid_counts[pos] = 0
        orf_nuc_start_counts[pos] = 0
        orf_nuc_stop_counts[pos] = 0

    mnase_data = mnase_data[mnase_data.time == time]

    i = 0
    # for each +1 nucleosome, sort by shiftedness
    # get mnase_fragments f
    for orf_name, row in p1_shift.loc[p1_shift_sorted_idx].iterrows():

        orf = orfs.loc[orf_name]

        span = orf.TSS-500, orf.TSS+500
        chrom = orf.chr

        orf_nuc_mnase = filter_mnase(mnase_data, start=span[0], end=span[1], chrom=chrom, translate_origin=orf.TSS, 
                                 flip=(orf.strand == '-'), length_select=(144, 174), sample=time)

        mid_counts = get_binned_counts(orf_nuc_mnase, 'mid')
        start_counts = get_binned_counts(orf_nuc_mnase, 'start')
        stop_counts = get_binned_counts(orf_nuc_mnase, 'stop')

        n = len(mid_counts)
        orf_nuc_mid_counts.loc[orf_name, :] = mid_counts.values.reshape(n)
        orf_nuc_start_counts.loc[orf_name, :] = start_counts.values.reshape(n)
        orf_nuc_stop_counts.loc[orf_name] = stop_counts.values.reshape(n)

        # get mnase-seq at this orf's TSS
        # get the counts of the start, stop, and mids of nucleosome sized fragments

        if i % 400 == 0:
            print("%d/%d - %s" % (i, len(p1_shift), timer.get_time()))

        i += 1

    return (orf_nuc_mid_counts, orf_nuc_start_counts, orf_nuc_stop_counts)


def main():

    print_fl("Loading MNase data")
    mnase_data = pd.read_hdf(mnase_seq_path, 'mnase_data')
    print_fl("Done.")

    called_p123_path = '%s/mnase_seq/called_orf_p123_nucleosomes_sense.csv' % OUTPUT_DIR
    called_nuc_path = '%s/mnase_seq/called_orf_nucleosomes_sense.csv' % OUTPUT_DIR
    called_nucs = pd.read_csv(called_nuc_path)
    called_p123 = pd.read_csv(called_p123_path)

    p1_nucs_links = called_p123['+1']
    p1_nucs = called_nucs[called_nucs.link.isin(p1_nucs_links)]

    from src.chromatin_metrics_data import ChromatinDataStore

    datastore = ChromatinDataStore()

    p1_shift = datastore.p1_shift.sort_values(120.0, ascending=False)

    orfs = read_sgd_orfs()
    TSSes = read_park_TSS_PAS()
    orfs = orfs.join(TSSes[['TSS']])

    print_fl("Getting MNase for p1")
    (orf_0_nuc_mid_counts, 
     orf_0_nuc_start_counts, 
     orf_0_nuc_stop_counts) = get_p1_mnase_by_TSS(mnase_data, p1_shift, orfs, 0)

    (orf_120_nuc_mid_counts, 
     orf_120_nuc_start_counts, 
     orf_120_nuc_stop_counts) = get_p1_mnase_by_TSS(mnase_data, p1_shift, orfs, 120)
    print_fl("Done.")

    # plot heatmaps and save
    plot_ends_heatmap(
        orf_0_nuc_mid_counts, orf_120_nuc_mid_counts,
        orf_0_nuc_start_counts, orf_120_nuc_start_counts,
        orf_0_nuc_stop_counts, orf_120_nuc_stop_counts, head=500)
    plt.savefig("%s/top_500_shift_heatmap.pdf" % misc_figures_dir)
    plt.savefig("%s/top_500_shift_heatmap.png" % misc_figures_dir)

    plot_ends_heatmap(
        orf_0_nuc_mid_counts, orf_120_nuc_mid_counts,
        orf_0_nuc_start_counts, orf_120_nuc_start_counts,
        orf_0_nuc_stop_counts, orf_120_nuc_stop_counts, tail=500)
    plt.savefig("%s/bottom_500_shift_heatmap.pdf" % misc_figures_dir)
    plt.savefig("%s/bottom_500_shift_heatmap.png" % misc_figures_dir)
    print("Saved plots to %s" % misc_figures_dir)


if __name__ == '__main__':
    main()
