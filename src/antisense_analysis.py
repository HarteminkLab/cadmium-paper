

import numpy as np
import matplotlib.pyplot as plt
from src.plot_utils import apply_global_settings, hide_spines, plot_density
from src.plot_utils import plot_violin, plot_density_scatter, plot_rect
from src.colors import parula
from src.datasets import read_orfs_data
import matplotlib.patheffects as path_effects
from config import rna_dir


def plot_association(data, key, name, color):
    apply_global_settings(titlepad=10)

    prom_data = data.sort_values(key)

    colors = plt.get_cmap('tab10')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    ax1.scatter(prom_data[key], np.arange(len(prom_data)), s=10, color=color)
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(0, len(prom_data))
    ax1.axvline(0, color='black', linestyle='solid', linewidth=1.5, zorder=1)
    ax1.set_yticks([])
    ax1.set_xlabel('$\\Delta$ %s z-score' % (name[0:1].upper() + name[1:]))
    ax1.axvline(1, color='red', linestyle='solid', linewidth=1.5, zorder=1, alpha=0.5)

    num_q = 10
    x = np.arange(len(prom_data))
    n_q = len(prom_data)/num_q

    for q in range(num_q):
        
        anti = prom_data['120.0_antisense_x_logfold'][(q*n_q):((q+1)*n_q)]
        y_center = n_q*q+n_q/2.

        plot_violin(anti, ax=ax2, bw=0.3, arange=(-10, 10, 0.05), 
                    y_offset=y_center, mult=500., color='#c3abdb')

        plt.plot(np.median(anti), y_center, markersize=7, 
            color='white', marker='D', zorder=4)
        plt.plot(np.median(anti), y_center, markersize=7, 
            color='black', fillstyle='none', marker='D', zorder=4)
        
    ax2.set_ylim(0, len(prom_data))
    ax2.set_xlim(-7, 7)
    ax2.set_xlabel('Log$_2 $ fold-change antisense transcripts')

    ax2.axvline(0, color='black', linestyle='solid', linewidth=1.5, zorder=2)
    ax2.set_yticks([])
    plt.suptitle('Antisense %s\n0-120 min' % name, fontsize=16)

    ax1.set_title("Sorted $\Delta$ %s" % name)
    ax2.set_title("Log$_2 $ fold-change Antisense transcripts")


def plot_antisense_transcripts(datastore):
    times = [0.0, 7.5, 15.0, 30.0, 60.0, 120.0]
    plt.figure(figsize=(4, 6))
    colors = plt.get_cmap('Purples')

    i = 0
    for time in times:
        cur = datastore.antisense_TPM_logfold[time].sort_values()
        plt.scatter(cur, np.arange(len(datastore.data)), color=colors(float(i)/6.+1./6), s=1, 
                    label="%s'" % str(time))
        i += 1

    plt.xlim(-10, 10)
    plt.ylim(0, len(datastore.data))
    plt.yticks([])
    plt.xlabel("log$_2$ fold-change transcripts", fontsize=14, fontdict={'weight':'ultralight'})
    plt.axvline(0, color='black')
    plt.title("Sorted change in\nantisense transcription", fontsize=16)
    plt.legend()

from src.chromatin_summary_plots import plot_distribution

def plot_antisense_vs_sense(antisense_logfold_TPM, sense_logfold_rate, time, highlight=[]):
    
    antisense_logfold_TPM = antisense_logfold_TPM.loc[sense_logfold_rate.index]

    apply_global_settings()

    sense_data = sense_logfold_rate[time]
    anti_data = antisense_logfold_TPM[time]

    ax = plot_distribution(sense_data, anti_data, 
        "log$_2$ fold-change Sense transcription rate", 
        "log$_2$ fold-change Antisense transcripts", 
        highlight=highlight,
        xlim=(-8, 8), xstep=2,
        ylim=(-8, 8), ystep=2,
        pearson=False, aux_lw=1.5,
        plot_minor=False,
        title="Sense vs antisense\ntranscription, 0-%.0f min" % time)

    for x in [-2, 2]:
        ax.axvline(x, linewidth=2, color='#505050', zorder=98)
        ax.axhline(x, linewidth=2, color='#505050', zorder=98)

        ax.axvline(x, linestyle='solid', color='#505050', linewidth=2.5, zorder=98)


def plot_bar_counts(antisense_TPM_logfold, transcript_rate_logfold,
        time=120.0):
    """
    Plot the number of genes that lie in each antisense sense bucket for the
    given time
    """

    data = antisense_TPM_logfold.join(transcript_rate_logfold, 
        lsuffix='_antisense_x_logfold',
        rsuffix='_xrate',
        how='inner')

    apply_global_settings(titlepad=20)

    time_str = str(time)

    # to calculate inclusive/exclusive values correctly
    epsilon = 1e-10

    spans = [
        (float('-inf'), -2-epsilon),
        (-2-epsilon, 2+epsilon),
        (2+epsilon, float('inf')),
    ]

    names = [
        'Decreased, <-2',
        'Unchanged, [-2, 2]',
        'Increased, >2'
    ]

    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    grays = plt.get_cmap('Greys')

    colors = [
        blues(0.35), 
        blues(0.5), 
        blues(0.65), 

        grays(0.35),
        grays(0.5),
        grays(0.65),

        reds(0.35),
        reds(0.5),
        reds(0.65)
    ]

    facecolors = [
        blues(0.35), 
        blues(0.25), 
        blues(0.65), 

        grays(0.35),
        grays(0.25),
        grays(0.65),

        reds(0.35),
        reds(0.25),
        reds(0.65)
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.tight_layout(rect=[0.1, 0.1, 0.75, 0.85])

    i = 0
    sense_i = 0
    ticks = []
    for span_sense in spans:
        anti_i = 0
        for span_antisense in spans:

            sense_k = '%s_xrate' % time_str
            anti_k = '%s_antisense_x_logfold' % time_str
            
            selected = data[(data[sense_k] >= span_sense[0]) & 
                            (data[sense_k] <  span_sense[1]) & 
                            (data[anti_k] >= span_antisense[0]) & 
                            (data[anti_k] <  span_antisense[1])]

            label = None
            if sense_i == 1: label = names[anti_i]

            x = sense_i*3 + anti_i*0.75 - 0.75
            
            y = len(selected)
            plot_y = y
            
            if plot_y > 1250:
                plot_y = 1275

            ax.text(x, plot_y+20, int(y), ha='center') 
            color = colors[i]
            ax.bar(x, plot_y, color=color, label=label, width=.5, 
                  facecolor=facecolors[i], linewidth=2,
                  edgecolor=color,
                   hatch='\\\\',
                  )
            
            i+= 1

            if anti_i == 1:
                ticks.append(x)

            anti_i += 1

        sense_i += 1

    ax.set_xticks(ticks)
    ax.set_xticklabels(names, rotation=0, ha='center')
    ax.tick_params(axis='x', length=0, pad=10)
    ax.set_title('')
    ax.set_ylim(0, 1400)
    ax.set_ylabel('# of genes', labelpad=0)
    ax.set_xlabel('Sense transcription', labelpad=10)
    yticks = np.arange(0, 1400, 200)
    ax.set_yticks(yticks)
    yticklabels = [str(y) for y in yticks]
    yticklabels = yticklabels[:-1] + [('>' + yticklabels[-1])]
    ax.set_yticklabels(yticklabels)
    ax.legend(loc=2, title='Antisense transcripts',
     bbox_to_anchor=(1.0, 1.0), frameon=False)

    for i in range(2):
        ax.axvline(i*3+1.5, color='#F0F0F0', lw=2)

    ax.set_title("Frequency of sense and\nantisense transcription, 0-120 min", fontsize=18)

    ax.plot([2.67, 3.305], [1115, 1185], lw=4, color='white')
    ax.plot([2.67, 3.305], [1100, 1170], lw=2, color=grays(0.5))
    ax.plot([2.67, 3.305], [1130, 1200], lw=2, color=grays(0.5))

def plot_antisense_dist(data):
    
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.tight_layout(rect=[0.1, 0.1, 0.85, 0.85])

    times = [0, 7.5, 15, 30, 60, 120]

    colors = plt.get_cmap('magma_r')
    i = 0
    for time in times[1:]:
        
        cur = data[time].values
        plot_density(cur, ax=ax, bw=0.15, arange=(-10, 10, 0.05), lw=2, 
                     color=colors(float(i)*0.8/5.+1./5), label="%s'" % str(time))
        i += 1
        
    ax.legend(bbox_to_anchor=(1.25, 1.), frameon=False)
    ax.set_xlim(-9, 9)
    ax.set_ylim(0, 0.85)
    ax.axvline(0, color='gray', lw=2, alpha=0.5)
    ax.set_title("Antisense transcripts\ndistribution over time", fontsize=18)
    ax.set_xlabel('Log$_2$ fold-change antisense TPM')
    ax.set_ylabel('Density')



def plot_antisense_lengths():

    antisense_boundaries = read_orfs_data('%s/antisense_boundaries_computed.csv' % rna_dir)

    from src.plot_utils import apply_global_settings
    apply_global_settings()

    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])

    antisense_lengths = (antisense_boundaries.stop - antisense_boundaries.start).dropna()

    ax.hist(antisense_lengths, 
             bins=25, linewidth=1, edgecolor='white')
    ax.set_title("Antisense transcript lengths, N=%d" % len(antisense_lengths),
                 fontsize=18)
    ax.set_xlabel("Length (bp)")
    ax.set_ylabel("# of genes")


def plot_antisense_calling(gene_name, rna_seq_pileup):

    from src.rna_seq_plotter import get_smoothing_kernel
    from src.plot_utils import apply_global_settings
    from src.utils import get_orf
    from src.transcription import filter_rna_seq
    from src.transcription import filter_rna_seq_pileup
    from src.transcript_boundaries import load_park_boundaries
    from src.plot_orf_annotations import ORFAnnotationPlotter
    from config import paper_orfs
    from src.reference_data import read_sgd_orfs, read_park_TSS_PAS
    from src.datasets import read_orfs_data

    all_orfs = read_sgd_orfs()
    all_orfs = all_orfs.join(read_park_TSS_PAS()[['TSS', 'PAS']])

    orfs_plotter = ORFAnnotationPlotter(orfs=all_orfs)
    
    antisense_boundaries = read_orfs_data('%s/antisense_boundaries_computed.csv' % rna_dir)

    park_boundaries = load_park_boundaries()
    park_boundaries = park_boundaries.join(paper_orfs[['name']])

    orf = get_orf(gene_name, park_boundaries)

    search_2 = 1000
    span = orf.transcript_start-search_2, orf.transcript_stop+search_2
    gene_pileup = filter_rna_seq_pileup(rna_seq_pileup, 
    span[0], span[1], orf.chr)

    plot_span = span
    gene = orf
    gene_rna_seq = gene_pileup

    apply_global_settings(30)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.))
    fig.tight_layout(rect=[0.1, 0, 1, 0.85])

    orfs_plotter.set_span_chrom(plot_span, gene.chr)
    orfs_plotter.plot_orf_annotations(ax1)

    sense_data = gene_rna_seq[gene_rna_seq.strand == '+']
    antisense_data = gene_rna_seq[gene_rna_seq.strand == '-']
    sense_data = np.log2(sense_data.groupby('position').sum()+1).pileup
    antisense_data = np.log2(antisense_data.groupby('position').sum()+1).pileup

    smooth_kernel = get_smoothing_kernel(100, 20)

    sense_strand = '+' if gene.strand == '+' else '-'
    antisense_strand = '+' if sense_strand == '-' else '-'

    x = sense_data.index
    sense_data = np.convolve(sense_data, smooth_kernel, mode='same')
    antisense_data = np.convolve(antisense_data, smooth_kernel, mode='same')

    ax2.plot(x, sense_data, color=plt.get_cmap('Blues')(0.5))
    ax2.plot(x, -antisense_data, color=plt.get_cmap('Reds')(0.5))
    ax2.set_xlim(*plot_span)
    ax2.set_ylim(-15, 15)
    ax2.axhline(0, color='black')

    if gene.name in antisense_boundaries.index:
        anti_gene = antisense_boundaries.loc[gene.name]
        
        y_plot = 0, 20 if gene.strand == '-' else -20, 0
        
        ax2.plot([anti_gene.start, anti_gene.start],
                [y_plot[0], y_plot[1]], color='red', linewidth=2.5, solid_capstyle='butt')
        ax2.plot([anti_gene.stop, anti_gene.stop],
                [y_plot[0], y_plot[1]], color='red', linewidth=2.5, solid_capstyle='butt')

    ax2.set_xticks(np.arange(plot_span[0], plot_span[1], 500))
    ax2.set_xticklabels([])
    _ = ax2.set_xticks(np.arange(plot_span[0], plot_span[1], 100), minor=True)

    ax2.tick_params(labelsize=14)
    ax2.set_ylabel("Sum log$_2$ (pileup+1)", fontsize=15)
    ax2.set_xlabel("Position (bp)", fontsize=15)

    ax1.set_title("Calling antisense transcripts", fontsize=26)

    ax2.axvline(383344)
    ax2.axvline(384114)
