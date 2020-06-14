

import numpy as np
import matplotlib.pyplot as plt
from src.plot_utils import apply_global_settings, hide_spines, plot_density
from src.plot_utils import plot_violin, plot_density_scatter, plot_rect
from src.colors import parula
from src.datasets import read_orfs_data
import matplotlib.patheffects as path_effects


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
    ax2.set_xlabel('$\\log_2 $ fc Antisense transcripts')

    ax2.axvline(0, color='black', linestyle='solid', linewidth=1.5, zorder=2)
    ax2.set_yticks([])
    plt.suptitle('Antisense %s\n0-120 min' % name, fontsize=16)

    ax1.set_title("Sorted $\Delta$ %s" % name)
    ax2.set_title("$\\log_2$ fc Antisense transcripts")


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

    apply_global_settings(10)

    fig = plt.figure(figsize=(6.5, 6.5))
    fig.tight_layout(rect=[0.0, 0.0, 1, 1])
    plt.subplots_adjust(hspace=0.05, wspace=0.04)
    fig.patch.set_alpha(0.0)

    grid_len = 9
    grid_size = (grid_len, grid_len)
    ax = plt.subplot2grid(grid_size, (1, 0), colspan=grid_len-1, rowspan=grid_len-1)
    tax = plt.subplot2grid(grid_size, (0, 0), colspan=grid_len-1, rowspan=1)
    rax = plt.subplot2grid(grid_size, (1, grid_len-1), colspan=1, rowspan=grid_len-1)

    sense_data = sense_logfold_rate[time]
    anti_data = antisense_logfold_TPM[time]

    ax = plot_distribution(sense_data, anti_data, 
        "log$_2$ fold-change Sense transcription rate", 
        "log$_2$ fold-change Antisense transcripts", 
        highlight=highlight,
        xlim=(-9, 9), xstep=2,
        ylim=(-9, 9), ystep=2,
        pearson=False,
        title="Sense vs antisense\ntranscription, 0-%.0f min" % time)

    for x in [-2, 2]:
        ax.axvline(x, linewidth=1.5, color='#a0a0a0')
        ax.axhline(x, linewidth=1.5, color='#a0a0a0')



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
    ax.set_ylim(0, 0.75)
    ax.axvline(0, color='gray', lw=2, alpha=0.5)
    ax.set_title("Antisense transcripts\ndistribution over time", fontsize=18)
    ax.set_xlabel('log$_2$ fold-change antisense TPM')
    ax.set_ylabel('Density')

