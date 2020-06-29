
import numpy as np
import matplotlib.pyplot as plt
from src.plot_utils import apply_global_settings, hide_spines, plot_density
from src.plot_utils import plot_violin, plot_density_scatter, plot_rect
from src.colors import parula
from src.datasets import read_orfs_data
import matplotlib.patheffects as path_effects
import statsmodels.api as sm
from scipy.stats import pearsonr 
from src.utils import get_orf_names, get_orf_name


def plot_disorg_vs_xrate(datastore, selected_genes):
    
    x = datastore.gene_body_disorganization_delta.mean(axis=1)
    y = datastore.transcript_rate_logfold.mean(axis=1).loc[x.index]

    model = plot_distribution(x, y, '$\\Delta$ Nucleosome disorganization', 
                              'Log$_2$ fold-change transcription rate', 
                              highlight=selected_genes,
                             title='Nucleosome disorganization vs transcription')

def plot_occ_vs_xrate(datastore, selected_genes):
    x = datastore.promoter_sm_occupancy_delta.mean(axis=1)
    y = datastore.transcript_rate_logfold.mean(axis=1).loc[x.index]

    model = plot_distribution(x, y, '$\\Delta$ Promoter occupancy', 
                              'Log$_2$ fold-change transcription rate', 
                              highlight=selected_genes,
                              xlim=(-2, 2),
                             title='Promoter occupancy vs transcription')

def plot_diosorg_vs_occ(datastore, selected_genes):
    x = datastore.promoter_sm_occupancy_delta.mean(axis=1)
    y = datastore.gene_body_disorganization_delta.mean(axis=1).loc[x.index]

    model = plot_distribution(x, y, '$\\Delta$ Promoter occupancy', 
                              '$\\Delta$ Nucleosome disorganization', 
                              highlight=selected_genes,
                             title='Promoter occupancy vs nucleosome disorganization',
                             xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))

def plot_combined_vs_xrate(datastore, selected_genes):

    x = datastore.combined_chromatin_score
    y = datastore.transcript_rate_logfold.loc[x.index].mean(axis=1).loc[x.index]

    model = plot_distribution(x, y, '$\\Delta$ Combined chromatin dynamics score', 
                              'Log$_2$ fold-change transcription rate', 
                              highlight=selected_genes,
                             title='Combined chromatin vs transcription')

def plot_sul_prom_disorg(datastore):

    x = datastore.promoter_sm_occupancy_delta.mean(axis=1)
    y = datastore.gene_body_disorganization_delta.mean(axis=1).loc[x.index]

    model = plot_distribution(x, y, '$\\Delta$ Promoter occupancy', 
                              '$\\Delta$ Nucleosome disorganization', 
                              highlight=['PDC1', 'PDC6', 'MET32', 'MET30'],
                              highlight_format={
                                'PDC1':{
                                    'color': '#B868A9',
                                    'marker': 'P',
                                    'filled': True,
                                },
                                'PDC6':{
                                    'color': '#B868A9',
                                    'marker': 'P',
                                    'filled': True,
                                },
                                'MET32':{
                                    'color': '#3C6096',
                                    'marker': 'v',
                                },
                                'MET30':{
                                    'color': '#7A7A7A',
                                    'va':'top',
                                    'ha':'left',
                                    'marker': 'o',
                                }
                              },
                              groups={
                                      'Sulfur assimilation': {
                                          'color': '#E27742',
                                          'orfs': ['YBR294W', 'YLR092W', 'YJR010W', 
                                          'YKL001C', 'YPR167C', 'YJR137C', 'YFR030W']
                                      }
                                    },
                              markersize=60,
                             title='Sulfur pathway genes',
                             xstep=1, ystep=1,
                             xlim=(-1.5, 2.5), ylim=(-1.5, 2.5), ha='right', pearson=False)

def plot_distribution(x_data, y_data, xlabel, ylabel, highlight=[], 
    title=None, xlim=(-2.5, 2.5), ylim=(-6, 10), xstep=2, ystep=2, pearson=True,
    ha='right', va='bottom', plot_aux='cross', groups={}, highlight_format={},
    aux_lw=1.5, s=5, markersize=53, ax=None, text_offset=None):
    
    apply_global_settings(10)

    plot_default_ax = ax is None
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6.5))

        fig.patch.set_alpha(0.0)

        grid_len = 9
        grid_size = (grid_len, grid_len)

        ax = plt.subplot2grid(grid_size, (1, 0), colspan=grid_len-1, rowspan=grid_len-1)
        tax = plt.subplot2grid(grid_size, (0, 0), colspan=grid_len-1, rowspan=1)
        rax = plt.subplot2grid(grid_size, (1, grid_len-1), colspan=1, rowspan=grid_len-1)
    else:
        tax = None
        rax = None

    if len(groups) > 0 and plot_default_ax:
        fig.tight_layout(rect=[0.14, 0.15, 0.9, 0.9])

    if plot_default_ax:
        plt.subplots_adjust(hspace=0.05, wspace=0.04)


    if plot_default_ax:
        xspan_diff = xlim[1]-xlim[0]
        yspan_diff = xlim[1]-xlim[0]
        y = plot_density(x_data, ax=tax, arange=(xlim[0], xlim[1], xspan_diff*1e-3),
                    bw=xspan_diff*1e-2, fill=True, color='#a0a0a0')
        y_max = np.max(y)
        tax.set_xlim(*xlim)
        tax.set_ylim(y_max*-1e-1, y_max*1.5)

        x = plot_density(y_data, ax=rax, arange=(ylim[0], ylim[1], yspan_diff*1e-3),
                    bw=yspan_diff*1e-2, flip=True, fill=True, color='#a0a0a0')
        x_max = np.max(x)
        rax.set_ylim(*ylim)
        rax.set_xlim(x_max*-1e-1, x_max*1.5)

        hide_spines(rax)
        hide_spines(tax)

    plot_density_scatter(x_data, y_data,
        s=s, bw=[0.15, 0.15], 
        ax=ax, cmap=parula(), alpha=1., zorder=20)

    plot_rect(ax, xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0],
     'white', fill_alpha=0.5, zorder=90)

    for group_name, group in groups.items():
        group_orfs = group['orfs']
        group_x = x_data[x_data.index.isin(group_orfs)]
        group_y = y_data[y_data.index.isin(group_orfs)]
        ax.scatter(group_x, 
                   group_y,
                   s=53, facecolor='none', color=group['color'],
                   zorder=98, marker='D', linewidth=1.5, label=group_name,
                   rasterized=True)

    for gene_name in highlight:
        orf_name = get_orf_name(gene_name)
        if orf_name not in x_data.index: continue

        selected_x = x_data.loc[orf_name]
        selected_y = y_data.loc[orf_name]
        marker = 'D'
        color = '#c43323'
        facecolor = 'none'

        if gene_name in highlight_format.keys():

            gene_fmt = highlight_format[gene_name]

            if 'marker' in gene_fmt.keys():
                marker = gene_fmt['marker']

            if 'color' in gene_fmt.keys():
                color =  gene_fmt['color']

            if 'filled' in gene_fmt.keys():
                facecolor = color

        ax.scatter(selected_x, 
                   selected_y,
                   s=markersize, facecolor=facecolor, color=color,
                   zorder=100, marker=marker, linewidth=1.5)

        if text_offset is None:
            text_offset = (xlim[1]-xlim[0]) * 5e-3

        offsets = text_offset, text_offset

        cur_ha = ha
        cur_va = va

        if gene_name in highlight_format.keys():
            cur_hl_fmt = highlight_format[gene_name]
            cur_ha = cur_hl_fmt['ha'] if 'ha' in cur_hl_fmt.keys() else ha
            cur_va = cur_hl_fmt['va'] if 'va' in cur_hl_fmt.keys() else va

        if cur_ha == 'right':
            offsets = -text_offset, offsets[1]
        elif cur_ha == 'left':
            offsets = text_offset, offsets[1]
        elif cur_ha == 'center':
            offsets = 0, offsets[1]

        if cur_va == 'top':
            offsets = offsets[0], -text_offset
        elif cur_va == 'bottom':
            offsets = offsets[0], text_offset

        text = ax.text(selected_x + offsets[0], 
                selected_y + offsets[1],
                gene_name,

            fontdict={'fontname': 'Open Sans', 
            'fontweight': 'regular'},
            fontsize=12,
            ha=cur_ha,
            va=cur_va,
            zorder=99
            )
        text.set_path_effects([path_effects.Stroke(linewidth=3, 
            foreground='white'),
                           path_effects.Normal()])

    ax.set_xticks(np.arange(-100, 100, xstep))
    ax.set_yticks(np.arange(-100, 100, ystep))

    if xstep < 5:
        ax.set_xticks(np.arange(-100, 100, 1), minor=True)

    if ystep < 5:
        ax.set_yticks(np.arange(-100, 100, 1), minor=True)

    ax.tick_params(axis='x', pad=5, labelsize=15)
    ax.tick_params(axis='y', pad=5, labelsize=15)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(groups) > 0:
        ax.legend(loc=1, bbox_to_anchor=(0.475, -0.2), frameon=False,
            fontsize=14)

    if plot_aux == 'cross' or plot_aux == 'both':
        ax.axvline(0, linestyle='solid', color='#505050', linewidth=aux_lw, zorder=98)
        ax.axhline(0, linestyle='solid', color='#505050', linewidth=aux_lw, zorder=98)

    if plot_aux == 'diag' or plot_aux == 'both':
        ax.plot([-100, 100], 
            [-100, 100],
            linestyle='solid', color='#505050', linewidth=aux_lw, zorder=98)

    if pearson:
        from src.math_utils import convert_to_latex_sci_not

        cor, pval = pearsonr(x_data, y_data)
        pval = convert_to_latex_sci_not(pval)

        title = ("%s\nPearson's r=%.2f, p=%s" % 
               (title, cor, pval))

    if plot_default_ax:
        tax.set_title(title, fontsize=18)
    else:
        ax.set_title(title, fontsize=18)

    return ax
