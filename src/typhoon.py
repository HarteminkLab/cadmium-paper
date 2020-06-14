

import sys
import math
import numpy as np
import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import collections as mc
from scipy.interpolate import InterpolatedUnivariateSpline

# from pipeline import plotRPKM
from src import math_utils
from src.datasets import read_orfs_data
from src.chromatin import filter_mnase
from src.rna_seq_plotter import RNASeqPlotter
from src.plot_orf_annotations import ORFAnnotationPlotter
from src import plot_utils
from src.utils import get_orf_name, print_fl
from src.math_utils import nearest_span
from src.reference_data import read_sgd_orf_introns


class TyphoonPlotter:

    def __init__(self, mnase_path=None, 
        rna_seq_pileup_path=None, orfs=None,
        times=[0.0, 7.5, 15, 30, 60, 120]):

        self.orfs = orfs
        
        print_fl("Loading MNase-seq...")
        self.all_mnase_data = pd.read_hdf(mnase_path, 'mnase_data')
        self.CDS_introns = read_sgd_orf_introns()

        print_fl("Loading RNA-seq pileup...")
        pileup = pd.read_hdf(rna_seq_pileup_path, 'pileup')
        self.rna_seq_plotter = RNASeqPlotter(pileup)

        self.orfs_plotter = ORFAnnotationPlotter(orfs, self.CDS_introns)

        self.times = times
        self.span = None
        self.chrom = None

        self.set_config()

    def set_config(self):

        self.figwidth = 8
        self.show_rna = True
        self.show_orfs = True
        self.title = None
        self.cur_mnase = []
        self.output = None
        self.show_saved_plot = False
        self.show_spines = True
        self.show_minor_ticks = False
        self.titles = None
        self.linkages = []
        self.plot_tick_labels = True
        self.vmax = 0.00004
        self.dpi = 200
        self.highlight_regions = []
        self.highlight_lines = []
        self.disable_mnase_seq = False
        self.relative_TSS = False

    def set_span_chrom(self, span, chrom):

        if span[0] > span[1]: 
            raise ValueError("invalid span %s" % span)

        if (span[1] - span[0]) > 10000: 
            raise ValueError("Span too large %s" % span)

        if not self.span == span or not chrom == self.chrom:
            self.chrom = chrom

            if not self.disable_mnase_seq:
                self.cur_mnase = filter_mnase(self.all_mnase_data, span[0], span[1], chrom, 
                                             length_select=(0, 250))
            else:
                self.cur_mnase = self.all_mnase_data.head(0)

            self.rna_seq_plotter.set_span_chrom(span, chrom)
            self.span = int(span[0]), int(span[1])

        self.orfs_plotter.set_span_chrom(span, chrom)

    def plot(self):
        """Plot the typhoon plot, RNA-seq pileup and annotations for the given
        span window and chromosome for each time"""

        # configuration
        times = self.times
        figwidth = self.figwidth
        show_rna = self.show_rna
        show_orfs = self.show_orfs
        title = self.title
        output = self.output
        show_saved_plot = self.show_saved_plot
        show_spines = self.show_spines
        show_minor_ticks = self.show_minor_ticks
        titles = self.titles
        plot_tick_labels = self.plot_tick_labels
        vmax = self.vmax
        dpi = self.dpi

        times = self.times
        n = len(times)

        start, end = self.span
        start, end = int(start), int(end)
        chrom = self.chrom
        mnase = self.cur_mnase

        fig, time_axes, tween_axes, orf_ax, rna_ax = self.plot_figure_setup()

        # set title
        if title is None: 
            title = "chr{} {}...{}".format(chrom, start, end)
        elif len(title) > 0:
            plt.suptitle(title.title(), fontsize=48)

        annotation_axes = []

        # plot the midpoints
        for i in range(len(times)):
            time = times[i]
            if titles is None: title = '{0:.3g}$^{{\\prime}}$'.format(time)
            else: title = titles[i]
            ax = time_axes[i]

            mnase = self.cur_mnase
            time_mnase = mnase[mnase.time == time]

            self.plot_typhoon_time(ax, time_mnase, time)
            ax.set_title(title, fontsize=24)

        if self.show_orfs:
            self.orfs_plotter.plot_orf_annotations(orf_ax)

        if self.show_rna: self.rna_seq_plotter.plot(rna_ax)

        fig.patch.set_alpha(0.0)
        for ax in time_axes + [orf_ax, rna_ax]:
            ax.patch.set_facecolor('#FCFCFC')

            # spines should be a top of every thing
            for k, spine in ax.spines.items():
                spine.set_zorder(100)

            for region in self.highlight_regions:
                plot_utils.plot_rect(ax, region[0], -250, region[1]-region[0],
                    500, color='#F0F0F0', 
                zorder=0)

            for line in self.highlight_lines:
                ax.axvline(x=line, color='#AAAAAA', linewidth=2, zorder=100)

            if len(self.linkages) > 0:
                plot_linkages(time_axes, tween_axes, self.linkages, times)

            plot_utils.format_ticks_font(ax, fontsize=20)

        if output is not None:
            # fig.patch.set_facecolor('blue')
            
            plt.savefig(output, dpi=dpi, transparent=False)
            if show_saved_plot: plt.show()
            plt.close(fig)

        return fig, time_axes, tween_axes

    def plot_figure_setup(self):
        """
        Setup figure for time series subplots with connecting plots between
        """

        plot_utils.apply_global_settings(titlepad=10, linewidth=2.5)

        # configuration
        times = self.times
        n = len(times)
        figwidth = self.figwidth
        show_rna = self.show_rna
        show_orfs = self.show_orfs
        figsize = (figwidth, None)
        plot_span = self.span

        add_rows = int(show_orfs + show_rna)
        nrows = n+add_rows
        grid_size = (nrows*3-1, 1)

        # default fig width and grid height
        if figsize is None:
            figsize = (23, grid_size[0])

        # set fig height to the grid height
        elif figsize[1] is None:
            figsize = (figsize[0], grid_size[0])

        fig = plt.figure(figsize=figsize)

        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=4, rowspan=2)
        ax0.set_xlim(*plot_span)

        time_axes = []
        tween_axes = []

        rna_ax, orf_ax = None, None

        if show_orfs: orf_ax = ax0
        else: time_axes.append(ax0)

        for i in range(0, nrows-1):
            
            y = 2+i*3

            tween_ax = plt.subplot2grid(grid_size, (y, 0), colspan=4, 
                rowspan=1, zorder=0)
            time_ax = plt.subplot2grid(grid_size, (y+1, 0), colspan=4, 
                rowspan=2, zorder=0.1)

            tween_ax.set_xlim(plot_span[0], plot_span[1])
            tween_ax.set_ylim(0, 10)
            
            time_ax.set_xlim(plot_span[0], plot_span[1])
            time_ax.set_ylim(0, 250)
            
            tween_ax.axis('off')
            tween_ax.xaxis.set_visible(False)

            if i == 0 and show_rna:
                rna_ax = time_ax
                leg_ax = tween_ax
            else:
                # between time subplots
                if i > 1 or not show_rna: tween_axes.append(tween_ax)

                # time subplot
                time_axes.append(time_ax)

        if True:
            draw_legend(leg_ax, plot_span)

        fig.tight_layout(rect=[0.075, 0.03, 0.95, 0.945])
        plt.subplots_adjust(hspace=0.0, wspace=0.5)

        time_axes[-1].set_xlabel("Position (nt)", fontsize=24)

        if len(time_axes) > 2:
            time_axes[2].set_ylabel("Fragment length (nt)", fontsize=24, labelpad=10)

        return fig, time_axes, tween_axes, orf_ax, rna_ax

    def plot_typhoon_time(self, ax, plot_data, time, scale_z=False):
        """Plot the typhoon plot for a single time"""

        # configuration
        vmax = self.vmax
        samples=-1
        xticks_interval = (500, 100)
        plot_tick_labels = True
        show_minor_ticks = self.show_minor_ticks

        query = (plot_data['time'] == time)
        time_data = plot_data[query]
        span = self.span
        plot_cutoff = 250

        # if show_minor_ticks:            
        ax.set_yticks(range(0, plot_cutoff, 50), minor=True)
        xticks = range(span[0], span[1], xticks_interval[1])
        ax.set_xticks(xticks, minor=True)

        # major xticks
        xticks_major = range(span[0], span[1]+xticks_interval[0], xticks_interval[0])
        ax.set_xticks(xticks_major, minor=False)
        # xticklabels = ['{:,}'.format(x) for x in xticks_major]
        xticklabels = ['' for x in xticks_major]
        ax.set_xticklabels(xticklabels)

        # major yticks
        yticks_major = range(0, plot_cutoff, 100)
        ax.set_yticks(yticks_major, minor=False)

        if not plot_tick_labels:
            ax.set_xticklabels(['' for _ in xticks_major])

        ax.tick_params(axis='both', labelsize=20, zorder=20)

        ax.grid(color='', linestyle='', linewidth=0, zorder=0, which='minor')
        ax.grid(color='', linestyle='', linewidth=0, zorder=0, which='major')

        if not samples is None and samples > 0:
            samples = min(samples, len(time_data))
            time_data = time_data.loc[np.random.choice(time_data.index, samples, replace=False)]

        # modify copy of data
        time_data = time_data.copy()

        ax.set_xlim(span[0], span[1])
        ax.set_ylim(0, plot_cutoff)

        # color scheme for density
        magma = cm.get_cmap('magma_r', 256)
        newcolors = np.vstack(magma(np.linspace(0.0, 1, 256)))
        cmap = ListedColormap(newcolors, name='typhoon')

        x = time_data.mid.values
        y = time_data.length.values

        # density calculation using multivariate gaussian kernel
        bw = [5, 15]
        try:
            kde = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=bw)
            z = kde.pdf([x, y])
        except ValueError:
            z = [0] * len(x)

        # attempt to normalize z so 0.95 quantile is 1.5e-5
        if scale_z:
            scale_factor = 5e-05
            scale_z = np.percentile(z, [95])[0] / scale_factor
            z = z / scale_z

        if not self.disable_mnase_seq:
            sorted_idx = np.argsort(z)
            x, y, z = x[sorted_idx], y[sorted_idx], z[sorted_idx]

            ax.scatter(x, y, c='', edgecolor='#c0c0c0', s=2, zorder=1)
            ax.scatter(x, y, c=z, edgecolor='', s=3, cmap=cmap, zorder=1,
             vmin=0, vmax=vmax)

        return ax


    def plot_gene(self, gene_name, save_dir=None, figwidth=10,
            padding=(1000, 1500), highlight=True, 
            custom_highlight_regions=None):

        orf_name = get_orf_name(gene_name)

        if orf_name not in self.orfs.index: return
        gene = self.orfs.loc[orf_name]

        # padding relative to 5', 3'
        if gene.strand == '-':
            padding = padding[1], padding[0]

        # center on TSS
        span = (gene.TSS - padding[0], gene.TSS + padding[1])

        self.set_span_chrom(span, chrom=gene.chr)
        self.title = gene_name
        
        if gene.strand == '+':
            self.highlight_regions = [(gene.TSS - 200, gene.TSS+500)]
        else:
            self.highlight_regions = [(gene.TSS - 500, gene.TSS+200)]

        if custom_highlight_regions is not None:
            self.highlight_regions = custom_highlight_regions

        if not highlight: self.highlight_regions = []

        self.highlight_lines = [gene.TSS]
        
        # write plot
        write_path = '%s/%s.png' % (save_dir, gene_name)

        if save_dir is not None:
            self.output = write_path
            print_fl("Wrote %s" % write_path)

        return self.plot()

    def plot_genes(self, genes, save_dir, 
                   figwidths={}, paddings={}):

        for gene_name in genes:

            figwidth = self.figwidth
            padding = 1000, 1500

            if gene_name in figwidths.keys(): figwidth = figwidths[gene_name]
            if gene_name in paddings.keys(): padding = paddings[gene_name]

            sys.stdout.write(gene_name + ' ')
            self.plot_gene(gene_name, save_dir,
                           figwidth=figwidth, padding=padding)


def _interp(x1, x2, y1, y2):
    """place points to create curve between"""
    y = [0, 0.1, 0.2, 5, 9.7, 9.9, 10]
    x = [x1, x1, x1, np.mean([x1, x2]), x2, x2, x2]
    yi = np.linspace(y1, y2, 101)
    ius = InterpolatedUnivariateSpline(y, x)
    xi = ius(yi)
    return xi, yi

def smoothed_linkage_x_y(xs1, xs2, y1=0, y2=10):
    # smooth splines
    xi1, yi1 = _interp(xs1, xs2, y1, y2)
    return xi1, yi1

def plot_linkages(time_axes, tween_axes, linkages, times):

    for i in range(len(times)):
        time = times[i]
        ax = time_axes[i]
        time_linkages = linkages[
            (linkages.time == time)]
        for idx, row in time_linkages.iterrows():
            ax.axvline(row.original_mid, zorder=0, alpha=0.25, color='#ff8c8c', 
                linestyle='solid', lw=3)

    for i in range(1, len(times)):

        prev_time = times[i-1]
        time = times[i]

        ax = tween_axes[i-1]

        for link in linkages.link.unique():
            cur_links = linkages[linkages.link == link]
            prev_nucs = cur_links[
                (cur_links.link == link) &
                (cur_links.time == prev_time)]
            nucs = cur_links[
                (cur_links.link == link) &
                (cur_links.time == time)]
            if len(prev_nucs) > 0 and len(nucs) > 0:
                prev_nuc = prev_nucs.reset_index().loc[0]
                nuc = nucs.reset_index().loc[0]

                xs, ys = smoothed_linkage_x_y(nuc.original_mid,
                    prev_nuc.original_mid)

                ax.plot(xs, ys,
                    color='#ff8c8c', alpha=0.25, linestyle='solid', lw=3)


def draw_legend(leg_ax, plot_span, key_length=500):

    span_width = plot_span[1] - plot_span[0]
    inset = span_width*2e-2
    y = 5

    leg_ax.set_xlim(*plot_span)

    leg_ax.set_ylim(0, 10)
    leg_ax.axis('off')
    leg_ax.xaxis.set_visible(False)

    leg_ax.plot([plot_span[0], 
                 plot_span[0]+key_length], 
        [y, y], 
        lw=24, color='#707070',
        solid_capstyle='butt')

    leg_ax.text(plot_span[0]+inset, y-0.2, 
        '%d nt' % key_length, 
        color='white', fontsize=18,
        fontdict={'fontname': 'Open Sans'},
        ha='left', va='center')


def draw_example_mnase_seq(plotter, save_dir):

    from src.chromatin import filter_mnase

    span = (124380, 125380)
    data = filter_mnase(plotter.all_mnase_data, span[0], span[1], chrom=2, time=0)

    fig, (ax, leg_ax) = plt.subplots(2, 1, figsize=(5, 4))
    fig.tight_layout(rect=[0.1, 0.1, 0.95, 0.945])
    plt.subplots_adjust(hspace=0.0, wspace=0.5)

    plotter.plot_typhoon_time(ax, data, 0, scale_z=True)
    ax.set_xlim(*span)
    ax.set_xticks(np.arange(span[0], span[1], 500))
    ax.set_xticks(np.arange(span[0], span[1], 100), minor=True)

    ax.set_xlabel("Position (nt)", fontsize=16)
    ax.set_ylabel("Fragment length (nt)", fontsize=16, labelpad=10)

    draw_legend(leg_ax, span, 500)

    write_path = '%s/%s.png' % (save_dir, 'example_mnase_seq')
    plt.savefig(write_path, dpi=200, transparent=True)


def draw_example_rna_seq(plotter, save_dir):

    from src.rna_seq_plotter import get_strand_colors

    span = 252000, 255500
    rna_plotter = plotter.rna_seq_plotter
    orf_plotter = plotter.orfs_plotter
    orfs = plotter.orfs

    rna_plotter.set_span_chrom(span, 6)
    orf_plotter.set_span_chrom(span, 6)

    fig = plt.figure(figsize=(7, 6))
    grid_size = (4,4)
    orf_ax = plt.subplot2grid(grid_size, (0, 0), colspan=4, rowspan=1)
    ax = plt.subplot2grid(grid_size, (1, 0), colspan=4, rowspan=1)
    leg_ax = plt.subplot2grid(grid_size, (2, 0), colspan=4, rowspan=2)

    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.945])
    plt.subplots_adjust(hspace=0.25, wspace=0.5)
        
    custom_orfs = orfs[orfs.name.isin(['RPN12', 'HXK1'])]
    custom_orfs = custom_orfs.reset_index(drop=True)
    custom_orfs['orf_name'] = ''

    orf_plotter.plot_orf_annotations(orf_ax, orf_classes=['Verified'], 
        custom_orfs=custom_orfs, should_auto_offset=False)
    rna_plotter.plot(ax=ax)
    orf_ax.set_ylim(-60, 60)
    ax.set_xlabel('Position (nt)', fontsize=20)

    offset = 400
    column_spacing = 750
    line_len = 400
    strand_spacing = 1800
    txt_space = 50

    color_maps = list(reversed(get_strand_colors()))
    times = rna_plotter.times

    strands = 'Watson', 'Crick'

    y_start = 2

    for strand_i in range(2):
        time_i = 0
        for column in range(2):
            for y in range(3):
                
                y_plot = y_start - y
                
                color = color_maps[strand_i][time_i]
                x_start = offset + strand_i*strand_spacing + column*column_spacing
                x_end = offset + line_len + strand_i*strand_spacing + column*column_spacing
                leg_ax.plot([x_start, x_end], 
                            [y_plot, y_plot], lw=4, color=color)
                leg_ax.text(x_start-txt_space, y_plot, "%s'" % str(times[time_i]), 
                            ha='right', va='center',
                           fontdict={'fontname': 'Open Sans', 'fontweight': 'regular'},
                           fontsize=14)
                time_i += 1
        leg_ax.text(offset + strand_i*strand_spacing + strand_spacing/4., 2.8, 
                    strands[strand_i], 
                    ha='center', va='bottom',
                   fontdict={'fontname': 'Open Sans', 'fontweight': 'regular'},
                   fontsize=16)

    span_width = span[1]-span[0]
        
    leg_ax.plot([20, 520], 
            [6, 6], 
            lw=24, color='#707070',
            solid_capstyle='butt')

    leg_ax.text(50, 6, '500 nt',
                ha='left', va='center', color='white',
               fontdict={'fontname': 'Open Sans', 'fontweight': 'regular'},
               fontsize=16)

    leg_ax.set_xlim(0, span_width)
    leg_ax.set_ylim(-3, 7)
    leg_ax.axis('off')
    plt.savefig('%s/example_rna_seq.png' % save_dir, dpi=150, transparent=True)


def plot_example_cross(plotter, save_dir):
    from src.chromatin import filter_mnase
    from src.plot_utils import apply_global_settings
    from config import cross_corr_sense_path
    from src.utils import get_orf

    orf_cc = pd.read_hdf(cross_corr_sense_path, 'cross_correlation')
    orfs = plotter.orfs

    gene = get_orf('APJ1', orfs)
    span = (gene.TSS - 500, gene.TSS + 500)
    plotter.set_span_chrom(span, gene.chr)

    cc_nuc = orf_cc.loc['nucleosomal'].loc[gene.name].loc[0.0]
    cc_small = orf_cc.loc['small'].loc[gene.name].loc[0.0]

    data = filter_mnase(plotter.all_mnase_data, span[0], span[1], chrom=gene.chr, time=0)

    fig, (ax, leg_ax) = plt.subplots(2, 1, figsize=(5, 6))
    fig.tight_layout(rect=[0.1, 0.1, 0.92, 0.945])
    plt.subplots_adjust(hspace=0.0, wspace=0.5)

    plotter.plot_typhoon_time(ax, data, 0, scale_z=True)
    ax.set_xlim(*span)
    ax.set_xticks(np.arange(span[0], span[1], 500))
    ax.set_xticks(np.arange(span[0], span[1], 100), minor=True)

    ax.set_xlabel("Position (nt)", fontsize=16)
    ax.set_ylabel("Fragment length (nt)", fontsize=16, labelpad=10)
    ax.set_ylim(-100, 250)

    draw_legend(leg_ax, span, 500)

    cc_ax = ax.twinx()
    cc_ax.set_ylabel("Cross correlation $\\times$0.1", rotation=270, fontsize=16, labelpad=10, va='bottom')

    scale_cc = 1
    y_origin = 0
    x = cc_nuc.index + gene.TSS
    y = cc_nuc.values*scale_cc + y_origin
    cc_ax.fill_between(x, y, y_origin, color='#28a098')

    y = -cc_small.values*scale_cc +y_origin
    cc_ax.fill_between(x, y_origin, y, color='#f28500')
    cc_ax.set_ylim(-0.1, 0.4)
    cc_ax.set_yticklabels(np.arange(-1, 5))

    write_path = '%s/%s.png' % (save_dir, 'example_cross_correlation')
    plt.savefig(write_path, dpi=200, transparent=True)
    print_fl("Wrote %s" % write_path)
