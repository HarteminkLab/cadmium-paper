
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_orf, print_fl
import plot_utils
from src.plot_utils import register_nucleosomal_small_cmap
from src.chromatin_metrics_data import ChromatinDataStore


# limits for line plots for chromatin metrics
CHROM_YLIMS = 1, 4

class SummaryPlotter:

    def __init__(self, store, orfs, cross_correlation):

        plot_utils.apply_global_settings(titlepad=10, linewidth=2.5)

        self.orfs = orfs
        self.cross_correlation = cross_correlation
        self.times = [0.0, 7.5, 15, 30, 60, 120]
        self.cc_class = 'diff'
        self.store = store

        # for normalization values
        self.cc_bp_mean = cross_correlation.loc['diff'].mean()
        self.cc_bp_std = cross_correlation.loc['diff'].std()

        timelabels = [('%.0f\'' % time) for time in self.times]
        timelabels[1] = '7.5\''
        self.timelabels = timelabels

        plot_utils.apply_global_settings()

    def set_gene(self, gene_name):
        gene = get_orf(gene_name)
        self.gene = gene
        self.get_cc()

    def get_cc(self):
        self.orf_cc = self.cross_correlation.loc[self.cc_class].loc[self.gene.name]

        # normalize
        self.orf_cc = self.orf_cc / self.cc_bp_std

    def plot_cross_correlation_heatmap(self, show_colorbar=False, title='', 
        nucs=None, large_font=False):

        return self.plot_df_heatmap(data=self.orf_cc*-1,
            show_colorbar=show_colorbar, title=title, shaded=(-200, 500), 
            nucs=nucs, large_font=large_font)

    def plot_df_heatmap(self, data, show_colorbar=False, title='', 
        vlims=[-3, 3], cmap='Spectral_r', cbarticks=1, cbarscale=1.,
        shaded=(-200, 500), nucs=None, large_font=False):

        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

        fig.tight_layout(rect=[0.075, 0.03, 0.8, 0.945])

        fig.patch.set_alpha(0.0)
        data = data[np.arange(-600, 600)]
        positions = data.columns

        im = ax.imshow(data, aspect=60., 
            extent=[positions.min(), positions.max(),
            0, 6], cmap=cmap,
            vmin=vlims[0], vmax=vlims[1])
        
        ax.set_yticks(np.arange(0.5, 6.5))
        ax.set_yticklabels(list(reversed(self.timelabels)))

        plot_utils.plot_rect(ax, -600, 0, shaded[0]+600,
                    6, color='#A0A0A0', fill_alpha=0.5,
                zorder=10)
        plot_utils.plot_rect(ax, shaded[1], 0, 600-shaded[1],
                    6, color='#A0A0A0', fill_alpha=0.5,
                zorder=10)

        xticks = np.arange(-600, 700, 200)
        ax.set_xticks(xticks)
        xticklabels = [str(x) if x < 0 else '+' + str(x) for x in xticks]
        xticklabels[3] = 'TSS'
        ax.set_xticklabels(xticklabels)

        ax.axvline(x=0, color='black', linewidth=1)
        plot_utils.format_ticks_font(ax, fontsize=16)

        if large_font:
            ax.set_title(title, fontsize=26)
        else:
            ax.set_title(title, fontsize=20)

        ax.set_xlim(-600, 600)

        if nucs is not None:
            from src.utils import time_to_index

            times = self.times
            for i in range(len(times)):
                time = times[i]
                time_data = nucs[nucs.time == time]
                ax.scatter(time_data.mid , [5.5-i]*len(time_data), 
                    facecolor='none', edgecolor='red', marker='D', s=16)

            for link in nucs.link.unique():
                link_nucs = nucs[nucs.link == link].reset_index(drop=True)
                for j in range(1, len(link_nucs)):
                    prev = link_nucs.loc[j-1]
                    cur = link_nucs.loc[j]

                    plt.plot([prev.mid, cur.mid], [5.5 - time_to_index(prev.time),
                     5.5 - time_to_index(cur.time)], color='red')

        for k, spine in ax.spines.items():
            spine.set_zorder(100)

        if show_colorbar:

            cbax = fig.add_axes([0.85, 0.1, 0.02, 0.8]) 

            cbar = plt.colorbar(im, cax=cbax)
            ticks = np.arange(vlims[0], vlims[1]+cbarticks, cbarticks)
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels(['%.0f' % (t/cbarscale) for t in ticks])
            cbar.ax.set_ylabel('Cross correlation, per bp',
                rotation=270, va='bottom')

        return fig

    def plot_lines(self, orf_name, title='', lims=(None, None, None),
                   large_font=False):

        #fig, dis_ax = plt.subplots(1, 1, figsize=(8, 6))
        fig, dis_ax = plt.subplots(1, 1, figsize=(8, 7))
        fig.tight_layout(rect=[0.15, 0.3, 0.9, 0.9])

        times = [0, 7.5, 15, 30, 60, 120]
        time_indices = np.arange(6)
        fig.patch.set_alpha(0.0)

        disorg_color = '#138E88'
        sm_color = '#D16207'
        xrate_color = '#5D2CA3'

        x_rate_ax = dis_ax.twinx()
        sm_ax = dis_ax.twinx()
        sm_ax.yaxis.tick_left()
        sm_ax.spines["left"].set_position(("axes", -0.13))

        sm_ax.yaxis.set_label_position("left")

        dis_ax.axhline(0, color='#909090', linestyle='dashed', lw=1, zorder=0)

        sm_values = self.store.promoter_sm_occupancy_delta.loc[orf_name].values
        dis_values = self.store.gene_body_disorganization_delta.loc[orf_name].values
        xrate_values = self.store.transcript_rate_logfold.loc[orf_name].values

        dis_line = dis_ax.plot(time_indices, dis_values, marker='o', 
            markersize=11, color=disorg_color, lw=2.5, 
            label='Nucleosome disorganization', linestyle='--')

        sm_line = sm_ax.plot(time_indices,
                             sm_values, marker='P', markersize=11,
                             color=sm_color, lw=2.5, linestyle='dotted',
                             label='Small fragment occupancy')

        xrate_line = x_rate_ax.plot(time_indices,
            self.store.transcript_rate_logfold.loc[orf_name].values, 
            marker='^', markersize=12, color=xrate_color, lw=2.5,
            label='Transcription rate')

        max_chrom = np.max(list(dis_values) + list(sm_values))
        min_chrom = np.min(list(dis_values) + list(sm_values))

        max_rate = np.max(xrate_values)
        min_rate = np.min(xrate_values)

        if max_rate > 4 or max_chrom > 2:
            lim_type = '+'
        elif min_rate < -4 or min_chrom < -2:
            lim_type = '-'
        else:
            lim_type = None

        if large_font:
            sm_ax.set_title(title, fontsize=28)
        else:
            sm_ax.set_title(title, fontsize=22)

        # format x axis
        sm_ax.set_xticks(time_indices)
        sm_ax.set_xticklabels(self.timelabels)
        sm_ax.set_xlim(-0.25, 5.25)

        format_line_ax(sm_ax, sm_color, 
            '$\\Delta$ small fragment occupancy', lim_type=lim_type,
            lims=lims[0])
        format_line_ax(dis_ax, disorg_color, 
            '$\\Delta}$ nucleosome disorganization',
            lim_type=lim_type,
            lims=lims[0])
        format_xrate_ax(x_rate_ax, xrate_color, 
            'Log$_2$ fold-change transcription rate', lim_type=lim_type,
            lims=lims[1])

        sm_ax.spines['right'].set_edgecolor(xrate_color)
        sm_ax.spines['right'].set_linewidth(2)

        sm_ax.spines['left'].set_edgecolor(sm_color)
        sm_ax.spines['left'].set_linewidth(2)

        x_rate_ax.spines['left'].set_edgecolor(disorg_color)
        x_rate_ax.spines['left'].set_linewidth(2)

        x_rate_ax.spines['top'].set_linewidth(2)
        x_rate_ax.spines['bottom'].set_linewidth(2)

        lines = sm_line + dis_line + xrate_line
        labels = [l.get_label() for l in lines]
        dis_ax.legend(lines, labels, bbox_to_anchor=(-0.2, -0.15), 
            frameon=False, fontsize=16, ncol=2, loc='upper left')

        for ax in [sm_ax, x_rate_ax, dis_ax]:
            plot_utils.format_ticks_font(ax, fontname='Open Sans', fontsize=16)

        return fig

    def write_gene_plots(self, genes, cc_dir, lines_dir, show_plot=True,
        custom_lims={}, suffix='', large_font=False):

        for gene_name in genes:
            
            # create heatmaps of the cross correlation for each gene
            write_path = "%s/%s%s.pdf" % (cc_dir, gene_name, suffix)
            
            try: self.set_gene(gene_name)
            except KeyError:
                print_fl("Could not plot %s" % gene_name)
                continue
            
            fig = self.plot_cross_correlation_heatmap(show_colorbar=True,
                title='$\it{' + gene_name + '}$ cross correlation',
                large_font=large_font)
            plt.savefig(write_path, transparent=False)

            # close plots
            if not show_plot:
                plt.close(fig)
                plt.cla()
                plt.clf()

            if gene_name in custom_lims.keys():
                lims = custom_lims[gene_name]
            else: lims = (None, None, None)

            # plot lines plots of time course
            write_path = "%s/%s%s.pdf" % (lines_dir, gene_name, suffix)
            fig = self.plot_lines(self.gene.name,
                title=r'$\it{' + gene_name + '}$ time course', lims=lims, 
                large_font=large_font)
            plt.savefig(write_path, transparent=False)
            
            # close
            if not show_plot:
                plt.close(fig)
                plt.cla()
                plt.clf()

def calc_padding(neg_chrom, tot_chrom, pos_xrate):
    # calculate how much to pad the origin to match the chromatin scale
    return float(neg_chrom)/tot_chrom * pos_xrate / (1 - float(neg_chrom)/tot_chrom)

def format_xrate_ax(x_rate_ax, color, ylabel, lim_type=None, 
    lims=None):

    # calculate how much to pad the origin to match the chromatin
    # scale
    neg_chrom = CHROM_YLIMS[0]
    tot_chrom = CHROM_YLIMS[0]+CHROM_YLIMS[1]
    pos_xrate = 16
    padding = calc_padding(neg_chrom, tot_chrom, pos_xrate)

    if lims is not None:
        ylim = lims
    elif lim_type == '-':
        ylim = (-pos_xrate, padding)
    elif lim_type == '+':
        ylim = (-padding, pos_xrate)
    else:
        ylim = (-pos_xrate/2., pos_xrate/2.)

    x_rate_ax.set_yticks(np.arange(-20, 24, 4))
    x_rate_ax.set_yticks(np.arange(-20, 24, 2), minor=True)
    x_rate_ax.set_ylim(ylim[0], ylim[1])

    x_rate_ax.tick_params(axis='y', colors=color)
    x_rate_ax.set_ylabel(ylabel, rotation=270, 
            va='bottom', color=color, fontsize=16)

def format_line_ax(ax, color, ylabel, 
        scale=1., lim_type=None, lims=None):

    if lims is not None:
        ylim_scaled = lims
    elif lim_type == '-':
        ylim_scaled = (-CHROM_YLIMS[1], CHROM_YLIMS[0])
    elif lim_type == '+':
        ylim_scaled = (-CHROM_YLIMS[0], CHROM_YLIMS[1])
    else:
        tot_chrom = CHROM_YLIMS[1]+CHROM_YLIMS[0]
        ylim_scaled = (-tot_chrom/2., tot_chrom/2.)

    # scaled values for output
    yticks_scaled = np.arange(-20, ylim_scaled[1]+2, 2)
    yticks_scaled_minor = np.arange(-20, ylim_scaled[1]+2, 1)
    ylim_scaled = ylim_scaled[0], ylim_scaled[1]
    yticklabels_scaled = ['%0.0f' % y for y in yticks_scaled]

    # unscale for plotting
    ylim = ylim_scaled[0]/scale, ylim_scaled[1]/scale
    yticks = yticks_scaled/scale
    yticks_minor = yticks_scaled_minor/scale

    ax.set_yticks(yticks)
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_yticklabels(yticklabels_scaled)
    ax.set_ylim(ylim)

    ax.tick_params(axis='y', colors=color)
    ax.set_ylabel(ylabel, color=color, labelpad=0, fontsize=16)
