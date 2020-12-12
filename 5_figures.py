

from src import plot_utils
from src.utils import mkdirs_safe
import numpy as np
from config import *
import matplotlib.pyplot as plt
import pandas as pd
from src.timer import Timer
from src.typhoon import TyphoonPlotter
from src import met4
from src.typhoon import draw_example_mnase_seq
from src.typhoon import draw_example_rna_seq
from src.typhoon import plot_example_cross, get_plotter
from src.gene_list import get_gene_list, get_paper_list
from src.datasets import read_orfs_data
from src.summary_plotter import SummaryPlotter
from src.chromatin_metrics_data import ChromatinDataStore
from src.utils import get_orf
from src.reference_data import all_orfs_TSS_PAS, read_sgd_orfs
from src.transformations import difference

all_orfs = all_orfs_TSS_PAS()
timer = Timer()
datastore = ChromatinDataStore()

genes = get_gene_list() + met4.all_genes()
plotter = None
selected_genes = ['HSP26', 'RPS7A', 'CKB1']

def go_bar_plots():

    from src.go_analysis import GOChromatinAnalysis

    write_dir = "%s/go" % OUTPUT_DIR

    save_path = '%s/disorg_terms.csv' % write_dir
    go_disorg_analysis = GOChromatinAnalysis(filepath=save_path)
    go_disorg_analysis.plot_bar()
    plt.savefig('%s/bar_top300.pdf' % write_dir)

    save_path = '%s/org_terms.csv' % write_dir
    go_org_analysis = GOChromatinAnalysis(filepath=save_path)
    go_org_analysis.plot_bar(activated_genes=False)
    plt.savefig('%s/bar_bottom300.pdf' % write_dir)


def tf_plots():

    global plotter

    from src.small_peak_calling import SmallPeakCalling
    from src.utils import mkdir_safe
    from src.small_peak_calling import plot_tf_scatter, plot_tf_heatmap, \
        plot_tf_summary
    from src.small_peak_calling import plot_colorbars

    small_peaks = SmallPeakCalling()
    small_peaks.load_data()

    save_dir = '%s/tf_analysis' % OUTPUT_DIR
    mkdir_safe(save_dir)

    plot_tf_scatter(small_peaks, t1=60)
    plt.savefig('%s/small_peaks_0_60.pdf' % save_dir, transparent=True)

    plot_tf_summary(small_peaks, tail=small_peaks.view_low)
    plt.savefig('%s/tf_means_bottom.pdf' % save_dir, transparent=True)

    plot_tf_summary(small_peaks, head=small_peaks.view_high)
    plt.savefig('%s/tf_means_top.pdf' % save_dir, transparent=True)

    plot_tf_summary(small_peaks)
    plt.savefig('%s/tf_means.pdf' % save_dir, transparent=True)

    # plot Aft1/Aft2 peaks
    labeled_peaks = small_peaks.all_motifs.copy()
    labeled_peaks = labeled_peaks[labeled_peaks.tf.isin(['AFT1', 'AFT2'])][['orf', 'peak']].drop_duplicates()
    labeled_peaks = labeled_peaks.merge(paper_orfs[['name']], left_on='orf', right_on='orf_name')
    labeled_peaks = labeled_peaks.set_index('peak')
    selected_labeled_peaks = labeled_peaks[labeled_peaks['name'].isin(['LEE1', 'SER33', 'ENB1'])]

    fig, ax = plot_tf_scatter(small_peaks, tf_names=['AFT1', 'AFT2'], 
        labeled_peaks=selected_labeled_peaks, t1=60.0)
    plt.savefig("%s/aft1_aft2_scatter.pdf" % save_dir, transparent=True)

    # typhoon dir
    save_typhoon_dir = '%s/tf_analysis/typhoon/' % OUTPUT_DIR
    save_dir_all_motifs = '%s/tf_analysis/typhoon_all_motifs/' % OUTPUT_DIR
    mkdirs_safe([save_typhoon_dir, save_dir_all_motifs])

    if plotter is None:
        plotter = get_plotter()

    aft_genes = ['SER33', 'LEE1', 'ENB1']
    plotter.plot_genes(aft_genes, save_typhoon_dir, save_dir_all_motifs,
        times=[0.0, 30.0, 60.0], titlesize=34)


def typhoon_plots():

    global plotter
    plotter = get_plotter()

    save_dir = '%s/typhoon' % OUTPUT_DIR
    save_dir_all_motifs = '%s/typhoon_all_motifs' % OUTPUT_DIR

    mkdirs_safe([save_dir, save_dir_all_motifs, misc_figures_dir])

    figwidths = {'MCD4': 12, 'APJ1': 12}
    paddings = {'MCD4': (1000, 2000), 'APJ1': (1000, 2000)}

    print_fl("Plotting typhoons...", end='')
    plotter.plot_genes(genes, save_dir, save_dir_all_motifs, figwidths=figwidths, paddings=paddings)
    print_fl("Done.")
    timer.print_time()

    # example plots
    print_fl("Plotting examples...", end='')
    draw_example_mnase_seq(plotter, misc_figures_dir)
    draw_example_rna_seq(plotter, misc_figures_dir)
    plot_example_cross(plotter, misc_figures_dir)
    print_fl("Done.")
    timer.print_time()


def summary_plots():

    orfs = paper_orfs
    orf_cc = pd.read_hdf(cross_corr_sense_path,
                         'cross_correlation')

    sum_plotter = SummaryPlotter(datastore, orfs, orf_cc)
    show_saved_plot = False

    custom_lims = {
        'TAD2': [(-3, 3), (-3, 3)],
        'MET31': [(-2.5, 2.5), (-8, 8)]
    }

    cc_dir = '%s/cc' % OUTPUT_DIR
    lines_dir = '%s/lines' % OUTPUT_DIR

    mkdirs_safe([cc_dir, lines_dir])

    sum_plotter.write_gene_plots(genes, cc_dir=cc_dir, 
                             lines_dir=lines_dir, show_plot=show_saved_plot, custom_lims=custom_lims)

    sum_plotter.write_gene_plots(['HSP26'], cc_dir=cc_dir, 
                             lines_dir=lines_dir, show_plot=show_saved_plot, 
                             custom_lims=custom_lims, suffix='_figure', 
                             large_font=True)

def regression_plots():

    from src.regression_compare import plot_compare_r2, load_results
    from src.gp import plot_res_distribution, plot_res_distribution_time
    from src.gp import GP

    gp_dir = "%s/gp" % OUTPUT_DIR

    mkdirs_safe([gp_dir])

    # plot comparison
    plot_compare_r2(gp_dir)
    plt.savefig('%s/compare_gp_r2.pdf' % gp_dir, transparent=True)

    plot_compare_r2(gp_dir, show_legend=True)
    plt.savefig('%s/compare_gp_r2_legend.pdf' % gp_dir, transparent=True)

    from src.gp import plot_res_distribution_time, plot_res_distribution, GP

    results = load_results(gp_dir)
    for name in ['Full']:
        cur = GP(name, results_path='%s/%s_results.csv' % (gp_dir, name))
        plot_res_distribution(cur, selected_genes=selected_genes)
        plt.savefig('%s/%s_predictions.pdf' % (gp_dir, name), 
            transparent=True)

        for time in [7.5, 30, 120]:
            plot_res_distribution_time(cur, time, selected_genes=selected_genes)
            plt.savefig('%s/%s_%s.pdf' % (gp_dir, name, time), 
                transparent=True, dpi=100)


def plot_ORFs_len(misc_figures_dir):

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.tight_layout(rect=[0.1, 0.15, 0.95, 0.8])

    orfs = read_sgd_orfs()
    orfs = orfs[(orfs['orf_class'] == 'Uncharacterized') | (orfs['orf_class'] == 'Verified')]

    _ = ax.hist(orfs['length'], bins=100, linewidth=1, edgecolor='white')
    ax.set_xlim(0, 6000)
    ax.axvline(x=500, color='red')
    ax.set_title('ORF length distribution\n', fontsize=18)
    ax.set_xlabel("ORF length", fontsize=15)
    ax.set_ylabel("# of genes", fontsize=15)
    plt.savefig("%s/length_dist.pdf" % misc_figures_dir, transparent=True)


def plot_coverage(misc_figures_dir):

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.tight_layout(rect=[0.1, 0.15, 0.95, 0.8])

    mnase_coverage = read_orfs_data('%s/coverage_2000.csv' % mnase_dir)
    mnase_coverage = mnase_coverage[mnase_coverage.coverage > 0.8]
    ax.hist(mnase_coverage['coverage'], edgecolor='white', bins=25)
    ax.set_title('MNase-seq coverage\n[-1000, 1000] around TSS', fontsize=18)
    ax.axvline(x=0.85, color='red')
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("Coverage", fontsize=15)
    ax.set_ylabel("# of genes", fontsize=15)
    plt.savefig("%s/coverage.pdf" % misc_figures_dir, transparent=True)

def misc_plots():

    scatter_dpi = 200

    from src.met4 import plot_timecourse
    from src.chromatin_summary_plots import (plot_combined_vs_xrate,
                                             plot_sul_prom_disorg,
                                             plot_occ_vs_xrate,
                                             plot_disorg_vs_xrate,
                                             plot_diosorg_vs_occ, 
                                             plot_frag_len_dist)
    from src.cross_correlation_kernel import MNaseSeqDensityKernel

    met4_dir = "%s/met4" % OUTPUT_DIR
    scatters_dir = "%s/scatters" % OUTPUT_DIR
    kernels_dir = "%s/kernels" % OUTPUT_DIR
    mkdirs_safe([met4_dir, scatters_dir, kernels_dir])

    nuc_kernel = MNaseSeqDensityKernel(filepath=nuc_kernel_path)
    nuc_kernel.plot_kernel(kernel_type='nucleosome')
    plt.savefig('%s/nuc_kernel.pdf' % (kernels_dir), transparent=True)

    sm_kernel = MNaseSeqDensityKernel(filepath=sm_kernel_path)
    sm_kernel.plot_kernel(kernel_type='small')
    plt.savefig('%s/sm_kernel.pdf' % (kernels_dir), transparent=True)

    from src.kernel_fitter import compute_triple_kernel
    triple_kernel = compute_triple_kernel(nuc_kernel)
    triple_kernel.plot_kernel(kernel_type='triple')
    plt.savefig('%s/triple_kernel.pdf' % (kernels_dir), transparent=True)

    from src.nucleosome_calling import plot_nuc_calls_cc
    plot_nuc_calls_cc()
    plt.savefig('%s/nuc_cross_cor_0_min.pdf' % (misc_figures_dir), transparent=True)

    # met4 plots
    plot_timecourse(datastore)
    plt.savefig('%s/met4_timecourse.pdf' % (met4_dir), transparent=True)

    plot_sul_prom_disorg(datastore)
    plt.savefig('%s/met4_scatter.pdf' % (met4_dir), transparent=True, dpi=scatter_dpi)

    # scatter plots
    plot_combined_vs_xrate(datastore, selected_genes)
    plt.savefig('%s/combined_vs_xrate.pdf' % (scatters_dir), 
        transparent=True, dpi=scatter_dpi)
    
    plot_occ_vs_xrate(datastore, selected_genes)
    plt.savefig('%s/small_vs_xrate.pdf' % (scatters_dir), transparent=True, dpi=scatter_dpi)

    plot_disorg_vs_xrate(datastore, selected_genes)
    plt.savefig('%s/disorg_vs_xrate.pdf' % (scatters_dir), transparent=True, dpi=scatter_dpi)

    plot_diosorg_vs_occ(datastore, selected_genes)
    plt.savefig('%s/disorg_vs_small.pdf' % (scatters_dir), transparent=True, dpi=scatter_dpi)

    plot_ORFs_len(misc_figures_dir)
    
    plot_coverage(misc_figures_dir)

    global plotter

    if plotter is None:
        plotter = get_plotter()

    # plot sampled mnase data
    plot_frag_len_dist(plotter.all_mnase_data)
    plt.savefig("%s/frag_length_distribution.pdf" % misc_figures_dir, transparent=True)

    print_fl("Load allMNase-seq data for fragment length distributions")
    all_mnase_data = pd.read_hdf('%/mnase_seq_merged_all.h5.z' % mnase_dir, 
                             'mnase_data')
    repl1_mnase = all_mnase_data[all_mnase_data['source'] == 'dm498_503']
    repl2_mnase = all_mnase_data[all_mnase_data['source'] == 'dm504_509']
    print_fl("Done.")

    plot_frag_len_dist(repl1_mnase, "Replicate 1", normalize=True)
    plt.savefig('%s/frag_length_distribution_repl1.pdf' % misc_figures_dir, transparent=True)

    plot_frag_len_dist(repl2_mnase, "Replicate 2", normalize=True)
    plt.savefig('%s/frag_length_distribution_repl2.pdf' % misc_figures_dir, transparent=True)


def entropy_examples():

    all_orfs = all_orfs_TSS_PAS()

    global plotter
    if plotter is None:
        plotter = get_plotter()

    from src.entropy import plot_entropy_example

    orf = get_orf('CLF1', all_orfs)
    plot_entropy_example(plotter, orf, (-460, 40), "Low entropy")
    plt.savefig('%s/low_entropy.pdf' % (misc_figures_dir), dpi=100)

    from src.entropy import plot_entropy_example

    orf = get_orf('HSP26', all_orfs)
    plot_entropy_example(plotter, orf, (200, 700), "High entropy")
    plt.savefig('%s/high_entropy.pdf' % (misc_figures_dir), dpi=100)


def antisense_plots():

    from src.antisense_analysis import plot_antisense_vs_sense
    from src.antisense_analysis import plot_bar_counts, plot_antisense_dist

    save_dir = '%s/antisense' % OUTPUT_DIR
    mkdirs_safe([save_dir])

    antisense_TPM = read_orfs_data('%s/antisense_TPM.csv' % rna_dir)
    antisense_TPM_logfold = read_orfs_data('%s/antisense_TPM_log2fold.csv' % rna_dir)

    plot_antisense_vs_sense(antisense_TPM_logfold, datastore.transcript_rate_logfold,
        120.0, highlight=['MET31', 'CKB1', 'RPS7A',
                          'YBR241C', 'UTR2'
        ])
    plt.savefig('%s/sense_antisense_distr.pdf' % save_dir, transparent=True, dpi=100)
    
    plot_bar_counts(antisense_TPM_logfold, datastore.transcript_rate_logfold)
    plt.savefig('%s/sense_antisense_counts.pdf' % save_dir)

    plot_antisense_dist(antisense_TPM_logfold)
    plt.savefig('%s/antisense_logfc_dist.pdf' % save_dir)

    from src.antisense_analysis import plot_antisense_lengths, plot_antisense_calling

    rna_seq_pileup = pd.read_hdf('%s/rna_seq_pileup.h5.z' % rna_dir, 
        'pileup')
    antisense_boundaries = read_orfs_data('%s/antisense_boundaries_computed.csv' % rna_dir)

    plot_antisense_lengths()
    plt.savefig('%s/antisense_lengths_dist.pdf' % save_dir)

    plot_antisense_calling('MET31', rna_seq_pileup)
    plt.savefig('%s/antisense_met31_calling.pdf' % save_dir)

    from src.chromatin_summary_plots import plot_distribution

    anti_datastore = ChromatinDataStore(is_antisense=True)
    x = anti_datastore.promoter_sm_occupancy_delta.mean(axis=1)
    y = anti_datastore.antisense_TPM_logfold.mean(axis=1).loc[x.index]
    model = plot_distribution(x, y, '$\\Delta$ Antisense promoter occupancy', 
                              'Log$_2$ fold-change antisense transcript', 
                              highlight=[],
                             title='Promoter occupancy vs transcription (Antisense)',
                              xlim=(-1.5, 1.5), ylim=(-4, 4), xstep=0.5, ystep=1)
    plt.savefig('%s/antisense_chrom_dist_prom_vs_xrate.pdf' % save_dir)

    x = anti_datastore.gene_body_disorganization_delta.mean(axis=1).dropna()
    y = anti_datastore.antisense_TPM_logfold.loc[x.index].mean(axis=1).loc[x.index]

    model = plot_distribution(x, y, '$\\Delta$ antisense nucleosome disorganization', 
                              'Log$_2$ fold-change antisense transcripts', 
                              highlight=[],
                             title='Nuc. disorganization vs transcription (Antisense)', 
                             xlim=(-1.5, 1.5), ylim=(-4, 4), xstep=0.5, ystep=1)
    plt.savefig('%s/antisense_chrom_dist_disorg_vs_xrate.pdf' % save_dir)


def plot_heatmaps():

    from config import OUTPUT_DIR
    from src.chromatin_metrics_data import pivot_metric
    from src.chromatin_heatmaps import ChromatinHeatmaps

    heatmaps = ChromatinHeatmaps(datastore)

    write_dir = '%s/heatmaps' % OUTPUT_DIR
    mkdirs_safe([write_dir])

    heatmaps.show_xlabels = True
    heatmaps.show_saved_plot = False
    heatmaps.plot_heatmap(write_path=("%s/all.png" % write_dir), aspect_scale=25., fig_height=20., 
                          lines=[200, -200])

    heatmaps.show_xlabels = False
    small_heatmap_scale = 15.
    heatmaps.plot_heatmap(head=200, write_path=("%s/t100.png" % write_dir), aspect_scale=small_heatmap_scale,
                          lines=[-20])

    heatmaps.show_xlabels = True
    heatmaps.plot_heatmap(tail=200, write_path=("%s/b100.png" % write_dir), aspect_scale=small_heatmap_scale,
                          lines=[20])
    heatmaps.show_xlabels = False

    heatmaps.plot_gene_names = True
    heatmaps.plot_heatmap(head=20, write_path=("%s/t20.png" % write_dir), aspect_scale=small_heatmap_scale)

    heatmaps.show_xlabels = True
    heatmaps.plot_heatmap(tail=20, write_path=("%s/b20.png" % write_dir), aspect_scale=small_heatmap_scale)
    heatmaps.plot_gene_names = False

    from src import plot_utils

    heatmaps.plot_colorbars(write_path='%s/cbar.png' % write_dir)

    from src import met4

    heatmaps.show_saved_plot = True
    heatmaps.plot_gene_names = True
    heatmaps.plot_heatmap(orf_groups=met4.orf_groups(), 
                          group_names=met4.groups(), 
                          group_colors=met4.group_colors(),
                          write_path=("%s/sulfur.pdf" % write_dir),
                          fig_height=10,
                          aspect_scale=5000.,
                          highlight_max=[],
                          y_padding=1)
    heatmaps.plot_gene_names = False


def shift_plots():

    from src.nucleosome_calling import plot_p123
    from src.reference_data import read_sgd_orf_introns, read_sgd_orfs
    from src.reference_data import read_park_TSS_PAS
    from src.summary_plotter import SummaryPlotter

    global plotter

    orf_cc = pd.read_hdf(cross_corr_sense_path,
                         'cross_correlation')

    all_orfs = all_orfs_TSS_PAS()

    sum_plotter = SummaryPlotter(datastore, all_orfs, orf_cc)

    if plotter is None:
        plotter = get_plotter()

    save_dir = '%s/shift' % OUTPUT_DIR
    mkdirs_safe([save_dir])

    for gene_name in genes:
        fig = plot_p123(gene_name, orf_cc, plotter, sum_plotter, save_dir)

    p1 = datastore.p1_shift[[120.0]]
    p2 = datastore.p2_shift[[120.0]]
    p3 = datastore.p3_shift[[120.0]]

    p12 = p1.join(p2, lsuffix='_+1', rsuffix='_+2')
    p23 = p2.join(p3, lsuffix='_+2', rsuffix='_+3')

    from src.chromatin_summary_plots import plot_distribution

    x = datastore.p1_shift[120]
    y = datastore.transcript_rate_logfold.loc[x.index][120.0]

    model = plot_distribution(x, y, '$\\Delta$ +1 nucleosome shift', 
                          '$\log_2$ fold-change transcription rate', 
                          title='+1 shift vs transcription, 0-120 min',
                          xlim=(-40, 40), ylim=(-8, 8), xstep=10, ystep=2, 
                          pearson=True, s=10)
    plt.savefig('%s/shift_+1_xrate.pdf' % save_dir, transparent=True)

    x = datastore.p1_shift[120]
    y = datastore.p2_shift[120]
    model = plot_distribution(x, y, '$\\Delta$ +1 nucleosome shift', 
                              '$\\Delta$ +2 nucleosome shift', 
                              title='+1, +2 nucleosome shift\n0-120 min',
                              xlim=(-40, 40), ylim=(-40, 40), xstep=10, ystep=10, 
                              pearson=False, s=10)

    plt.savefig('%s/shift_p12.pdf' % save_dir, transparent=True)

def locus_plots():
    """Merge typhoon, cc, and line plots into a single pdf"""

    from src.pdf_utils import merge_locus_pdf

    save_dir = '%s/locus_plots' % OUTPUT_DIR
    mkdirs_safe([save_dir])

    for gene_name in genes:
        write_path = '%s/locus_%s.pdf' % (save_dir, gene_name)
        merge_locus_pdf(OUTPUT_DIR, gene_name, write_path)


def main():

    print_fl("*********************")
    print_fl("* 5    Figures     *")
    print_fl("*********************")

    print_preamble()

    plot_utils.apply_global_settings()

    print_fl("\n------- Typhoon ----------\n")
    typhoon_plots()

    print_fl("\n------- Line/Cross Plots ----------\n")
    summary_plots()

    print_fl("\n------- Locus plots ----------\n")
    locus_plots()

    print_fl("\n------- GO Plots ----------\n")
    go_bar_plots()

    print_fl("\n------- Heatmap Plots ----------\n")
    plot_heatmaps()

    print_fl("\n------- Regression Plots ----------\n")
    regression_plots()

    print_fl("\n------- Antisense Plots ----------\n")
    antisense_plots()

    print_fl("\n------- TF Plots ----------\n")
    tf_plots()

    print_fl("\n------- Other ----------\n")
    misc_plots()

    print_fl("\n------- Entropy ----------\n")
    entropy_examples()

    print_fl("\n--------- Shift -----------\n")
    shift_plots()


if __name__ == '__main__':
    main()
