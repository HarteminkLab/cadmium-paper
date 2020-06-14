

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
from src.typhoon import plot_example_cross
from src.gene_list import get_gene_list
from src.datasets import read_orfs_data
from src.summary_plotter import SummaryPlotter
from src.chromatin_metrics_data import ChromatinDataStore
from src.utils import get_orf
from src.reference_data import read_park_TSS_PAS, read_sgd_orfs

timer = Timer()
datastore = ChromatinDataStore()

def go_bar_plots():

    from src.go_analysis import GOChromatinAnalysis

    write_dir = "%s/go" % OUTPUT_DIR

    save_path = '%s/disorg_terms.csv' % write_dir
    go_disorg_analysis = GOChromatinAnalysis(filepath=save_path)
    go_disorg_analysis.plot_bar()
    plt.savefig('%s/bar_top300.png' % write_dir, dpi=150)

    save_path = '%s/org_terms.csv' % write_dir
    go_org_analysis = GOChromatinAnalysis(filepath=save_path)
    go_org_analysis.plot_bar(activated_genes=False)
    plt.savefig('%s/bar_bottom300.png' % write_dir, dpi=150)


def typhoon_plots():

    orfs = paper_orfs

    plotter = TyphoonPlotter(mnase_path=mnase_seq_path,
                             rna_seq_pileup_path=pileup_path,
                             orfs=orfs)

    save_dir = '%s/typhoon' % OUTPUT_DIR

    mkdirs_safe([save_dir, misc_figures_dir])

    genes = get_gene_list() + met4.all_genes()

    figwidths = {'MCD4': 12, 'APJ1': 12}
    paddings = {'MCD4': (1000, 2000), 'APJ1': (1000, 2000)}

    print_fl("Plotting typhoons...", end='')
    plotter.plot_genes(genes, save_dir, figwidths=figwidths, paddings=paddings)
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
                         'cross_correlation'    )

    plotter = SummaryPlotter(datastore, orfs, orf_cc)
    show_saved_plot = False

    genes = met4.all_genes() + get_gene_list()

    custom_lims = {'TAD2': [(-4, 15), (-4.267, 16)],
                   'APJ1': [(-2, 7), (-1.78, 8)]}

    cc_dir = '%s/cc' % OUTPUT_DIR
    lines_dir = '%s/lines' % OUTPUT_DIR

    mkdirs_safe([cc_dir, lines_dir])

    plotter.write_gene_plots(genes, cc_dir=cc_dir, 
                             lines_dir=lines_dir, show_plot=show_saved_plot, custom_lims=custom_lims)

def regression_plots():

    from src.regression_compare import plot_compare_r2, load_results
    from src.gp import plot_res_distribution, plot_res_distribution_time
    from src.gp import GP

    gp_dir = "%s/gp" % OUTPUT_DIR

    mkdirs_safe([gp_dir])

    # plot comparison
    plot_compare_r2(gp_dir)
    plt.savefig('%s/compare_gp_r2.png' % gp_dir, dpi=150, transparent=True)

    from src.gp import plot_res_distribution_time, plot_res_distribution, GP

    selected_genes = ['HSP26', 'MET32', 'MET31', 'RPS7A', 'CKB1']

    results = load_results(gp_dir)
    for name in results.columns:
        cur = GP(name, results_path='%s/%s_results.csv' % (gp_dir, name))
        plot_res_distribution(cur, selected_genes=selected_genes)
        plt.savefig('%s/%s_predictions.png' % (gp_dir, name), dpi=150, 
            transparent=True)

        plot_res_distribution_time(cur, 120, selected_genes=selected_genes)
        plt.savefig('%s/%s_120.png' % (gp_dir, name), dpi=150, 
            transparent=True)


def plot_ORFs_len(misc_figures_dir):

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.tight_layout(rect=[0.1, 0.15, 0.95, 0.8])

    orfs = read_sgd_orfs()
    orfs = orfs[(orfs['orf_class'] == 'Uncharacterized') | (orfs['orf_class'] == 'Verified')]

    _ = ax.hist(orfs['length'], bins=100, linewidth=1, edgecolor='white')
    ax.set_xlim(0, 6000)
    ax.axvline(x=500, color='red')
    ax.set_title('ORF length distribution', fontsize=18)
    ax.set_xlabel("ORF length", fontsize=15)
    ax.set_ylabel("# of genes", fontsize=15)
    plt.savefig("%s/length_dist.png" % misc_figures_dir, dpi=200, transparent=True)


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
    plt.savefig("%s/coverage.png" % misc_figures_dir, dpi=200, transparent=True)

def misc_plots():

    from src.met4 import plot_timecourse
    from src.chromatin_summary_plots import (plot_combined_vs_xrate,
                                             plot_sul_prom_disorg,
                                             plot_occ_vs_xrate,
                                             plot_disorg_vs_xrate,
                                             plot_diosorg_vs_occ)
    from src.cross_correlation_kernel import MNaseSeqDensityKernel

    met4_dir = "%s/met4" % OUTPUT_DIR
    scatters_dir = "%s/scatters" % OUTPUT_DIR
    kernels_dir = "%s/kernels" % OUTPUT_DIR
    mkdirs_safe([met4_dir, scatters_dir, kernels_dir])

    nuc_kernel = MNaseSeqDensityKernel(filepath=nuc_kernel_path)
    nuc_kernel.plot_kernel(kernel_type='nucleosome')
    plt.savefig('%s/nuc_kernel.png' % (kernels_dir), dpi=150, transparent=True)

    sm_kernel = MNaseSeqDensityKernel(filepath=sm_kernel_path)
    sm_kernel.plot_kernel(kernel_type='small')
    plt.savefig('%s/sm_kernel.png' % (kernels_dir), dpi=150, transparent=True)

    from src.kernel_fitter import compute_triple_kernel
    triple_kernel = compute_triple_kernel(nuc_kernel)
    triple_kernel.plot_kernel(kernel_type='triple')
    plt.savefig('%s/triple_kernel.png' % (kernels_dir), dpi=150, transparent=True)

    from src.nucleosome_calling import plot_nuc_calls_cc
    plot_nuc_calls_cc()
    plt.savefig('%s/nuc_cross_cor_0_min.png' % (misc_figures_dir), dpi=150, transparent=True)

    # met4 plots
    plot_timecourse(datastore)
    plt.savefig('%s/met4_timecourse.png' % (met4_dir), dpi=150, transparent=True)

    plot_sul_prom_disorg(datastore)
    plt.savefig('%s/met4_scatter.png' % (met4_dir), dpi=150, transparent=True)

    selected_genes = ['HSP26', 'MET32', 'MET31', 'RPS7A', 'CKB1']

    # scatter plots
    plot_combined_vs_xrate(datastore, selected_genes)
    plt.savefig('%s/combined_vs_xrate.png' % (scatters_dir), dpi=150, transparent=True)
    
    plot_occ_vs_xrate(datastore, selected_genes)
    plt.savefig('%s/small_vs_xrate.png' % (scatters_dir), dpi=150, transparent=True)

    plot_disorg_vs_xrate(datastore, selected_genes)
    plt.savefig('%s/disorg_vs_xrate.png' % (scatters_dir), dpi=150, transparent=True)

    plot_diosorg_vs_occ(datastore, selected_genes)
    plt.savefig('%s/disorg_vs_small.png' % (scatters_dir), dpi=150, transparent=True)

    plot_ORFs_len(misc_figures_dir)
    
    plot_coverage(misc_figures_dir)


def antisense_plots():

    from src.antisense_analysis import plot_antisense_vs_sense
    from src.antisense_analysis import plot_bar_counts, plot_antisense_dist

    save_dir = '%s/antisense' % OUTPUT_DIR
    mkdirs_safe([save_dir])

    antisense_TPM = read_orfs_data('%s/antisense_TPM.csv' % rna_dir)
    antisense_TPM_logfold = read_orfs_data('%s/antisense_TPM_log2fold.csv' % rna_dir)

    plot_antisense_vs_sense(antisense_TPM_logfold, datastore.transcript_rate_logfold,
        120.0, highlight=['MET31', 'TAD2', 'CKB1', 
        'MET32', 'RPL31B', 'RPS0A'])
    plt.savefig('%s/sense_antisense_distr.png' % save_dir, dpi=150, transparent=True)
    
    plot_bar_counts(antisense_TPM_logfold, datastore.transcript_rate_logfold)
    plt.savefig('%s/sense_antisense_counts.png' % save_dir, dpi=150)

    plot_antisense_dist(antisense_TPM_logfold)
    plt.savefig('%s/antisense_logfc_dist.png' % save_dir, dpi=150)


def plot_heatmaps():

    from config import OUTPUT_DIR
    from src.chromatin_metrics_data import pivot_metric, ChromatinDataStore
    from src.chromatin_heatmaps import ChromatinHeatmaps

    print_fl(datastore.orfs.head())
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
                          write_path=("%s/sulfur.png" % write_dir),
                          fig_height=10,
                          aspect_scale=5000.,
                          highlight_max=[],
                          y_padding=1)
    heatmaps.plot_gene_names = False


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

    print_fl("\n------- GO Plots ----------\n")
    go_bar_plots()

    print_fl("\n------- Heatmap Plots ----------\n")
    plot_heatmaps()

    print_fl("\n------- Regression Plots ----------\n")
    regression_plots()

    print_fl("\n------- Antisense Plots ----------\n")
    antisense_plots()

    print_fl("\n------- Other ----------\n")
    misc_plots()

    print_fl("\n------- Antisense Plots ----------\n")
    antisense_plots()

    print_fl("\n------- Other ----------\n")
    misc_plots()


if __name__ == '__main__':
    main()
