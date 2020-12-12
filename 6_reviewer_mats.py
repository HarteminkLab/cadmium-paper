

from src import plot_utils
from src.utils import mkdirs_safe
import numpy as np
from config import *
import matplotlib.pyplot as plt
import pandas as pd
from src.timer import Timer
from src.typhoon import TyphoonPlotter
from src import met4
from src.datasets import read_orfs_data
from src.chromatin_metrics_data import ChromatinDataStore
from src.utils import get_orf, print_fl
from src.reference_data import all_orfs_TSS_PAS, read_sgd_orfs
from src.transformations import difference
from src import shift_edge_analysis
from src.chromatin_summary_plots import (plot_xrate_vs_TPM,
                                         plot_combined_vs_TPM,
                                         plot_disorg_vs_TPM,
                                         plot_occ_vs_TPM)

timer = Timer()
datastore = ChromatinDataStore()

save_dir = '%s/reviewer_mats/' % OUTPUT_DIR
scatter_dpi = 200
selected_genes = ['HSP26', 'RPS7A', 'CKB1']


def main():

    print_fl("*******************************")
    print_fl("* 6    Reviewer Materials     *")
    print_fl("*******************************")

    print_preamble()

    mkdirs_safe([save_dir])

    plot_utils.apply_global_settings()

    # plots for shift edge analysis
    shift_edge_analysis.main()

    # additional scatter plots
    scatters()

    xrate_vs_TPM()

    # danpos
    danpos()

    # OD curve
    plot_OD_curve()


def plot_OD_curve():

    from src.plot_utils import apply_global_settings
    from src.colors import parula

    data = pd.read_csv('data/070513_cadmium.csv')
    data = data.set_index('Hour')

    apply_global_settings()

    fig = plt.figure(figsize=(8, 6))
    cols = list(reversed(data.columns[data.columns.str.startswith('ave ')]))
    colors = plt.get_cmap('magma_r')

    i = 0
    for col in cols:
        label = "%s uM" % (col.split(' ')[1])
        plt.plot(data.index, data[col], label=label, 
                 color=colors(0.1 + 0.9*(i*1./len(cols))),
                lw=2)
        i += 1

    plt.xlim(0, data.index.max())
    plt.ylim(0, 1.7)
    plt.legend(ncol=4)
    plt.suptitle("Growth curve, cadmium", fontsize=24)
    plt.ylabel("Optical density, OD$_{600}$", fontsize=18)
    plt.xlabel("Time, hours", fontsize=18)

    save_path = '%s/cadmium_growth.pdf' % (save_dir)
    plt.savefig(save_path, transparent=True, dpi=scatter_dpi)


def danpos():

    from src.dpos_bed import create_bed_for_dpos
    import os
    from src.utils import run_cmd

    working_dir = os.getcwd()

    danpos_output = '%s/danpos/' % (OUTPUT_DIR)
    mkdirs_safe([danpos_output])

    danpos_path = "%s/danpos-2.2.2/danpos.py" % working_dir

    # create DANPOS Bed file
    mnase = pd.read_hdf(mnase_seq_path, 'mnase_data')
    mnase = mnase[mnase.time == 0]

    # save_file = 'mnase_0.bed'
    # save_path = '%s/%s' % (danpos_output, save_file)
    # create_bed_for_dpos(mnase, save_path)
    # print_fl("Wrote %s" % save_path)

    # bash_command = "scripts/6_reviewer_mats/run_danpos.sh %s %s %s" % \
    #     (save_file, OUTPUT_DIR, danpos_path)
    # output, error = run_cmd(bash_command, stdout_file=None)

    danpos_calls_path = '%s/result/pooled/mnase_0.smooth.positions.xls' % \
        (danpos_output)
    danpos_positions = pd.read_csv(danpos_calls_path, sep='\t')

    plt.hist(danpos_positions[danpos_positions.smt_value < 10000].smt_value, bins=100)
    plt.savefig("%s/danpos_smt_pos.png" % danpos_output)

    danpos_positions = danpos_positions.sort_values('smt_value', 
        ascending=False)

    top_danpos = danpos_positions.head(2500)
    top_danpos = top_danpos.rename(columns={'chr': 'chromosome', 
        'smt_pos': 'position'})
    
    from src.chromatin import collect_mnase
    from src.kernel_fitter import compute_nuc_kernel

    nuc_kernel = compute_nuc_kernel(mnase, top_danpos)
    nuc_kernel.save_kernel("%s/danpos_kernel.json" % danpos_output)

    from src.kernel_fitter import  compute_triple_kernel
    nuc_kernel.plot_kernel(kernel_type='nucleosome')
    plt.savefig('%s/danpos_nuc_kernel.pdf' % (save_dir), transparent=True)

    triple_kernel = compute_triple_kernel(nuc_kernel)
    triple_kernel.plot_kernel(kernel_type='triple')
    plt.savefig('%s/danpos_triple_kernel.pdf' % (save_dir), transparent=True)


def xrate_vs_TPM():

    plot_xrate_vs_TPM(datastore)
    save_path = '%s/xrate_vs_tpm_half_life.pdf' % (save_dir)
    plt.savefig(save_path, transparent=True, 
        dpi=scatter_dpi)
    print_fl("Wrote %s" % save_path)

    for half_lifes in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        save_path = '%s/xrate_vs_tpm_half_life_%d_%d.pdf' % (save_dir, 
            half_lifes[0], half_lifes[1])
        plot_xrate_vs_TPM(datastore, half_lifes)
        plt.savefig(save_path, transparent=True, dpi=scatter_dpi)
        print_fl("Wrote %s" % save_path)

def scatters():

    from src.chromatin_summary_plots import plot_distribution
    from src.chromatin_metrics_data import ChromatinDataStore

    ind_names = [
        "$\\Delta_{t=120}$ promoter nucleosome occupancy",
        "$\\Delta_{t=120}$ small fragment occupancy",
        "$\\Delta_{t=120}$ pene body disorganization",
        "$\\Delta_{t=120}$ gene body nucleosome occupancy",

        "Average $\\Delta$ promoter nucleosome occupancy",
        "Average $\\Delta$ small fragment occupancy",
        "Average $\\Delta$ gene body disorganization",
        "Average $\\Delta$ gene body nucleosome occupancy",

        "Combined chromatin score",
    ]

    titles = [
    ]

    xs = [
        # measures @ 120
        datastore.promoter_nuc_occ_delta[120],
        datastore.promoter_sm_occupancy_delta[120],
        datastore.gene_body_disorganization_delta[120],
        datastore.gene_body_nuc_occ_delta[120],

        # mean measures
        datastore.promoter_nuc_occ_delta.mean(axis=1),
        datastore.promoter_sm_occupancy_delta.mean(axis=1),
        datastore.gene_body_disorganization_delta.mean(axis=1),
        datastore.gene_body_nuc_occ_delta.mean(axis=1),

        datastore.combined_chromatin_score
    ]
    
    y = datastore.sense_log2_TPM[120]

    for i in range(len(xs)):
        ind_name = ind_names[i]
        x = xs[i]
        ind_title_name = ind_name.replace('$\\Delta_{t=120}$', '120')
        save_title_name = ind_name.replace('$\\Delta_{t=120}$', '120')\
            .replace('$\\Delta', '')

        plot_distribution(x, y.loc[x.index], '%s' % ind_name, 
                                  'True Log$_2$ transcript level, TPM', 
                                  highlight=selected_genes,
                                  xlim=(-3, 3),
                                  ylim=(0, 16),
                                  title='%s\nvs transcript level @ 120 min' % ind_title_name,
                                  tight_layout=[0.1, 0.075, 0.9, 0.85],
                                  xticks=(-4, 4, 2),
                                  yticks=(0, 16, 5),
                                  plot_aux='cross')
        save_path = '%s/%s_vs_tpm.pdf' % (save_dir, save_title_name\
            .replace(' ', '_').lower())
        plt.savefig(save_path, transparent=True, 
            dpi=scatter_dpi)
        print_fl("Wrote %s" % save_path)

    save_path = '%s/combined_vs_TPM.pdf' % (save_dir)
    plot_combined_vs_TPM(datastore, selected_genes)
    plt.savefig(save_path, transparent=True, 
            dpi=scatter_dpi)

    save_path = '%s/disorg_vs_TPM.pdf' % (save_dir)
    plot_disorg_vs_TPM(datastore, selected_genes)
    plt.savefig(save_path, transparent=True, 
            dpi=scatter_dpi)

    save_path = '%s/small_occ_vs_TPM.pdf' % (save_dir)
    plot_occ_vs_TPM(datastore, selected_genes)
    plt.savefig(save_path, transparent=True, 
            dpi=scatter_dpi)

    from src.gp import plot_res_distribution_time
    from src.gp import GP
    from src.regression_compare import plot_compare_r2, load_results

    gp_dir = "%s/gp" % OUTPUT_DIR

    results = load_results(gp_dir)
    name = 'Full'

    save_path = '%s/gp_120.pdf' % (save_dir)
    time = 120
    cur = GP(name, results_path='%s/%s_results.csv' % (gp_dir, name))
    plot_res_distribution_time(cur, time, selected_genes=selected_genes, 
        show_pearsonr=True, plot_aux='none', show_r2=False, 
        tight_layout=[0.1, 0.075, 0.9, 0.85])
    plt.savefig(save_path, transparent=True, dpi=scatter_dpi)

if __name__ == '__main__':
    main()

