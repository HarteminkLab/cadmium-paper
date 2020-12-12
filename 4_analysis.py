

from src import plot_utils
from src.utils import mkdirs_safe, write_str_to_path
import numpy as np
from config import *
import matplotlib.pyplot as plt
import pandas as pd
from src.timer import Timer
from src.reference_data import all_orfs_TSS_PAS

all_orfs = all_orfs_TSS_PAS()
timer = Timer()


def gene_ontology_analysis():

    from src.go_analysis import GOChromatinAnalysis
    from src.chromatin_metrics_data import ChromatinDataStore

    write_dir = "%s/go" % OUTPUT_DIR
    mkdirs_safe([write_dir])

    datastore = ChromatinDataStore()
    go_disorg_analysis = GOChromatinAnalysis(datastore, agg_fun=np.mean)
    go_disorg_analysis.subset_genes(head=300)
    go_disorg_analysis.run_go(fdr_sig=0.2)

    # save results
    save_path = '%s/disorg_terms.csv' % write_dir
    go_disorg_analysis.collect_counts()
    go_disorg_analysis.get_go_terms_sorted().to_csv(save_path, index=False)
    latext_str = go_disorg_analysis.get_latex_table()
    write_str_to_path(latext_str, '%s/disorg_latex_table.txt' % write_dir)

    go_org_analysis = GOChromatinAnalysis(datastore, agg_fun=np.mean)
    go_org_analysis.subset_genes(tail=300)
    go_org_analysis.run_go(fdr_sig=1e-5)

    # save org results
    save_path = '%s/org_terms.csv' % write_dir
    go_org_analysis.collect_counts()
    go_org_analysis.get_go_terms_sorted().to_csv(save_path, index=False)
    latext_str = go_org_analysis.get_latex_table()
    write_str_to_path(latext_str, '%s/org_latex_table.txt' % write_dir)


def tf_analysis():

    from src.reference_data import calculate_promoter_regions
    from src.small_peak_calling import SmallPeakCalling

    calculate_promoter_regions()

    small_peaks = SmallPeakCalling()

    small_peaks.collect_peaks()
    small_peaks.link_peaks()
    small_peaks.collect_motifs()
    small_peaks.save_data()


def cluster_analysis():

    from src.chrom_clustering import ChromatinClustering
    from src.chrom_clustering import go_bar_plot

    write_dir = "%s/clusters" % OUTPUT_DIR
    mkdirs_safe([write_dir])

    chrom_clustering = ChromatinClustering()
    chrom_clustering.select_data()

    k = 8

    # try different k values
    print_fl("Determining k...")
    ks = np.concatenate([
        np.arange(2, 14, 2)
        ])
    chrom_clustering.determine_k(ks)
    chrom_clustering.k_metrics.to_csv('%s/k_metrics.csv' % write_dir)
    chrom_clustering.plot_k_metrics(k=k)
    plt.savefig('%s/clusters_choose_k.pdf' % write_dir, transparent=True)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Clustering k=%d" % k)
    chrom_clustering.cluster(k)
    chrom_clustering.group_colors = None
    chrom_clustering.plot_heatmap(write_path='%s/heatmap_activated_%d.png'
        % (write_dir, k))
    chrom_clustering.run_go()

    go_bar_plot(chrom_clustering)
    plt.savefig('%s/clustered_go.pdf' % (write_dir), transparent=True)

    chrom_clustering.plot_antisense()
    plt.savefig('%s/cluster_antisense_transcripts.png' % write_dir, transparent=True)

    print_fl("Done.")
    timer.print_time()
    print_fl()


def regression():

    from src.gp_models import run_models

    mkdirs_safe([gp_dir])

    # run model
    print_fl("Running GP models...", end='')
    run_models(gp_dir, timer)

    print_fl("Done.")
    timer.print_time()
    print_fl()

def fimo_analysis():

    from src.fimo import FIMO, find_motif
    from src.utils import get_gene_named

    fimo = FIMO()

    from src.fimo import find_relevant_gene_motifs
    write_path = '%s/found_motifs.csv' % mnase_dir

    found_motifs = find_relevant_gene_motifs(fimo, all_orfs)
    found_motifs = all_orfs[['name']].reset_index().merge(found_motifs, 
        left_on='orf_name', right_on='target').rename(columns={'name':'target_name'})
    found_motifs.to_csv(write_path)
    print_fl("Wrote %s" % write_path)


def main():

    print_fl("*********************")
    print_fl("* 4    Analysis     *")
    print_fl("*********************")

    print_preamble()

    plot_utils.apply_global_settings()

    print_fl("\n------- FIMO ----------\n")
    fimo_analysis()

    print_fl("\n------- Gene Ontology ----------\n")
    gene_ontology_analysis()

    print_fl("\n------- Clustering ----------\n")
    cluster_analysis()

    print_fl("\n------- Regression ----------\n")
    regression()

    print_fl("\n------- TF Analysis ----------\n")
    tf_analysis()


if __name__ == '__main__':
    main()
