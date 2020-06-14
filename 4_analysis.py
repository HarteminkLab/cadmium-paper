

from src import plot_utils
from src.utils import mkdirs_safe
import numpy as np
from config import *
import matplotlib.pyplot as plt
import pandas as pd
from src.timer import Timer

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
    go_disorg_analysis.terms_res.to_csv(save_path)

    go_org_analysis = GOChromatinAnalysis(datastore, agg_fun=np.mean)
    go_org_analysis.subset_genes(tail=300)
    go_org_analysis.run_go(fdr_sig=1e-5)

    # save org results
    save_path = '%s/org_terms.csv' % write_dir
    go_org_analysis.collect_counts()
    go_org_analysis.terms_res.to_csv(save_path)


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
    plt.savefig('%s/clusters_choose_k.png' % write_dir, dpi=150, transparent=True)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Clustering k=%d" % k)
    chrom_clustering.cluster(k)
    chrom_clustering.group_colors = None
    chrom_clustering.plot_heatmap(write_path='%s/heatmap_activated_%d.png'
        % (write_dir, k))
    chrom_clustering.run_go()

    go_bar_plot(chrom_clustering, cluster=cluster, height=5, color=color)
    plt.savefig('%s/clustered_go.png' % (write_dir), dpi=150, transparent=True)

    chrom_clustering.plot_antisense()
    plt.savefig('%s/cluster_antisense_transcripts.png' % write_dir, dpi=200, transparent=True)

    from src.chrom_clustering import plot_cluster_lines

    plot_cluster_lines(chrom_clustering)
    plt.savefig('%s/cluster_lines.png' % write_dir, dpi=150, transparent=False)

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


def main():

    print_fl("*********************")
    print_fl("* 4    Analysis     *")
    print_fl("*********************")

    print_preamble()

    plot_utils.apply_global_settings()

    # gene ontology
    # print_fl("\n------- Gene Ontology ----------\n")
    # gene_ontology_analysis()

    # clustering
    print_fl("\n------- Clustering ----------\n")
    cluster_analysis()

    # regression
    print_fl("\n------- Regression ----------\n")
    # print_fl("TODO: Save time....Skipping...")
    # regression()



if __name__ == '__main__':
    main()
