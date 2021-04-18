
import math
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hier
from src.gene_ontology import GeneOntology
from src.datasets import read_orfs_data
from src.plot_utils import apply_global_settings
from src.utils import get_orf_names, get_std_cutoff
from src.transformations import z_score_norm
from src.chromatin_metrics_data import ChromatinDataStore
from src.chromatin_heatmaps import ChromatinHeatmaps
from src.colors import parula
from src.hierarchical_clustering import HierarchicalClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from src import utils
from config import *
from src.utils import print_fl


class ChromatinClustering:

    def __init__(self):
        self.orfs = paper_orfs
        self.store = ChromatinDataStore()
        self.heatmaps = ChromatinHeatmaps(self.store)
        self.go = GeneOntology()

    def select_data(self, head=500):

        act_prm = self.store.promoter_sm_occupancy_delta
        act_dorg = self.store.gene_body_disorganization_delta

        mean_prm = act_prm.mean(axis=1).sort_values(ascending=False)
        mean_dorg = act_dorg.mean(axis=1).sort_values(ascending=False)

        prm_orfs = mean_prm.head(head).index
        dorg_orfs = mean_dorg.head(head).index

        self.data = self.store.chromatin_data.loc[set(prm_orfs).union(set(dorg_orfs))]

        def _inverse_quantile(array, val):
            """What is the quantile of the value in the array. (CDF)"""
            return np.mean(array <= val)

        prom_quantile = _inverse_quantile(mean_prm.values, mean_prm.values[head])
        dorg_quantile = _inverse_quantile(mean_dorg.values, mean_dorg.values[head])

        print_fl("Promoter ORFs: %d (%.1f%%)\n"
                 "Disorganization ORFs: %d (%.1f%%)" % \
                 (len(prm_orfs),  prom_quantile*100.,
                  len(dorg_orfs), dorg_quantile*100.
                  ))

        print_fl("%d genes total" % len(self.data))

    def determine_k(self, ks):

        df = pd.DataFrame({'k': ks}).set_index('k')
        df['silhouette_score'] = 0
        df['wss'] = 0
        df['n_go'] = 0
        df['avg_n_go'] = 0
        df['avg_n_go_per_cluster'] = 0
        df['num_clusters_with_go'] = 0
        df['min_fdr'] = 0
        df['max_fdr'] = 0
        df['avg_fdr'] = 0

        for k in ks:
            self.cluster(k=k)

            # mute logging
            self.run_go()

            n_go = len(self.clustered_go_sig)
            avg_n_go = n_go / float(k)
            avg_n_go_per_cluster = self.clustered_go_sig.groupby('cluster')\
                .count().mean().values[0]

            idx = self.hc.clustered_data.index
            data = self.hc.data.loc[idx]
            cluster_data = self.hc.clustered_data
            clusters = self.hc.clustered_data.cluster.values
            score = silhouette_score(data, clusters)
            wss_score = wss(cluster_data)

            num_clusters_with_go = len(self.clustered_go_sig.groupby('cluster')\
                .count())

            df.loc[k, 'n_go'] = n_go
            df.loc[k, 'silhouette_score'] = score
            df.loc[k, 'wss'] = wss_score
            df.loc[k, 'avg_n_go'] = avg_n_go
            df.loc[k, 'avg_n_go_per_cluster'] = avg_n_go_per_cluster
            df.loc[k, 'num_clusters_with_go'] = num_clusters_with_go

            fdr_bh = self.clustered_go_sig.fdr_bh
            fdr_bh = fdr_bh[fdr_bh < 1.0]

            df.loc[k, 'min_fdr'] = fdr_bh.min()
            df.loc[k, 'max_fdr'] = fdr_bh.max()
            df.loc[k, 'med_fdr'] = fdr_bh.median()


        self.k_metrics = df

    def plot_k_metrics(self, k=10):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
        fig.tight_layout(rect=[0.075, 0.03, 0.95, 0.8])
        plt.subplots_adjust(hspace=0.5, wspace=.7)

        ks = np.arange(self.k_metrics.index.min(), 
            self.k_metrics.index.max()+1, 2)
        xlim = ks[0], ks[-1]
        data = self.k_metrics

        ax1.plot(data.index, data.med_fdr,
         linewidth=2, color='#558DE7')
        ax1.axvline(k, color='red', linestyle='dotted', lw=2)
        ax1.set_xticks(ks)
        # ax1.set_yticks(np.arange(0.1, 0.5, 0.05))
        # ax1.set_ylim(0.1, 0.35)
        ax1.set_xlabel('# clusters', fontsize=14)
        ax1.set_ylabel('med_fdr', fontsize=14)

        ax2.plot(data.index, data.wss, linewidth=2, 
            color='#558DE7')
        ax2.axvline(k, color='red', linestyle='dotted', lw=2)
        ax2.set_xticks(ks)
        ax2.set_yticks(np.arange(0, 2000, 500))
        ax2.set_xlabel('# clusters', fontsize=14)
        ax2.set_ylabel('Average within-cluster\nsum of squares', fontsize=14)

        ax3.plot(data.index, data.num_clusters_with_go, linewidth=2, 
            color='#558DE7')
        ax3.axvline(k, color='red', linestyle='dotted', lw=2)
        ax3.set_xticks(ks)
        ax3.set_xlabel('# clusters', fontsize=14)
        ax3.set_ylabel('# of clusters with 1+ GO term', fontsize=14)

        ax4.plot(data.index, data.n_go, 
            linewidth=2, color='#558DE7')
        ax4.axvline(k, color='red', linestyle='dotted', lw=2)
        ax4.set_xticks(ks)
        ax4.set_xlabel('# clusters', fontsize=14)
        ax4.set_ylabel('# total GO terms', fontsize=14)

        ax1.set_xlim(*xlim)
        ax2.set_xlim(*xlim)
        ax3.set_xlim(*xlim)
        ax4.set_xlim(*xlim)

        ax1.set_title("Silhoutte score for k clusters", fontsize=24)

            
    def cluster(self, k=8):

        normalized_data = normalize_z(self.data)
        self.k = k

        self.hc = HierarchicalClustering(data=normalized_data)
        self.hc.cluster(features=normalized_data, N_clusters=k)

        self.clusters = sorted(self.hc.clustered_data.cluster.unique())
        clustered_data = self.hc.clustered_data

        groups = {}
        group_colors = {}
        group_names = []
        for cluster in self.clusters:
            cur_data = clustered_data[clustered_data.cluster == cluster]
            group_name = "Cluster %d" % cluster

            groups[group_name] = cur_data.index.values
            group_names.append(group_name)

        self.groups = groups
        self.group_colors = None
        self.group_names = group_names

    def run_go(self, sig=0.1):
        
        clustered_go = pd.DataFrame()
        clustered_go_sig = pd.DataFrame()
        clustered_data = self.hc.clustered_data

        for cluster in self.clusters:
            names = clustered_data[clustered_data.cluster == cluster][[]].\
                join(self.orfs[['name']])['name'].values

            self.go.run_go(names, sig=sig)
            
            cur_sig = self.go.results_sig_df.copy()
            cur_sig['cluster'] = cluster
            
            cur = self.go.results_df.copy()
            cur['cluster'] = cluster
            
            clustered_go_sig = clustered_go_sig.append(cur_sig)
            clustered_go = clustered_go.append(cur)

        self.clustered_go = clustered_go
        self.clustered_go_sig = clustered_go_sig


    def print_go(self):
        go_terms = {}

        for cluster in self.clusters:
            terms = list(self.clustered_go_sig[\
                self.clustered_go_sig['cluster'] == cluster]['name'].values)
            
            # remove high-level terms
            drop_items = {'molecular_function', 
                          'cytoplasm', 
                          'cellular_component',
                          'biological_process',
                          'nucleolus',
                          'cytoplasmic vesicle'}
            terms = set(terms) - drop_items
            terms = [t[0].upper() + t[1:] for t in terms]
            go_value = ('Cluster %d\n' % (cluster)) + '\n'.join(list(terms))
            go_terms[str(cluster)] = go_value

            if len(terms) > 0:
                print_fl(go_value)
                print_fl("-------------------")

        self.go_terms = go_terms

    def plot_heatmap(self, clusters=None, plot_dend=False, write_path=None):

        if clusters is None:
            self.heatmaps.show_saved_plot = True

            if plot_dend:
                fig, (hm_ax, den_ax) = plt.subplots(1, 2, figsize=(14, 8))
                fig.tight_layout()

                den_ax.set_xticks([])
                den_ax.set_yticks([])
                den_ax.axis('off')
            else: 
                fig, hm_ax = plt.subplots(1, 1, figsize=(8, 8))
                fig.tight_layout()
                fig.patch.set_alpha(0.0)
                hm_ax.patch.set_alpha(0.0)

            orf_names = self.hc.clustered_data.index.values
            self.heatmaps.plot_heatmap(
                orf_names=orf_names,
                orf_groups=self.groups, 
                group_names=self.group_names, 
                group_colors=self.group_colors, aspect_scale=40.,
                ax=hm_ax,
                write_path=write_path, 
                group_spacing=2)
            if plot_dend:
                        
                _ = hier.dendrogram(self.hc.linkages, color_threshold=0, 
                    link_color_func=(lambda _: '#000000'), 
                   orientation='right', ax=den_ax)

        elif clusters is not None:
            cdata = self.hc.clustered_data
            orf_names = cdata[cdata.cluster.isin(clusters)].index.values
            
            group_names = ["Cluster %d" % c for c in clusters]
            
            self.heatmaps.plot_heatmap(
                            orf_names=orf_names,
                            orf_groups=self.groups, 
                            group_names=group_names, 
                            group_colors=self.group_colors, aspect_scale=20.,
                            write_path=write_path)

    def plot_antisense_heatmap(self, orf_names=None):
        clustered_data = self.hc.clustered_data
        clustered_data = clustered_data.sort_values('cluster')

        antisense_TPM = read_orfs_data('%s/antisense_TPM.csv' % rna_dir)
        antisense_TPM = antisense_TPM.loc[clustered_data.index]

        if orf_names is not None:
            antisense_TPM = antisense_TPM.loc[orf_names]

        plt.figure(figsize=(4, 10))
        plot_data = np.log2(antisense_TPM+1)

        plt.imshow(plot_data, aspect=40./len(plot_data), 
            vmin=0, vmax=12, cmap='viridis')
        plt.yticks([])
        plt.xticks([])


    def plot_cluster(self, sm_ax, dis_ax, x_ax, cluster, title='', xlab=False):

        def _plot_fill(ax, x, mean, lower, upper, color, bg_color):
            ax.plot(x, mean, color=color, lw=3)
            ax.fill_between(x, lower, 
                upper, color=bg_color)
            ax.set_xlim(0, 5)
            ax.set_ylim(-3, 3)
            ax.axhline(y=0, linewidth=2, color='black', linestyle='solid')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('white')

        data = self.hc.clustered_data
        data = data[data.cluster == cluster].copy()
        data = normalize_z(data)
        xrate = self.store.transcript_rate_logfold
        xrate = data[[]].join(xrate)
        xrate = normalize_z(xrate)

        mean = data.mean()
        upper = data.quantile(0.95)
        lower = data.quantile(0.05)
        
        times=[0.0, 7.5, 15, 30, 60, 120]
        x = np.arange(len(times))
        
        disorg_color = '#138E88'
        sm_color = '#D16207'
        xrate_color = '#5D2CA3'

        cols = data.columns
        prom_cols = cols[0:6]
        dorg_cols = cols[6:12]
    
        _plot_fill(sm_ax, x, mean[prom_cols], lower[prom_cols], 
            upper[prom_cols], sm_color, '#f4d6c1')
        _plot_fill(dis_ax, x, mean[dorg_cols], lower[dorg_cols], 
            upper[dorg_cols], disorg_color, '#d0e8e6')

        xrate_mean = xrate.mean().values
        xrate_upper = xrate.quantile(0.75)
        xrate_lower = xrate.quantile(0.25)

        _plot_fill(x_ax, x, xrate_mean, xrate_upper, 
            xrate_lower, '#8050CC', '#B7AACE')

        dis_ax.set_title(title, fontsize=24)

        if xlab:
            sm_ax.set_xlabel("Small fragment\noccupancy", fontsize=16, labelpad=10)
            dis_ax.set_xlabel("Nucleosome\ndisorganization", fontsize=16, labelpad=10)
            x_ax.set_xlabel("Transcript\nrate", fontsize=16, labelpad=10)

    def plot_antisense(self, antisense=None):

        apply_global_settings(titlepad=45)

        cluster_data = self.hc.clustered_data
        from src.datasets import read_orfs_data

        if antisense is None:
            antisense = read_orfs_data('%s/antisense_TPM.csv' % rna_dir)

        data = antisense.loc[cluster_data.index]
        data = data.join(cluster_data[['cluster']])

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.tight_layout(rect=[0.05, 0.1, 0.95, 0.8])
        times = [0.0, 7.5, 15, 30, 60, 120]

        num_clusters = len(data.cluster.unique())

        for c in range(1, num_clusters+1):

            c_data = data[data.cluster == c][times]
            for i in range(len(times)):
                time = times[i]
                cur = c_data[time].values
                lower = np.quantile(cur, 0.75)
                upper = np.quantile(cur, 0.25)
                median = np.median(cur)

                spacing = 0.13
                x = c+spacing*i - spacing*2.5
                ax.plot([x, x], [lower, upper], linewidth=3., 
                    color='#FF5C5C', alpha=1, solid_capstyle='butt')
                ax.scatter(x, median, s=6, marker='D', color='black', zorder=10)

        ticks = np.arange(num_clusters+1)
        ax.set_xticks(ticks)
        ax.set_xlim(0.5, num_clusters+0.5)
        # ax.set_yticks(np.arange(0, 40, 10))

        ax.tick_params(axis='x', length=0, pad=10, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_ylabel('Transcripts per million', fontsize=18)
        ax.set_xlabel('Cluster', fontsize=18)
        ax.set_title('Antisense transcripts per cluster', fontsize=23)

        for x in np.arange(1, num_clusters):
            ax.axvline(x + 0.5, color='#d0d0d0', linewidth=1)

    def plot_half_lifes(self):

        apply_global_settings(titlepad=20)

        cluster_data = self.hc.clustered_data
        from src.datasets import read_orfs_data
        half_lifes = read_orfs_data('data/half_life.csv')[['half_life']]

        data = half_lifes.loc[cluster_data.index]
        data = data.join(cluster_data[['cluster']])

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.tight_layout(rect=[0.05, 0.1, 0.95, 0.8])
        times = [0.0, 7.5, 15, 30, 60, 120]

        for c in range(1, 8):
            cur = data[data.cluster == c].half_life
            lower = np.quantile(cur, 0.75)
            upper = np.quantile(cur, 0.25)
            median = np.median(cur)

            spacing = 0.13
            x = c+spacing*3 - spacing*2.5
            ax.plot([x, x], [lower, upper], linewidth=6., 
                color='#abd1fc', alpha=1, solid_capstyle='butt')
            ax.scatter(x, median, s=16, marker='D', color='black', zorder=10)

        ticks = np.arange(8)
        ax.set_xticks(ticks)
        ax.set_xlim(0.5, 7.5)
        # ax.set_yticks(np.arange(0, 200, 50))
        ax.set_ylim(0, 50)

        ax.tick_params(axis='x', length=0, pad=10, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_ylabel('Half life, min', fontsize=18)
        ax.set_xlabel('Cluster', fontsize=18)
        ax.set_title('Half lifes per cluster', fontsize=30)

        for x in np.arange(1, 8):
            ax.axvline(x + 0.5, color='#d0d0d0', linewidth=1)

def plot_cluster_lines(chrom_clustering):

    apply_global_settings(titlepad=15)

    cluster_data = chrom_clustering.hc.clustered_data
    clusters = cluster_data.cluster.unique()
    n_clusters = len(clusters)

    fig, axs = plt.subplots(n_clusters, 3, figsize=(3*2, n_clusters*2.2))
    _ = axs
    fig.patch.set_alpha(0.0)

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.75, wspace=0.25)
        
    for i in np.arange(n_clusters):
        axs_row = axs[i]
        cluster = clusters[i]
        title = 'Cluster %d' % cluster
        chrom_clustering.plot_cluster(axs_row[0], axs_row[1], axs_row[2], 
            cluster, title=title, xlab=(i == n_clusters-1))

        for ax in axs_row:
            ax.patch.set_alpha(1.0)
            ax.patch.set_facecolor('white')


def go_bar_plot(chrom_clustering):
    terms = chrom_clustering.clustered_go
    terms = terms[terms.fdr_bh < 0.1]
    terms = terms.sort_values(['cluster', 'fdr_bh'], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.tight_layout(rect=[0.3, 0.1, 0.8, 0.9])

    term_labels = terms['name']
    term_labels = [l[0:1].upper() + l[1:] for l in term_labels]
    fdr = -np.log(terms['fdr_bh'])
    y = np.arange(0, -len(term_labels), -1)

    color_map = ['#05668D', '#9BC53D', 'gray']

    i = 0
    last_clust = None
    colors = []
    for idx, term in terms.iterrows():
        
        if last_clust is None: last_clust = term.cluster
        if term.cluster != last_clust:
            i += 1
            last_clust = term.cluster

        colors.append(color_map[i])

        ax.text(fdr[idx]+0.1,
                y[idx], "10$^{-%.1f}$" % fdr[idx],
                ha='left', va='center', fontsize=13,
                fontdict={'fontname': 'Open Sans'})

    ax.barh(y=y, width=fdr, height=0.5, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(term_labels)

    clus_ax = ax.twinx()

    clust_counts = terms.groupby('cluster').count()[['name']].rename(columns={'name':'count'})
    cumulative_sum = 0
    clust_label_pos = []
    clust_labels = []
    i = 0
    for idx, c in clust_counts.iterrows():
        clust_label_pos.append(-(cumulative_sum + c['count']/2.)+0.5)
        clust_labels.append(idx)
        cumulative_sum += c['count']
        if i < len(clust_counts)-1:
            ax.axhline(-cumulative_sum + 0.5, color='#dedede')
        i+=1

    clus_ax.set_yticks(clust_label_pos)
    clus_ax.set_yticklabels(['Cluster %d' % c for c in clust_labels])
    clus_ax.set_ylim(-len(terms)+0.5, 0.5)

    clus_ax.tick_params(axis='y', length=0, pad=10, labelsize=16)
    ax.tick_params(axis='y', length=0, pad=10, labelsize=14)
    ax.set_xlabel('Log$_{10}$ FDR')
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_xlim(0, np.max(fdr)+3)

    ax.set_ylim(-len(terms)+0.5, 0.5)
    plt.suptitle("Clustered GO terms", fontsize=28)


def plot_silhouettes(chrom_clustering):

    cluster_data = chrom_clustering.hc.clustered_data

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 8))
    fig.tight_layout(rect=[0.2, 0.03, 0.8, 0.8])

    cluster_labels = cluster_data.cluster.values
    clusters_unique = sorted(cluster_data.cluster.unique(), reverse=True)
    n_clusters = cluster_labels.max()
    X = cluster_data[cluster_data.columns[:-1]]

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    cmap = parula()

    padding = 10
    y_lower = padding
    yticks = []
    yticklabels = []
    avgs = []
    total_sil = 0
    for i in clusters_unique:

        color = cmap(float(i) / n_clusters * 0.8)

        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        yticks.append(y_lower + 0.5 * size_cluster_i)
        yticklabels.append("Cluster %d" % (i))

        mean_sil = ith_cluster_silhouette_values.mean()
        total_sil += ith_cluster_silhouette_values.sum()
        avgs.append("%.2f" % mean_sil)

        if i < n_clusters:
            ax1.axhline(y_lower-padding/2., color='#e0e0e0', lw=1.5)

        # Compute the new y_lower for next plot
        y_lower = y_upper + padding  # 10 for the 0 samples

    twinax = ax1.twinx()
    twinax.set_yticks(yticks)
    twinax.set_yticklabels(yticklabels)
    ax1.set_ylabel("Average silhoutte score", fontsize=14)

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(avgs)
    twinax.tick_params(axis='y', length=0, pad=10, labelsize=16)
    ax1.tick_params(axis='y', length=0, pad=10, labelsize=13)
    ax1.set_xlabel("Silhoutte score", fontsize=16)

    ylim = 0, len(X)+padding*n_clusters+padding/2
    ax1.set_ylim(*ylim)
    twinax.set_ylim(*ylim)
    ax1.set_title("Silhoutte scores, k=%d\nAverage score=%.2f" % 
        (n_clusters, total_sil/len(cluster_data)),
     fontsize=24)
    return fig

def cluster_ss(cluster_data, cluster):
    """Calculate within-cluster sum of squares"""
    cluster_data = cluster_data[cluster_data.cluster == cluster]
    cluster_data = cluster_data[cluster_data.columns[:-1]]
    
    centroid = cluster_data.mean()
    dist_centroid = cluster_data - centroid

    sum_squares_per_feature = dist_centroid.applymap(lambda x: x**2).sum()
    cluster_sum_squares = sum_squares_per_feature.sum()
    return cluster_sum_squares

def wss(cluster_data):
    """Calculate average within-cluster sum of squares"""
    ks = cluster_data.cluster.unique()
    wss = 0
    for k in ks:
        ss = cluster_ss(cluster_data, k)
        wss += ss
    return wss/len(ks)

def normalize_z(data):
    data = data.copy()
    cols = data.columns
    prom_cols = cols[0:6]
    dorg_cols = cols[6:12]
    prom_z = z_score_norm(data[prom_cols])
    dorg_z = z_score_norm(data[dorg_cols])
    data.loc[:, prom_cols] = prom_z
    data.loc[:, dorg_cols] = dorg_z
    return data

