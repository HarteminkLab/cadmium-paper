
import pandas
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hier
from scipy.spatial.distance import squareform
from scipy.spatial.distance import euclidean, pdist, squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


class HierarchicalClustering:

    def __init__(self, data, times=[0, 7.5, 15, 30, 60, 120]):
        self.data = data
        self.N = len(data)
        self.times = times

    def cluster(self, features, N_clusters, how='euclidean'):

        # set seed
        np.random.seed(123)

        N = self.N

        self.features = features
        if how == 'correlation':
            dist = 1 - features.T.corr()
            dist = squareform(np.matrix(dist.fillna(0.0)))
        elif how == 'hybrid':
            cor_dist = 1 - features.T.corr()
            cor_dist = squareform(np.matrix(cor_dist.fillna(0.0)))
            euc_dist = pdist(features, 'euclidean')
            dist = cor_dist*0.75 + euc_dist*0.25
        else:
            dist = pdist(features, 'euclidean')

        linkages = hier.linkage(dist, method='ward')
        ordering = seriation(linkages, N, N+N-2)
        ordering = list(reversed(ordering))

        self.N_clusters = N_clusters
        clusters = hier.cut_tree(linkages, N_clusters)

        orig_index_name = self.data.index.name
        data_reordered = self.data.copy()
        data_reordered['cluster'] = clusters
        data_reordered = data_reordered.reset_index().loc[ordering]
        data_reordered = data_reordered.set_index(orig_index_name)

        # change the cluster names so they appear in the order of the dataframe
        clusters = remap_cluster_names(data_reordered)
        data_reordered.cluster = clusters

        self.clustered_data = data_reordered

        self.calculate_cluster_counts()
        self.linkages = linkages

        self.clustered_data.cluster = self.clustered_data.cluster + 1
        self.clusters = self.clustered_data.cluster

    def calculate_cluster_counts(self):

        count_data = self.clustered_data.copy()
        count_data['count'] = 1

        cluster_counts = count_data.groupby('cluster')[['count']].count()
        cluster_counts['cumulative'] = cluster_counts['count']
        for i in range(1, self.N_clusters):
            cluster_counts.loc[i, 'cumulative'] = cluster_counts.loc[i-1]['cumulative'] + cluster_counts.loc[i]['count']
        self.cluster_counts = cluster_counts


def hier_cluster_sort(plot_data, cluster_columns):
    cluster_data = plot_data.copy()
    index_name = plot_data.index.name
    cor_dist = 1. - cluster_data[cluster_columns].T.corr()
    linkages = hier.linkage(squareform(np.matrix(cor_dist.fillna(0.0))), method='ward')
    N = len(cor_dist)
    ordering = seriation(linkages, N, N+N-2)
    sorted_data = cluster_data.reset_index().loc[ordering].set_index(index_name)
    return sorted_data


def remap_cluster_names(data):
    """Change cluster names to map in order of appearance in dataframe"""
    orig_clusters = data.cluster.unique()
    i = 0
    cmapping = {}
    for c in orig_clusters:
        cmapping[c] = i
        i+=1
    cmapping

    cluster_data = data[['cluster']]
    return data.cluster.map(lambda old: cmapping[old]).values


def seriation(Z, N, cur_index):
    '''
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z
        
    seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

