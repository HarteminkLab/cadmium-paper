
from pandas import Series
import pandas as pd
import numpy as np
from src.timer import Timer
import matplotlib.pyplot as plt
from src.utils import print_fl, mkdirs_safe
from src.utils import get_orf, get_orf_name
from src.reference_data import read_park_TSS_PAS, read_sgd_orfs, \
    all_orfs_TSS_PAS
from config import times, cc_sense_chrom_dir, paper_orfs
from src.reference_data import read_park_TSS_PAS, read_sgd_orfs, \
                               all_orfs_TSS_PAS, load_calculated_promoters
from src.plot_utils import plot_density_scatter, apply_global_settings, \
    hide_spines
from src.colors import parula
from src.fimo import FIMO, find_motif
from src.utils import get_gene_named
from src.timer import Timer
from sklearn.preprocessing import scale
from src.transformations import difference
from src.transformations import log2_fold_change
from src.colors import red, blue, purple, parula
from config import OUTPUT_DIR
from src.datasets import read_orfs_data
from src.chromatin_metrics_data import ChromatinDataStore
from src.plot_utils import plot_rect


class SmallPeakCalling:

    def __init__(self):

        self.all_orfs = all_orfs_TSS_PAS()
        self.fimo = FIMO()
        self.window = 80
        self.window_2 = self.window/2

        # plotting
        self.bin_scale = 14.
        self.im_scale = 7
        self.tf_threshold = 0.1
        self.fc_threshold = 1

        self.datastore = ChromatinDataStore()


    def save_data(self):
        save_dir = '%s/tf_analysis' % OUTPUT_DIR
        mkdirs_safe([save_dir])

        print_fl("Saving %s" % save_dir)
        self.all_peaks.to_csv('%s/all_peaks.csv' % save_dir)
        self.linked_peaks_normalized.to_csv('%s/linked_peaks_norm.csv' % save_dir)
        self.linked_peaks.to_csv('%s/linked_peaks.csv' % save_dir)
        self.prom_peaks.to_csv('%s/prom_peaks.csv' % save_dir)
        self.all_motifs.to_csv('%s/all_motifs.csv' % save_dir)


    def load_data(self):
        save_dir = '%s/tf_analysis' % OUTPUT_DIR

        self.all_peaks = pd.read_csv('%s/all_peaks.csv' % save_dir)\
            .set_index('name')
        self.linked_peaks = read_orfs_data(
            '%s/linked_peaks.csv' % save_dir, 'name')
        self.linked_peaks_normalized = read_orfs_data(
            '%s/linked_peaks_norm.csv' % save_dir, 'name')
        self.prom_peaks = pd.read_csv('%s/prom_peaks.csv' % save_dir)\
            .set_index('Unnamed: 0')
        self.all_motifs = pd.read_csv('%s/all_motifs.csv' % save_dir)\
            .set_index('Unnamed: 0')

        # rename tf motifs to match gene names
        data = self.all_motifs

        rename = {
            'RCS1': 'AFT1',
            'YML081W': 'TDA9'
        }

        for k, v in rename.items():
            selected = data.tf == k
            data.loc[selected, 'tf'] = v

        self.summarize_tfs()


    def collect_peaks(self):
        self.all_peaks = collect_small_peaks()

    def link_peaks(self):

        all_peaks = self.all_peaks
        # select highest 10% of peaks
        q = np.quantile(all_peaks.cross_correlation, 0.9)
        print("Peak cutoff %.1f" % q)

        # Remove duplicate peaks across times
        test_peaks = all_peaks[all_peaks.cross_correlation > q]

        timer = Timer()

        collect_peaks = pd.DataFrame()

        test_peaks = test_peaks.sort_values('cross_correlation', ascending=False)

        window_2 = self.window_2
        while len(test_peaks) > 0:
            
            highest = test_peaks.reset_index().loc[0]
            selected_near = test_peaks[(test_peaks.chr == highest.chr) & 
                                       (test_peaks.original_mid < highest.original_mid + window_2) & 
                                       (test_peaks.original_mid > highest.original_mid - window_2)]
            test_peaks = test_peaks.drop(selected_near.index)
            collect_peaks = collect_peaks.append(highest)

            if len(collect_peaks) % 100 == 0:
                print("%d, (-%d) - %s" % (len(collect_peaks), len(test_peaks), timer.get_time()))

        self.collected_peaks = collect_peaks
        timer = Timer()

        test_peaks = collect_peaks.set_index('name')
        linked_peaks = test_peaks[[]].copy()

        for time in times:
            linked_peaks[time] = 0.0

        i = 0
        for chrom in range(1, 17):    
            
            chrom_peaks = test_peaks[test_peaks.chr == chrom]
            
            if len(chrom_peaks) == 0: continue

            chrom_cross_correlation = pd.read_hdf(
                    '%s/cross_correlation_chr%d.h5.z' % 
                    (cc_sense_chrom_dir, chrom))

            for idx, peak in chrom_peaks.iterrows():
                cols = np.arange(peak.mid-window_2, peak.mid+window_2)
                
                try:
                    peak_cc = chrom_cross_correlation.loc['small'].loc[peak.orf][cols].mean(axis=1)
                except KeyError:
                    continue

                linked_peaks.loc[idx] = peak_cc

                if i % 100 == 0:
                    print("%d/%d - %s" % (i, len(test_peaks), timer.get_time()))
                i += 1

        self.linked_peaks = linked_peaks

        # normalize linked_peaks
        linked_peaks_normlized = linked_peaks.copy()
        value_0 = linked_peaks[0.0].copy()
        for time in times[1:]:
            # normalize to t=0's mean
            values = linked_peaks[time] + (value_0.mean() - linked_peaks[time].mean())
            linked_peaks_normlized[time] = values
        self.linked_peaks_normalized = linked_peaks_normlized

    def collect_motifs(self):
        
        fimo = self.fimo
        timer = Timer()

        all_motifs = pd.DataFrame()

        # filter peaks outside of ORFs promoter
        promoters = load_calculated_promoters()

        search_peaks = self.collected_peaks.reset_index(drop=True).copy()
            
        # filter out peaks outside of promoters
        print("Filtering peaks outside of promoters")
        print(len(search_peaks))
        for orf_name, row in promoters.iterrows():
            
            cur_peaks = search_peaks[search_peaks.orf == orf_name]
            
            if len(cur_peaks) > 0:
                
                # remove if outside of promoter   
                remove_peaks = cur_peaks[(cur_peaks.original_mid > row.promoter_stop) | 
                                         (cur_peaks.original_mid < row.promoter_start)]
                search_peaks = search_peaks.drop(remove_peaks.index)

        self.prom_peaks = search_peaks
        print(len(search_peaks))

        for idx, peak in search_peaks.reset_index().iterrows():
            search_window = (peak.original_mid-50, peak.original_mid+50)

            try:
                motifs = find_motif(fimo, None, peak.chr, search_window)
            except KeyError:
                continue

            motifs['orf'] = peak.orf
            motifs['peak'] = peak['name']
            motifs = motifs[['orf', 'tf', 'score', 'p-value', 'q-value', 'motif_mid', 
                             'strand', 'peak']]
            all_motifs = all_motifs.append(motifs)

            if idx % 100 == 0:
                print("%d/%d - %s" % (idx, len(search_peaks), timer.get_time()))
        all_motifs = all_motifs.reset_index(drop=True)
        self.all_motifs = all_motifs


    def summarize_tfs(self):
        motif_tfs = self.all_motifs
        self.tf_means_df = get_summary_df(self.linked_peaks_normalized, motif_tfs, 
            lambda x: np.mean(x))

        self.filter_threshold()

        datastore = self.datastore
        all_motifs = self.all_motifs
        regulon_xrate = all_motifs[['tf', 'orf']].merge(
            datastore.transcript_rate_logfold.reset_index(), 
            left_on='orf', right_on='orf_name')[['tf', 'orf_name'] + times]
        self.regulon_xrate = regulon_xrate
        self.mean_regulon_xrate = self.regulon_xrate.groupby('tf').mean()

        def _select_regulon_from_peaks(small_peaks, selected_peaks):
            """Select a subset of the regulon xrates from peaks. For selecting
            highest and lowest threshold peaks"""
            all_motifs = small_peaks.all_motifs
            selected_orfs = all_motifs[all_motifs.peak.isin(selected_peaks)].orf

            regulon_xrate = small_peaks.regulon_xrate
            selected_regulon_xrates = regulon_xrate[regulon_xrate.orf_name.isin(selected_orfs)]

            return selected_regulon_xrates, \
                   selected_regulon_xrates.groupby('tf').mean()

        # Set the TF set by if there is transcription rate and regulon for the TF
        # sorted by the bin occupancy change
        all_xrates = self.datastore.transcript_rate_logfold

        tfs_sorted = self.tf_means_df.loc[self.tf_means_df.mean(axis=1).sort_values().index]
        tfs_sorted = tfs_sorted.join(paper_orfs[['name']].reset_index().set_index('name'), how='inner')

        selected = tfs_sorted[(tfs_sorted.orf_name.isin(all_xrates.index)) &
                              (tfs_sorted.index.isin(self.mean_regulon_xrate.index))]

        self.tf_set = selected[['orf_name']]

        tf_mean_means = self.tf_means_df[self.tf_means_df.index\
            .isin(self.tf_set.index)].copy()
        tf_mean_means['mean'] = tf_mean_means.mean(axis=1)
        tf_mean_means = tf_mean_means.sort_values('mean')
        self.tf_mean_means = tf_mean_means


    def filter_threshold(self):

        tf_mean = self.tf_means_df.mean(axis=1)
        tf_mean = tf_mean.sort_values()

        selected_high = tf_mean[tf_mean > self.tf_threshold]
        selected_low = tf_mean[tf_mean < -self.tf_threshold]

        self.selected_high_tfs = selected_high
        self.selected_low_tfs = selected_low

        self.view_high = 15
        self.view_low = 10



def call_orf_small_peaks(cross_correlation, orf):

    ret_peaks_df = pd.DataFrame()
    times = [0.0, 7.5, 15, 30, 60, 120]

    for time in times:
        idx = orf.name
        data = cross_correlation.loc[idx].loc[time]
        peaks_df = call_peaks(data).copy()
        peaks_df['original_mid'] = peaks_df.mid 

        if orf.strand == '-': 
            peaks_df['original_mid'] = -peaks_df.mid

        peaks_df.loc[:, 'original_mid'] = peaks_df.original_mid + orf.TSS
        peaks_df['original_mid'] = peaks_df.original_mid.astype(int)
        peaks_df['mid'] = peaks_df.mid.astype(int)

        peaks_df['time'] = time
        peaks_df['orf'] = idx
        peaks_df['chr'] = orf.chr
        peaks_df = peaks_df.sort_values(['mid'])

        ret_peaks_df = ret_peaks_df.append(peaks_df)

    return ret_peaks_df.reset_index(drop=True)


def call_peaks(data):

    cutoff = 0.02
    window = 80
    window_2 = window/2

    cur_data = data.sort_values(ascending=False).copy()
    cur_data.index[cur_data.index > 50]

    peaks_df = pd.DataFrame()
    last_nuc = float('inf')
    while len(cur_data) > 0 and last_nuc > cutoff:
        
        highest_idx = cur_data.index.values[0]
        highest = cur_data.loc[highest_idx]
        
        remove_span = highest_idx-window_2, highest_idx+window_2
        
        drop_idx = cur_data.index[(cur_data.index > remove_span[0]) & 
            (cur_data.index <= remove_span[1])]
        cur_data = cur_data.drop(drop_idx)
        
        peaks_df = peaks_df.append(Series({'cross_correlation':highest,
            'mid':highest_idx}), ignore_index=True)
        last_nuc = highest

    return peaks_df


def collect_small_peaks():

    from src.small_peak_calling import call_orf_small_peaks
    from src.timer import Timer
    
    orfs = all_orfs_TSS_PAS()

    timer = Timer()
    all_peaks = pd.DataFrame()

    for chrom in range(1, 17):

        print("Chromosome %d" % chrom)
        chr_orfs = orfs[orfs.chr == chrom]

        # load relevant cross correlations
        chrom_cross_correlation = pd.read_hdf(
        '%s/cross_correlation_chr%d.h5.z' % 
        (cc_sense_chrom_dir, chrom))
        small_cc = -1 * chrom_cross_correlation.loc['diff']
        
        for idx, orf in chr_orfs.iterrows():
            
            try:
                peaks = call_orf_small_peaks(small_cc, orf)
            except KeyError:
                continue

            all_peaks = all_peaks.append(peaks)
        
        timer.print_time()


    all_peaks = all_peaks.reset_index(drop=True)
    all_peaks['name'] = all_peaks['orf']  + '_' + all_peaks['time'].astype(str) + '_' + \
        all_peaks['chr'].astype(str) + '_' + all_peaks['original_mid'].astype(str)
    all_peaks = all_peaks.set_index('name')

    return all_peaks


def plot_small_peaks(gene, all_peaks, plotter):

    all_orfs = all_orfs_TSS_PAS()

    orf_name = get_orf_name(gene)
    orf = get_orf(orf_name, all_orfs)

    span = orf.TSS - 1000, orf.TSS + 1000
    plotter.set_span_chrom(span, orf.chr)
    plotter.dpi = 100
    fig, axs, tween_axs = plotter.plot()

    for i in range(len(times)):
        time = times[i]
        ax = axs[i]
        
        data = all_peaks[(all_peaks.cross_correlation > 0.05) & 
                         (all_peaks.orf == orf.name) & 
                         (all_peaks.time == time)]
        ax.scatter(data.original_mid, data.cross_correlation+10.0)


def plot_tf_scatter(small_peaks, tf_name=None, tf_names=None, t0=0.0, t1=120.0,
    no_annotations=False, labeled_peaks=None, dpi=300):
    
    apply_global_settings(dpi=dpi)

    linked_peaks = small_peaks.linked_peaks_normalized
    all_motifs = small_peaks.all_motifs

    plot_data = linked_peaks\
        .loc[small_peaks.prom_peaks['name']].copy()

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])

    x = plot_data[t0]
    y = plot_data[t1]

    def plot_line(ax, line):
        x = np.array([0, 1])
        m, b = line
        y = x*m + b
        ax.plot(x, y, c='gray', linestyle='dashed', linewidth=1)

    plot_line(ax, (1, 0))

    if tf_name is not None and tf_names is None:
        tf_names = [tf_name]

    if tf_names is None:
        ax.scatter(x, y, s=1, c='#b0b0b0')
 
        if not no_annotations:

            high_peaks, low_peaks = get_threshold_peaks(small_peaks, 
                plot_data, t0, t1)

            sc1 = ax.scatter(plot_data[plot_data.index.isin(high_peaks)][t0], 
                       plot_data[plot_data.index.isin(high_peaks)][t1], s=20, 
                       color=red(), marker='D', linewidth=1,
                       facecolor='none',)
                
            sc2 = ax.scatter(plot_data[plot_data.index.isin(low_peaks)][t0], 
                       plot_data[plot_data.index.isin(low_peaks)][t1], s=20, 
                       color=blue(), marker='o', linewidth=1,
                       facecolor='none')

            plt.legend([sc1, sc2], 
                ['Increased, N=%d' % len(high_peaks),
                 'Decreased, N=%d' % len(low_peaks)])

            plot_threshold_line(ax, 1)

    else:
        ax.scatter(x, y, s=1, c='#d0d0d0')

        i = 0
        markers = ['o', 'x']
        selected_sc = []
        labels = []
        colors = [parula()(0.5), parula()(0.0)]

        sizes = 0
        for tf_name in tf_names:

            color = colors[i]
            sel_peaks = linked_peaks.loc[all_motifs[all_motifs.tf == tf_name].peak]
            sc = ax.scatter(sel_peaks[t0], sel_peaks[t1],
                color=color, marker=markers[i])

            selected_sc.append(sc)
            labels.append("%s, N=%d" % (tf_name.title(), len(sel_peaks)))

            i += 1

        plt.legend(selected_sc, labels)

        plot_threshold_line(ax, small_peaks.fc_threshold)

    if labeled_peaks is not None:

        labeled_peaks = plot_data.join(labeled_peaks, how='inner')

        for idx, p in labeled_peaks.iterrows():
            ax.text(p.loc[t0], p.loc[t1]+0.005, p['name'],
                    ha='center', va='center', fontsize=13, fontdict={'style':'italic'})

    ax.set_xlim(0., 0.12)
    ax.set_ylim(0., 0.12)
    ax.set_xlabel('Peak occupancy, 0 min')
    ax.set_ylabel('Peak occupancy, %s min' % (str(t1)))

    if tf_names is not None:
        tf_names = [tf.title() for tf in tf_names]
        ax.set_title("%s change in promoter small fragment\npeaks, 0-%.0f min" % 
                     ("/".join(tf_names), t1), fontsize=20)
        
    else:
        ax.set_title("Change in promoter small fragment\npeaks, 0-%s min, N=%d" % 
                     (str(t1), len(x)), fontsize=20)
    
    return fig, ax


def regulon_for_peak(small_peaks, peak):
    pass

def peaks_for_tf(small_peaks, tf):
    pass


def get_summary_df(linked_peaks, all_motifs, func):
    
    min_n = 1

    tf_summary_df = pd.DataFrame()

    for tf in all_motifs.tf.unique():
        motif_tfs = all_motifs[all_motifs.tf == tf]

        if len(motif_tfs) >= min_n:
            values = linked_peaks.loc[motif_tfs.peak.values]
            
            values = log2_fold_change(values, pseudo_count=0.01)

            tf_summary_df = tf_summary_df.append(pd.Series(func(values), name=tf))
    tf_summary_df.columns = times
    return tf_summary_df


def plot_tf_summary(small_peaks, head=None, tail=None):

    summ_dif = small_peaks.tf_mean_means

    if head is not None:
        summ_dif = summ_dif.tail(head) # sorted descending
    elif tail is not None:
        summ_dif = summ_dif.head(tail) # sorted descending

    summ_dif = summ_dif.reset_index().rename(columns={'index': 'name'})

    x = summ_dif.index.values

    subset = head is not None or tail is not None

    if subset:
        apply_global_settings()

        if head is not None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3, 4))

        fig.tight_layout(rect=[0.15, 0.1, 0.99, 0.9])
        lw = 7

        ax.set_xticks(x)
        ax.set_xticklabels(summ_dif['name'].str.title(), rotation=90, 
            ha='center', va='top')
        ax.tick_params(axis='x', length=0, pad=4, labelsize=13.5)
        ax.set_yticks(np.arange(-0.6, 0.6, 0.1))

        if head is None:
            ax.set_ylabel("Log$_2$ fold-change\naverage occupancy", fontsize=14)
        else:
            ax.set_yticks([])

    else:
        apply_global_settings(linewidth=4, titlepad=80)
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.tight_layout(rect=[0.1, 0.1, 0.99, 0.75])
        lw = 7
        ax.set_title("Transcription factor binding\noccupancy dynamics, "
            "0-120 min", fontsize=50)
        ax.tick_params(axis='y', length=10, pad=5, labelsize=22)

        ax.set_ylabel("Log$_2$ fold-change\nin average occupancy", fontsize=30)

        ax.set_xticks([])
        ax.set_yticks(np.arange(-0.6, 0.6, 0.1))

    ax.set_xlim(-0.75, len(x)-1+0.75)

    plot_key = 'mean'

    if subset:
        for x in np.arange(0, len(summ_dif)):
            ax.plot([x, x], [-10, 0], lw=1.5, linestyle='solid', color='#f9f9f9')

    for idx, row in summ_dif.iterrows():
        ax.plot([idx, idx], [0, row.loc[plot_key]], c='#c0c0c0', lw=lw, 
            solid_capstyle='butt')
        
    # high
    filtered = summ_dif[summ_dif['name'].isin(small_peaks.selected_high_tfs.index)]
    for idx, row in filtered.iterrows():
        if subset:
            # ax.axvline(idx, lw=1.5, linestyle='solid', color=red(0.075))
            ax.plot([idx, idx], [-10, 0], lw=1.5, linestyle='solid', color=red(0.075))
        ax.plot([idx, idx], [0, row.loc[plot_key]], c=red(), lw=lw, solid_capstyle='butt')    

    for ticklabel in ax.get_xticklabels():
        if ticklabel.get_text() in filtered['name'].str.title().values:
            ticklabel.set_color(red())

    # low
    filtered = summ_dif[summ_dif['name'].isin(small_peaks.selected_low_tfs.index)]
    for idx, row in filtered.iterrows():
        if subset:
            # ax.axvline(idx, lw=1.5, linestyle='solid', color=blue(0.1))
            ax.plot([idx, idx], [-10, 0], lw=1.5, linestyle='solid', color=blue(0.1))
        ax.plot([idx, idx], [0, row.loc[plot_key]], c=blue(), lw=lw, solid_capstyle='butt')    

    for ticklabel in ax.get_xticklabels():
        if ticklabel.get_text() in filtered['name'].str.title().values:
            ticklabel.set_color(blue())

    if not subset:
        high_n, low_n = small_peaks.view_high, small_peaks.view_low

        plot_rect(ax, -0.5, -1, low_n, 2, color='#f0f0f0', zorder=0)
        plot_rect(ax, len(summ_dif)-high_n+.5, -1, 20, 2, color='#f0f0f0', zorder=0)

    ax.set_ylim(-0.25, 0.25)
    ax.axhline(0, linewidth=2, color='black')

    
def plot_gene_tfs_scatter(tf, small_peaks, datastore):

    time = 120.0

    # select orfs with the tf bound
    selected_peaks = small_peaks.all_motifs[small_peaks.all_motifs.tf == tf]
    selected_orfs = small_peaks.all_motifs[small_peaks.all_motifs.tf == tf].orf.values

    # load the transcription data
    xrate = datastore.transcript_rate_logfold[[time]]\
        .loc[selected_orfs]

    # load the linked peaks
    peaks = small_peaks.linked_peaks_normalized
    peaks = peaks.loc[selected_peaks.peak.values]
    peaks_diff = difference(peaks)

    plot_data = peaks_diff.join(selected_peaks.set_index('peak')[['orf']]).reset_index()
    plot_data = plot_data.groupby('orf').mean()
    plot_data = plot_data[[time]].join(xrate[[time]], rsuffix='_logfold_TPM', lsuffix='_sm_occ')
    plot_data = plot_data.sort_values('120.0_sm_occ', ascending=False)
    names = plot_data.join(small_peaks.all_orfs[['name']])

    plt.figure(figsize=(4, 4))
    plt.scatter(plot_data['120.0_sm_occ'], plot_data['120.0_logfold_TPM'])
    plt.plot([-1, 1], [0, 0], color='gray', linestyle='dotted')
    plt.plot([0, 0], [-100, 100], color='gray', linestyle='dotted')
    plt.xlim(-0.05, 0.05)
    plt.ylim(-10, 10)

    plt.set_xlabel()
    plt.set_xlabel()


def plot_gene_tfs_hm(tf, small_peaks, datastore):

    from config import times
    time = 120.0

    # select orfs with the tf bound
    selected_peaks = small_peaks.all_motifs[small_peaks.all_motifs.tf == tf]
    selected_orfs = small_peaks.all_motifs[small_peaks.all_motifs.tf == tf].orf.values

    # load the transcription data
    xrate = datastore.transcript_rate_logfold[times]\
        .loc[selected_orfs]

    # load the linked peaks
    peaks = small_peaks.linked_peaks_normalized
    peaks = peaks.loc[selected_peaks.peak.values]
    peaks_diff = difference(peaks)
    peaks_diff *= 100.0

    plot_data = peaks_diff.join(selected_peaks.set_index('peak')[['orf']])\
        .reset_index()
    plot_data = plot_data.groupby('orf').mean()

    plot_data = plot_data[times].join(xrate[times], rsuffix='_logfold_TPM', 
        lsuffix='_sm_occ').reset_index().groupby('index').mean()
    
    plot_data = plot_data.sort_values('120.0_sm_occ', ascending=False)
    names = plot_data.join(small_peaks.all_orfs[['name']])['name']
    names = [n for n in names]

    apply_global_settings()

    plt.figure(figsize=(8, 18))
    plt.imshow(plot_data, aspect=20./len(plot_data), vmin=-10, vmax=10, cmap='RdBu_r')
    plt.yticks(np.arange(len(plot_data)), names)
    plt.xticks([3, 9], ['Bin occupancy', 'Transcription'])


def plot_tf_heatmap(small_peaks, lim=5, is_high=True):

    apply_global_settings(titlepad=15)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
    
    datastore = small_peaks.datastore

    if is_high:
        selected_tfs = small_peaks.tf_mean_means.tail(small_peaks.view_high)
        highlighted = small_peaks.selected_high_tfs.index.values
    else:
        selected_tfs = small_peaks.tf_mean_means.head(small_peaks.view_low)
        highlighted = small_peaks.selected_low_tfs.index.values

    all_xrates = datastore.transcript_rate_logfold

    # select which orfs
    all_motifs = small_peaks.all_motifs
    selected = small_peaks.tf_set[(small_peaks.tf_set.index.isin(selected_tfs.index))]

    selected = selected[::-1]

    # collect the regulon for the TF
    regulon_xrate = small_peaks.mean_regulon_xrate\
        .loc[selected.index.values]

    # transcription rate of the TF
    xrate = all_xrates.loc[selected.orf_name]

    # average occupancy of peaks (scale to similar values to xrates)
    bins = small_peaks.tf_means_df.loc[selected.index]*small_peaks.bin_scale

    zeros = np.zeros((len(bins), 1))
    data = np.concatenate([xrate.values, zeros, bins.values, 
                           zeros, regulon_xrate.values], axis=1)

    ax.imshow(data, vmin=-small_peaks.im_scale, vmax=small_peaks.im_scale, 
        cmap='RdBu_r', origin='lower',
        extent=[0, data.shape[1], 0, data.shape[0]], aspect=15./data.shape[1])

    ax.set_xlim(-0.1, data.shape[1]+0.1)
    ax.set_ylim(-0.1, data.shape[0]+0.1)

    hide_spines(ax)

    tfs = [n.title() for n in selected.index]
    ax.set_yticks(np.arange(len(selected))+0.5)
    ax.set_yticklabels(tfs)

    ax.set_xticks([3, 10, 17])
    ax.set_xticklabels(['Transcription', 'Binding\noccupancy', 'Regulon\ntranscription'])

    ax.tick_params(axis='y', length=0, pad=2, labelsize=10)
    ax.tick_params(axis='x', length=0, pad=4, labelsize=11.5)

    title_prefix = "increased" if is_high else "decreased"

    ax.set_title("Transcription factors\nwith %s occupancy" % title_prefix, 
        fontsize=16)

    for x in [6, 13]:
        plot_rect(ax, x, 0, 1, len(data), color='white', fill=True,
            joinstyle='miter')

    for x in [0, 7, 14]:
        plot_rect(ax, x, 0, 6, len(data), edgecolor='black', lw=2, fill=False,
            joinstyle='miter')

    if is_high: color = red()
    else: color = blue()

    for ticklabel in ax.get_yticklabels():
        if ticklabel.get_text().upper() in highlighted:
            ticklabel.set_color(color)


def plot_tf_regulon_heatmap(tf, small_peaks, is_high=True):
    
    apply_global_settings(titlepad=15)
    
    regulon_xrate = small_peaks.regulon_xrate

    plot_data = regulon_xrate[regulon_xrate.tf == tf]\
        .set_index('orf_name')[times]
    plot_data = plot_data.drop_duplicates()
    plot_data = plot_data.loc[plot_data[times].mean(axis=1)\
        .sort_values(ascending=False).index]

    fig, ax = plt.subplots(1, 1, figsize=(4, 10))
    fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])

    ax.imshow(plot_data, vmin=-small_peaks.im_scale, vmax=small_peaks.im_scale,
        cmap='RdBu_r', aspect=1.)

    if len(plot_data) < 50:
        ax.set_yticks(np.arange(len(plot_data)))
        ax.set_yticklabels(paper_orfs[['name']].loc[plot_data.index]['name'])
    else:
        ax.set_yticks([])

    ax.set_title("%s regulon\ntranscription" % tf)

    ax.tick_params(axis='y', length=0, pad=4, labelsize=10)
    ax.set_xticks([])
    return fig


def get_threshold_peaks(small_peaks, plot_data, t0=0.0, t1=120.0):

    x = plot_data[t0]
    y = plot_data[t1]

    fc_threshold = small_peaks.fc_threshold
    pseudo_count = 0.01

    slope = 2**fc_threshold

    x[x < 0] = 0
    y[y < 0] = 0

    lfc = np.log2((y+pseudo_count)/(x+pseudo_count))

    high_peaks = lfc[lfc > fc_threshold].index
    low_peaks = lfc[lfc < -fc_threshold].index

    return high_peaks, low_peaks


def plot_colorbars(small_peaks, write_path=None):

    from src.chromatin_heatmaps import _make_fake_cbar
    from src import plot_utils

    apply_global_settings(linewidth=2)

    fig, axs = plt.subplots(2, 1, figsize=(8,2))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.subplots_adjust(left=0.5)
    fig.patch.set_alpha(0.0)

    ax1, ax2 = tuple(axs)

    titles = ['Log$_2$ fold-change\ntranscription rate',
              'Log$_2$ fold-change\nbinding occupancy']

    scale_cbars = [1, small_peaks.bin_scale]
    formating = ['%.0f', '%.2f']
    for i in range(len(axs)):
        ax = axs[i]
        title = titles[i]
        vlim = small_peaks.im_scale
        scale_cbar = 1./scale_cbars[i]
        _make_fake_cbar(ax, vlim, title, scale=scale_cbar, 
            str_format=formating[i])
        plot_utils.format_spines(ax, lw=1.2)


def plot_threshold_line(ax, k):
        eps = 0.01
        x = np.arange(0, 0.5, 0.01)
        y = (x+eps)*2**k - eps
        ax.plot(x, y, color='gray', lw=1, zorder=0, linestyle='dashed')

        k = -k
        y = (x+eps)*2**k - eps
        ax.plot(x, y, color='gray', lw=1, zorder=0, linestyle='dashed')


if __name__ == '__main__':
    collect_small_peaks()

