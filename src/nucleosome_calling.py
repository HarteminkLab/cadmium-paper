
from pandas import Series
import pandas as pd
import numpy as np
from src.timer import Timer
import matplotlib.pyplot as plt
from src.utils import print_fl


def call_orf_nucleosomes(cross_correlation, orf):

    all_nucs = pd.DataFrame()
    times = [0.0, 7.5, 15, 30, 60, 120]

    for time in times:
        idx = orf.name
        data = cross_correlation.loc[idx].loc[time]
        nucs_df = call_nucleosomes(data).copy()
        nucs_df['original_mid'] = nucs_df.mid 

        if orf.strand == '-': 
            nucs_df['original_mid'] = -nucs_df.mid

        nucs_df.loc[:, 'original_mid'] = nucs_df.original_mid + orf.TSS
        nucs_df['original_mid'] = nucs_df.original_mid.astype(int)
        nucs_df['mid'] = nucs_df.mid.astype(int)

        nucs_df['time'] = time
        nucs_df['orf'] = idx
        nucs_df['chr'] = orf.chr
        nucs_df = nucs_df.sort_values(['mid'])

        all_nucs = all_nucs.append(nucs_df)

    return all_nucs.reset_index(drop=True)

def call_nucleosomes(data):

    cutoff = 0.02
    window = 160
    window_2 = window/2

    cur_data = data.sort_values(ascending=False).copy()
    cur_data.index[cur_data.index > 50]

    nucleosomes_df = pd.DataFrame()
    last_nuc = float('inf')
    while len(cur_data) > 0 and last_nuc > cutoff:
        
        highest_idx = cur_data.index.values[0]
        highest = cur_data.loc[highest_idx]
        
        remove_span = highest_idx-window_2, highest_idx+window_2
        
        drop_idx = cur_data.index[(cur_data.index > remove_span[0]) & 
            (cur_data.index <= remove_span[1])]
        cur_data = cur_data.drop(drop_idx)
        
        nucleosomes_df = nucleosomes_df.append(Series({'cross_correlation':highest,
            'mid':highest_idx}), ignore_index=True)
        last_nuc = highest

    return nucleosomes_df



def plot_nuc_calls_cc():
    from src.plot_utils import apply_global_settings

    from config import cross_corr_sense_path
    cross = pd.read_hdf(cross_corr_sense_path, 'cross_correlation')
    time = 0

    cur_cross = cross.loc['nucleosomal'].query('time == %s' % str(time))
    cols = cur_cross.columns
    cur_cross = cur_cross.reset_index().set_index('orf_name')[cols]

    peak_1 = cur_cross.sum().idxmax()
    peak_2 = cur_cross[np.arange(peak_1+80, 500)].sum().idxmax()
    peak_3 = cur_cross[np.arange(peak_2+80, 500)].sum().idxmax()

    print_fl("Computed nucleosome spacing:", log=True)
    print_fl("+1, +2 distance: %0.0f" % (peak_2 - peak_1), log=True)
    print_fl("+2, +3 distance: %0.0f" % (peak_3 - peak_2), log=True)

    apply_global_settings()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])

    ax.plot(cur_cross.sum())

    import matplotlib.patheffects as path_effects
    for p in [peak_1, peak_2, peak_3]:
        ax.axvline(p, linestyle='solid', color='red', alpha=0.25, lw=3)
        text = ax.text(p, 500, "TSS+%d" % p, ha='center', fontsize=12)
        text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'),
                               path_effects.Normal()])
    x = np.arange(-200, 800, 100)
    ax.set_xticks(x)
    xlabels = [str(val) if val < 0 else '+%d' % val for val in x]
    xlabels[2] = 'TSS'
    ax.set_xticklabels(xlabels)
    ax.set_title("Gene body nucleosomes, 0 min", fontsize=24)
    ax.set_ylim(0, 600)
    ax.set_xlim(-200, 600)
    ax.set_xlabel('Position (bp)')
    ax.set_ylabel('Cumulative nucleosome\ncross correlation score across genes')

def plot_p123(gene_name, orf_cc, plotter, sum_plotter, save_dir):

    from src.nucleosome_linkages import plot_linkages_cc, get_linkages_cc, find_p123_gene
    from src.utils import get_orf 

    plot_linkages_lines = False
    orf = get_orf(gene_name)
    cur_cc, linkages, _ = get_linkages_cc(orf_cc, gene_name, plotter.orfs)
    p1, p2, p3 = find_p123_gene(orf, linkages)

    if plot_linkages_lines:
        plot_linkages = linkages[linkages.link.isin([p1, p2, p3])]
        plot_linkages_cc(cur_cc, plot_linkages)
    
    plotter.disable_mnase_seq = False

    min_cc_plotting = -1

    plot_linkages = linkages[linkages.link.isin([p1, p2, p3])]
    plot_linkages = plot_linkages[plot_linkages.cross_correlation > min_cc_plotting]

    # typhoon plot of linkages
    plotter.linkages = plot_linkages
    fig, time_ax, twen_axs = plotter.plot_gene(gene_name, 
        figwidth=12, padding=(500, 1000), highlight=False, dpi=100,
        save_dir=save_dir, prefix='typhoon_shift_')
    plt.close(fig)

    # get nucleosome positions by time and linkage
    # TODO: use this data structure for typhoon plot
    time_mids = pd.DataFrame()
    for p_nuc in [p1, p2, p3]:
        p_pos = linkages[linkages.link == p_nuc]
        cur_mids = linkages[linkages.link == p_nuc][['time', 'mid']]
        cur_mids['link'] = p_nuc
        time_mids = time_mids.append(cur_mids)

    # plot cross correlation
    sum_plotter.set_gene(gene_name)
    write_path = "%s/cc_%s.pdf" % (save_dir, gene_name)
    fig = sum_plotter.plot_cross_correlation_heatmap(show_colorbar=True,
                    title='%s cross correlation' % gene_name, nucs=time_mids)
    plt.savefig(write_path, transparent=False)
    plt.close(fig)


def is_row_valid(row, key1, key2):
    
    from config import times

    # constrain +1 and +2 to a distance
    # greater than 147
    repeat_min = 147
    
    for time in times:
        p1_pos = row["%.1f_%s" % (time, key1)]
        p2_pos = row["%.1f_%s" % (time, key2)]
        p1_p2_difference = p2_pos - p1_pos
        if p1_p2_difference < repeat_min:
            return False
            break

    return True

def validate_pair(pair_of_nucs, key1, key2):
    valid_rows = pair_of_nucs.apply(lambda row: is_row_valid(row, key1, key2), axis=1)
    return pair_of_nucs[valid_rows].index

def load_p123(strand_name):
    from config import mnase_dir
    from src.datasets import read_orfs_data
    from src.transformations import difference

    p1_positions = read_orfs_data('%s/p1_%s.csv' % (mnase_dir, strand_name))
    p2_positions = read_orfs_data('%s/p2_%s.csv' % (mnase_dir, strand_name))
    p3_positions = read_orfs_data('%s/p3_%s.csv' % (mnase_dir, strand_name))

    p12 = p1_positions.join(p2_positions, lsuffix='_+1', rsuffix='_+2')
    p23 = p2_positions.join(p3_positions, lsuffix='_+2', rsuffix='_+3')

    valid_12_orfs = validate_pair(p12, '+1', '+2')
    valid_23_orfs = validate_pair(p23, '+2', '+3')

    valid_orfs = list(set(valid_12_orfs).intersection(set(valid_23_orfs)))

    p1_shift = difference(p1_positions.loc[valid_orfs])
    p2_shift = difference(p2_positions.loc[valid_orfs])
    p3_shift = difference(p3_positions.loc[valid_orfs])

    p1 = p1_positions.loc[valid_orfs]
    p2 = p2_positions.loc[valid_orfs]
    p3 = p3_positions.loc[valid_orfs]

    return p1, p2, p3, p1_shift, p2_shift, p3_shift
