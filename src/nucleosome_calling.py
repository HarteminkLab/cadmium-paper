
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
    window = 150
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
    ax.set_xlabel('Position, bp')
    ax.set_ylabel('Cumulative nucleosome\ncross-correlation score across genes')
