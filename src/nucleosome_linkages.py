
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.nucleosome_calling import call_nucleosomes
from src.utils import get_orf
from src.nucleosome_calling import call_orf_nucleosomes
import matplotlib.pyplot as plt
import os
from src.datasets import read_orfs_data
from src.utils import get_orf, print_fl
from src.timer import Timer
from config import *
from src.tasks import TaskDriver, child_done
from src.slurm import submit_sbatch


def get_p_positions(p123, linkages, which_p):
    """
    Get the positions of +1, +2, +3 nucleosomes per time
    """

    # enforce cutoff
    min_cross_corr = 0.05
    
    linkages = linkages[linkages.cross_correlation > min_cross_corr].copy()
    linkages_pvt = linkages.pivot(index='link', columns='time', values='mid')

    times = linkages_pvt.columns
    p1_positions = p123[[]].copy()
    for t in times: p1_positions[t] = np.nan

    for orf_name, row in p123[[which_p]].iterrows():
        p1_link = row[which_p]
        if type(p1_link) != str: continue
        p1_positions.loc[orf_name] = linkages_pvt.loc[p1_link].values
    p1_positions = p1_positions.dropna().astype(int)

    return p1_positions


def find_p123_gene(gene, nucleosome_linkages):

    origin = 0 # TSS
    nuc_padding = 80
    p1_link, p2_link, p3_link = None, None, None

    p1_link, min_p1_pos = find_p_nucleosome_link(gene, nucleosome_linkages, origin=origin, 
        search_window=240)

    if p1_link is not None:
        p2_link, min_p2_pos = find_p_nucleosome_link(gene, nucleosome_linkages, 
            min_mid=min_p1_pos + nuc_padding,
            exclude=[p1_link])

    if p2_link is not None:
        p3_link, min_p3_pos = find_p_nucleosome_link(gene, nucleosome_linkages, 
            min_mid=min_p2_pos + nuc_padding, 
            exclude=[p1_link, p2_link])

    return p1_link, p2_link, p3_link


def find_p_nucleosome_link(gene, nucleosome_linkages, min_mid=None, 
    search_window=0, origin=0,
    exclude=[]):

    search_window_2 = search_window/2.
    min_cc = 0.05
    max_mid = origin + 700 + search_window_2

    if min_mid is None:
        min_mid = origin - search_window_2

    mask = ((nucleosome_linkages.generated == 0.0) &
            (nucleosome_linkages.cross_correlation > min_cc) &
            (nucleosome_linkages.mid > min_mid) & 
            (nucleosome_linkages.mid < max_mid) &
            (nucleosome_linkages.chr == gene.chr) &
            (nucleosome_linkages.orf == gene.name) & 
            (~nucleosome_linkages.link.isin(exclude)))

    near_nucs = nucleosome_linkages[mask].copy()
    near_nucs['dist'] = near_nucs.mid - origin
    near_nucs['abs_dist'] = np.abs(near_nucs.mid - origin)
    near_nucs = near_nucs.sort_values('abs_dist')

    if len(near_nucs) == 0:
        return None, None

    nearest_link = near_nucs.reset_index().loc[0]
    nearest_link_min_pos = nucleosome_linkages[nucleosome_linkages.link == 
        nearest_link.link].mid.min()

    return nearest_link.link, nearest_link_min_pos


def find_linkages_peak(cur_peak, link_idx, working_peaks, 
    times=[0.0, 7.5, 15, 30, 60, 120]):
    ""
    linkages = pd.DataFrame()

    cur_peak['link'] = link_idx
    cur_peak['generated'] = False
    idx_time = np.where(np.array(times) == cur_peak.time)[0][0]
    linkages = linkages.append(cur_peak)

    # find peaks backwards
    linkages, del_1 = find_linkages_fwd_bck(cur_peak, linkages, 
        np.array(times)[np.arange(idx_time-1, -1, -1)], working_peaks, link_idx)

    # find peaks forward
    linkages, del_2 = find_linkages_fwd_bck(cur_peak, linkages, 
        np.array(times)[np.arange(idx_time+1, len(times))], working_peaks, link_idx)

    return linkages.sort_values(['link', 'time']), ([cur_peak.name] + del_1 + del_2)

def create_peak(peak, time):
    """Create a peak by projecting the boundary of a 
    peak at another time onto a time"""
    new_peak = peak.copy()

    new_peak.time = time
    new_peak['generated'] = True
    new_peak['cross_correlation'] = 0

    return new_peak

def find_linkages_fwd_bck(last_peak, linkages, times, 
    working_peaks, link_idx):
    if len(times) == 0: return linkages, []

    del_indices = []

    for time in times:
        cur_peak = find_nearest_index(last_peak, working_peaks, time)

        # didnt find a peak, create a new one
        if cur_peak is None:
            last_peak = create_peak(last_peak, time)
        # found a peak, add to deletion index
        else: 
            last_peak = cur_peak.copy()
            last_peak['generated'] = False
            del_indices.append(last_peak.name)

        last_peak['link'] = link_idx
        linkages = linkages.append(last_peak)

    return linkages, del_indices

def find_nearest_index(cur_peak, peaks, time, search_padding=100):
    """Find nearest peak from another time. return the index of the peak"""
    
    cur_peaks = peaks[(cur_peak.mid < peaks.mid + search_padding) & 
                      (cur_peak.mid >= peaks.mid - search_padding) & 
                      (peaks.time == time)].copy()

    if len(cur_peaks) == 0: return None
    cur_peaks['dist'] = np.abs(cur_peaks.mid - cur_peak.mid)
    cur_peaks = cur_peaks.sort_values('dist')

    return peaks.loc[cur_peaks[0:1].index[0]]


def find_linkages(peaks):
    """find peak with the largest signal across all times

    collect to peaks across earlier and later times

    if a peak does not exist earlier or later,  create one by casting 
    it from the previous high signal
    and finding a peak in that window"""

    working_peaks = peaks.sort_values('cross_correlation', ascending=False)

    # combine peaks across times
    linkages = pd.DataFrame()
    all_linkages = pd.DataFrame()

    # iterate through peaks with the highest signal
    while len(working_peaks) > 0:

        # get the highest peak available
        cur_peak = working_peaks.loc[working_peaks[0:1].index[0]].copy()
        link_idx = (cur_peak.orf + '_' + 
                    cur_peak.chr.astype(int).astype(str) + "_" + 
                    cur_peak.original_mid.astype(int).astype(str) + '_' + 
                    cur_peak.time.astype(str))

        # find linkages forward and backwards, create if necessary
        linkages, del_indices = find_linkages_peak(cur_peak, link_idx, 
            working_peaks)

        # drop peaks in already found linkages
        working_peaks = working_peaks.drop(del_indices)

        all_linkages = all_linkages.append(linkages)

    all_linkages.mid = all_linkages.mid.astype(int)
    all_linkages.original_mid = all_linkages.original_mid.astype(int)

    return all_linkages


def get_linkages_cc(orf_cc, gene_name, orfs):
    orf = get_orf(gene_name, orfs)

    cur_cc = orf_cc.loc['diff'].loc[orf.name]
    
    nucs = call_orf_nucleosomes(orf_cc.loc['diff'], orf)
    linkages = find_linkages(nucs)

    return cur_cc, linkages, nucs

def plot_linkages_cc(cur_cc, linkages):

    times = [0.0, 7.5, 15, 30, 60, 120]
    fig, axs = plt.subplots(11, 1, figsize=(10, 6))
    axs = np.array(axs)

    time_axs = axs[np.arange(6)*2]
    tween_axs = axs[np.arange(5)*2+1]

    fig.tight_layout(rect=[0.075, 0.03, 0.95, 0.945])
    plt.subplots_adjust(hspace=0.0, wspace=0.5)

    plot_linkages = linkages#[(linkages.generated == 0) & 
                            #(linkages.cross_correlation > 0.02)]

    span = cur_cc.columns.min(), cur_cc.columns.max()

    for i in range(len(times)):
        time = times[i]
        time_cc = cur_cc.loc[time]

        ax = time_axs[i]
        ax.plot(time_cc)
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(*span)
        ax.axhline(0)
        cur_linkages = plot_linkages[(plot_linkages.time == time)]
        links = cur_linkages.link.unique()
        ax.axvline(0, color='black')

        for idx, link in cur_linkages.iterrows():
            ax.axvline(link.mid)

        if i > 0:
            prev_time = times[i-1]
            prev_nucs = plot_linkages[plot_linkages.time == prev_time]
            tween_ax = tween_axs[i-1]
            tween_ax.set_xlim(*span)
            tween_ax.set_ylim(0, 10)

            for link in links:
                prevs = prev_nucs[prev_nucs.link == link]
                curs = cur_linkages[cur_linkages.link == link]

                if len(prevs) > 0 and len(curs) > 0:
                    prev = prevs.reset_index().loc[0]
                    nuc = curs.reset_index().loc[0]
                    tween_ax.plot([prev.mid, nuc.mid], [10, 0])

def task_name(antisense):
    return "call_nucleosomes_%s" % ('antisense' if antisense else 'sense')

def call_all_nucleosome_p123(orfs, antisense,
    cross_correlation_dir, chrom_save_dir, timer):
  
    linkages = pd.DataFrame()

    name = task_name(antisense)
    driver = TaskDriver(name, WATCH_TMP_DIR, 16, timer=timer)
    driver.print_driver()

    for chrom in range(1, 17):

        if not USE_SLURM:
            call_nucleosomes_p123_chrom(orfs, chrom, antisense, 
                cross_correlation_dir, chrom_save_dir, timer)
            child_done(name, WATCH_TMP_DIR, chrom)
        else:
            exports = ("CHROM=%d,ANTISENSE=%s,SLURM_WORKING_DIR=%s,CONDA_PATH=%s,CONDA_ENV=%s"
                       % (chrom, str(antisense), SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV))
            script = 'scripts/2_preprocessing/call_nucleosomes.sh'
            submit_sbatch(exports, script, WATCH_TMP_DIR)

    # wait for all chromosomes to finish
    # superfluous if not in SLURM mode
    driver.wait_for_tasks()

    print_fl()

    # merge
    nucleosomes = pd.DataFrame()
    p123 = pd.DataFrame()
    for chrom in range(1, 17):

        if not os.path.exists(nucleosomes_filename(chrom_save_dir, chrom)): 
            continue
        nuc_chr = pd.read_csv(nucleosomes_filename(chrom_save_dir, chrom))\
            .set_index('orf')
        p123_chr = pd.read_csv(p123_filename(chrom_save_dir, chrom))\
            .set_index('orf_name')

        nucleosomes = nucleosomes.append(nuc_chr)
        p123 = p123.append(p123_chr)

    return nucleosomes, p123

def convert_to_pos_time_df(p123, nucleosomes, key):

    times = [0.0, 7.5, 15, 30, 60, 120]

    orf_p_nuc = p123.reset_index()[['orf_name', key]]

    p_positions = nucleosomes.pivot(index='link', columns='time', values='mid')
    res_df = orf_p_nuc.reset_index().merge(p_positions, left_on=key, right_on='link')\
        .set_index('orf_name')

    res_df = res_df[times]

    return res_df


def nucleosomes_filename(save_chrom_dir, chrom):
    return "%s/called_nucleosomes_chr%d.csv" % (save_chrom_dir, chrom)

def p123_filename(save_chrom_dir, chrom):
    return "%s/p123_chr%d.csv" % (save_chrom_dir, chrom)

def call_nucleosomes_p123_chrom(orfs, chrom, antisense,
    cross_correlation_dir, save_chrom_dir, timer):

    chrom_orfs = orfs[orfs.chr == chrom]

    if len(chrom_orfs) == 0: return None

    print_fl("Chromosome %d. %d genes" % (chrom, len(chrom_orfs)))
    timer.print_time()

    p123_orfs = chrom_orfs[[]].copy()
    p123_orfs['+1'] = np.nan
    p123_orfs['+2'] = np.nan
    p123_orfs['+3'] = np.nan

    linkages = pd.DataFrame()

    # load relevant cross correlations
    chrom_cross_correlation = pd.read_hdf(
        '%s/cross_correlation_chr%d.h5.z' % 
        (cross_correlation_dir, chrom))

    i = 0
    for idx, orf in chrom_orfs.iterrows():

        if i % 200 == 0: 
            print_fl("  %d/%d - %s" % (i, len(chrom_orfs), timer.get_time()))

        i += 1
        # call nucleosomes and link them in ORF window

        try:
            nucs = call_orf_nucleosomes(chrom_cross_correlation.loc['diff'],
                orf)
            cur_linkages = find_linkages(nucs)
            linkages = linkages.append(cur_linkages)
        except KeyError:
            continue

        # +1, +2, +3    
        p1, p2, p3 = find_p123_gene(orf, linkages)
        p123_orfs.loc[idx] = [p1, p2, p3]

    save_path = nucleosomes_filename(save_chrom_dir, chrom)
    linkages.to_csv(save_path)

    p123_orfs = p123_orfs.dropna()
    save_path = p123_filename(save_chrom_dir, chrom)
    p123_orfs.to_csv(save_path)


def main():

    (_, chrom, antisense) = \
        tuple(sys.argv)
    antisense = antisense.lower() == 'true'

    chrom = int(chrom)
    print_fl("Running nucleosome calling on chromosome %d, antisense: %s" % 
        (chrom, str(antisense)))

    name = task_name(antisense)
    timer = Timer()

    p123_orfs = paper_orfs
    save_chrom_dir = sense_nuc_chrom_dir
    cc_dir = cc_sense_chrom_dir

    if antisense:
        p123_orfs = antisense_orfs
        save_chrom_dir = anti_nuc_chrom_dir
        cc_dir = cc_antisense_chrom_dir

    call_nucleosomes_p123_chrom(p123_orfs, chrom, antisense, 
        cc_dir, save_chrom_dir, timer)

    child_done(name, WATCH_TMP_DIR, chrom)

if __name__ == '__main__':
    main()

