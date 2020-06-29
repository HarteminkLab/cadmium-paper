
from src.utils import get_gene_name, get_orf_name
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def groups():
    return ['Met4-Complex', 'SCF-Met30', 'Sulfur Sparing Isoforms', 
        'Sulfur Assimilation', 'Methyl Cycle', 'Transsulfuration', 'Glutathione Biosynthesis']

def group_colors():
    return {'Met4-Complex': '#A3C6EA', 
            'SCF-Met30': '#979796',
            'Sulfur Sparing Isoforms': '#B967A9',
            'Sulfur Assimilation': '#F9A775',
            'Methyl Cycle': '#F9E762', 
            'Transsulfuration': '#C7DB6D', 
            'Glutathione Biosynthesis': '#8AC64F'}

def orf_groups():
    return {'Glutathione Biosynthesis': ['YJL101C', 'YOL049W'],
            'Met4-Complex': ['YNL103W', 'YJR060W','YIR017C', 'YPL038W', 'YDR253C'],
            'Methyl Cycle': ['YER091C', 'YLR180W', 'YDR502C', 'YER043C'],
            'SCF-Met30': ['YDR054C', 'YDL132W', 'YDR328C', 'YIL046W'],
            'Sulfur Assimilation': [
            'YBR294W', 'YLR092W', 'YJR010W', 'YKL001C', 'YPR167C', 'YJR137C', 'YFR030W'],
            'Sulfur Sparing Isoforms': ['YGR087C',
            'YLR044C', 'YGR254W', 'YHR174W', 'YOR374W', 'YPL061W'],
            'Transsulfuration': ['YAL012W', 'YGR155W']}

def all_genes():
    genes = []
    for gs in orf_groups().values():
        for orf in gs:
            g = get_gene_name(orf)
            genes.append(g)
    return genes


def plot_timecourse(data_store):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.tight_layout(rect=[0.05, 0.3, 0.95, 0.85])

    met30_orf = get_orf_name('MET30')
    met32_orf = get_orf_name('MET32')
    sulf_assim = orf_groups()['Sulfur Assimilation']

    disorg = data_store.gene_body_disorganization_delta
    times = [0, 7.5, 15, 30, 60, 120]

    x = np.arange(6)

    ax.plot(x, disorg.loc[met30_orf], color='#7A7A7A', mec='#7A7A7A', mfc='white',
            marker='o', fillstyle='full', markersize=7,
          label='MET30')
    ax.plot(x, disorg.loc[met32_orf], mec='#4F8DC6', mfc='white',
            color='#4F8DC6', marker='v', fillstyle='full', markersize=9,
          label='MET32')

    # for orf in sulf_assim:
    sa_disorg = disorg.loc[sulf_assim].mean()
    sa_disorg_lower = disorg.loc[sulf_assim].min()
    sa_disorg_upper = disorg.loc[sulf_assim].max()
    ax.plot(x, sa_disorg, mfc='#FAF1EC', color='#EF7030', mec='#EF7030',
            zorder=1, marker='D', fillstyle='full', markersize=7, 
            label='Sulfur assimilation')
    ax.fill_between(x, sa_disorg_lower, sa_disorg_upper, color='#FAF1EC', zorder=1, alpha=1.)

    dis_ticks = np.arange(-8, 9, 2)
    ax.set_yticks(dis_ticks)
    labels = ['%d' % s for s in dis_ticks]
    labels[4] = 0
    labels = labels[0:3] + labels[3:4] + ['%s' % l for l in labels[4:]]
    ax.set_yticklabels(labels)
    ax.set_ylabel('$\\Delta$ Nucleosome disorganization')

    ax.set_xticks(x)
    ax.set_xticklabels(['%s\'' % str(t) for t in times])

    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.5, 5)

    ax.set_title('Sulfur pathway\ntime course', fontdict={'family':'Open Sans'}, fontsize=18)
    ax.legend(bbox_to_anchor=(0.45, -0.15),
              frameon=False, fontsize=12)

