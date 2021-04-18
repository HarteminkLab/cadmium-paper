
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import collections as mc
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from src.plot_utils import plot_rect

class ORFAnnotationPlotter:

    def __init__(self, orfs=None, introns=None):
        self.orfs = orfs
        self.introns = introns
        self.show_spines = True
        self.show_minor_ticks = False
        self.plot_tick_labels = True
        self.span = None
        self.chrom = None
        self.inset = (0, 10.0)
        self.triangle_width = 80
        self.height = 50
        self.text_vertical_offset = 22
        self.text_horizontal_offset = 10
        self.plot_orf_names = False

    def set_span_chrom(self, span, chrom):
        if not self.span == span or not chrom == self.chrom:
            self.chrom = chrom
            self.span = int(span[0]), int(span[1])

        span_width = self.span[1] - self.span[0]
        self.span_width = span_width

        # triangle is proportion of the span width
        self.triangle_width = span_width/50.*1.25

        # overlap triangle with rect
        self.epsilon = span_width * 1e-4

        self.text_horizontal_offset = span_width * 1e-2

    def get_color(self, watson, gene_type):
        colors = {
            "Verified": cm.Blues(0.7),
            "Putative": cm.Blues(0.35),
            "Dubious": "#898888"
        }
        if not watson:
            colors['Verified'] = cm.Reds(0.5)
            colors['Putative'] = cm.Reds(0.3)

        colors['Uncharacterized'] = colors['Putative']
        color = colors[gene_type]
        return color

    def plot_orf_annotation(self, ax, start, end, name, CDS_introns=None, 
        watson=True, offset=False, gene_type="Verified", TSS=None, PAS=None,
        plot_orf_names=True):

        color = self.get_color(watson, gene_type)
        rect_width = end - start - self.triangle_width

        if watson:
            rect_start = start
            triangle_start = end - self.triangle_width - self.epsilon*2 # add one to overlap triangle with rect
            y_baseline = offset * self.height
            text_start = start + self.text_horizontal_offset
        else:
            triangle_start = start + self.epsilon # add one to overlap triangle with rect
            rect_start = start + self.triangle_width
            y_baseline = (offset+1) * -self.height
            text_start = end - self.text_horizontal_offset

        # plot pointed rectangle for ORF

        plot_rect(ax, rect_start, y_baseline, rect_width, self.height, color, 
            inset=self.inset)

        plot_iso_triangle(ax, triangle_start, y_baseline, 
            self.triangle_width, self.height, color, facing_right=watson,
            inset=self.inset[1])

        # plot introns as rectangles overtop of ORF
        if CDS_introns is not None and len(CDS_introns) > 0:
            for idx, intron in CDS_introns.iterrows():
                if intron['cat'] == 'intron':
                    intron_width = intron.stop - intron.start
                    # plot introns as a lighter version of the CDS
                    plot_rect(ax, intron.start, y_baseline, intron_width, 
                        self.height, 'white', inset=self.inset)
                    plot_rect(ax, intron.start, y_baseline, intron_width, 
                        self.height, color, fill_alpha=0.25, inset=self.inset)

        # plot name of ORF
        if plot_orf_names:
            plot_text(ax,
                text_start, 
                y_baseline+self.text_vertical_offset, 
                name, color, flipped=(not watson))

        # Plot TSS PAS line indicators
        plot_TSS_PAS(ax, start, end, TSS, PAS, 
            y_baseline, self.height, color, flipped=watson, inset=self.inset[1])

    def plot_orf_annotations(self, ax, 
        orf_classes=['Verified', 'Uncharacterized', 'Dubious'],
        custom_orfs=None, should_auto_offset=True):

        if custom_orfs is None:
            orfs = self.orfs
        else: orfs = custom_orfs

        span = self.span
        chrom = self.chrom
        genes = orfs[(orfs['chr'] == int(chrom)) & 
                          (orfs['stop'] > span[0]) & 
                          (orfs['start'] < span[1]) & 
                          (orfs.orf_class.isin(orf_classes))]

        try:
            genes = genes.sort_values(['strand', 'start']).reset_index()

        # older pandas version
        except AttributeError:
            genes = genes.sort(['strand', 'start']).reset_index()

        ax.set_ylim(-100, 100)

        tick_intervals = 500, 100

        ax.set_xticks(range(span[0], span[1], tick_intervals[1]), minor=True)

        if not self.show_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        xticks = range(span[0], span[1]+tick_intervals[0], tick_intervals[0])
        ax.set_xticks(xticks, minor=False)

        # if not self.plot_tick_labels:
        ax.set_xticklabels(['' for _ in xticks])

        ax.set_xlim(span[0], span[1])
        ax.tick_params(axis='both', labelsize=12)
        ax.yaxis.set_visible(False)

        for idx, gene in genes.iterrows():

            name = gene['orf_name']

            if self.plot_orf_names:
                if not name == gene['name']:
                    name = "{}/{}".format(gene['name'], gene['orf_name'])
            else:
                name = gene['name']

            if not self.introns is None:
                gene_introns = self.introns[
                    self.introns['parent'] == gene['orf_name']]
            else: gene_introns = None

            offset = False
            if should_auto_offset: offset = (idx % 2) == 0

            self.plot_orf_annotation(ax, gene.start,
                      gene.stop, name,
                      CDS_introns=gene_introns,
                      watson=(gene.strand is "+"),
                      gene_type=gene.orf_class,
                      offset=offset,
                      TSS=gene['TSS'],
                      PAS=gene['PAS'])
        return ax

def plot_TSS_PAS(ax, start, end, TSS, PAS, y, height, color, flipped=False, 
    inset=0):
    """# Plot TSS PAS lines, if TSS or PAS does not exist use ORF start
        # and end boundaries"""

    y_bottom = y+inset/2.
    y_top = y+height-inset/2

    if TSS is not None:
        ax.plot([TSS, TSS], [y_bottom, y_top], color=color, lw=3, zorder=111)

    if PAS is not None:
        ax.plot([PAS, PAS], [y_bottom, y_top], color=color, lw=3, zorder=111)

    if not flipped:
        if np.isnan(TSS): TSS = start
        if np.isnan(PAS): PAS = end
    else:
        if np.isnan(TSS): TSS = end
        if np.isnan(PAS): PAS = start

    TSS_span = TSS, PAS    
    ax.plot(TSS_span, [y+height/2., y+height/2.], lw=3, color=color, zorder=60)


def plot_iso_triangle(ax, x, y, width, height, color, facing_right=True,
    inset=0):
    """
    Plot left or right pointing isosceles triangle for end cap of ORF
    """

    y_bottom = y+inset/2.
    y_top = y+height-inset/2.
    x_flat = x

    if facing_right: 
        x_pointed = x+width
    else: 
        x_pointed = x
        x_flat += width

    X = np.array([[x_flat,y_bottom], [x_flat,y_top], 
        [x_pointed, (y_bottom+y_top)/2.0]])

    triangle = plt.Polygon(X, color=color, linewidth=0, zorder=2)
    ax.add_patch(triangle)

def plot_text(ax, start, y, name, color, flipped=False):
    ha = 'left'
    if flipped: ha = 'right'

    text = ax.text(start, y, name, fontsize=22, clip_on=True, zorder=65, 
                   rotation=0, va='center', ha=ha, 
                   fontdict={'fontname': 'Open Sans', 'fontweight': 'regular',
                   'style': 'italic'},
                   color='white')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=color),
                           path_effects.Normal()])
