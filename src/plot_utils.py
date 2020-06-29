
import matplotlib.patches as patches
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
from src.utils import print_fl


def apply_global_settings(titlepad=20, linewidth=2, dpi=300):

    from matplotlib import rcParams
    rcParams['axes.titlepad'] = titlepad 

    # set font globally
    from matplotlib import rcParams
    rcParams['figure.dpi'] = dpi
    rcParams['font.family'] = 'Open Sans'
    rcParams['font.weight'] = 'regular'
    rcParams['figure.titleweight'] = 'regular'
    rcParams['axes.titleweight'] = 'regular'
    rcParams['axes.labelweight'] = 'regular'
    rcParams['axes.labelsize'] = 13
    rcParams['axes.linewidth'] = linewidth
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11    

    rcParams['ytick.major.width'] = linewidth * 3./4
    rcParams['xtick.major.width'] = linewidth * 3./4
    rcParams['ytick.major.size'] = linewidth*2.5
    rcParams['xtick.major.size'] = linewidth*2.5

    rcParams['ytick.minor.width'] = linewidth * 3./4
    rcParams['xtick.minor.width'] = linewidth * 3./4
    rcParams['ytick.minor.size'] = linewidth*1
    rcParams['xtick.minor.size'] = linewidth*1

    rcParams['axes.labelpad'] = 6


def hide_spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def format_spines(ax, lw=1.5):
    ax.spines['right'].set_linewidth(lw)
    ax.spines['top'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)

def format_ticks_font(ax, fontname='Open Sans', weight='regular',
        fontsize=10, which='both'):

    ticks = []
    if which == 'both':
        ticks = ax.get_xticklabels() + ax.get_yticklabels()
    elif which == 'y':
        ticks = ax.get_yticklabels()
    elif which == 'x':
        ticks = ax.get_xticklabels()

    for tick in ticks:
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize=fontsize)

def plot_rect(ax, x, y, width, height, color=None, facecolor=None, 
    edgecolor=None, ls='solid', fill_alpha=1., zorder=40, lw=0.0, inset=(0.0, 0.0)):
    """
    Plot a rectangle for ORF plotting
    """

    if edgecolor is None: edgecolor = color
    if facecolor is None: facecolor = color

    patch = ax.add_patch(
                    patches.Rectangle(
                        (x, y + inset[1]/2.0),   # (x,y)
                        width, height - inset[1], # size
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        lw=lw,
                        joinstyle='round',
                        ls=ls,
                        fill=True,
                        alpha=fill_alpha,
                        zorder=zorder
                    ))

def plot_hist_2d(x, y):

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[200,221])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='magma_r')
    plt.colorbar()
    plt.show()
    return heatmap, xedges, yedges


def plot_density_scatter(x, y, bw, cmap='magma_r', vmin=None, 
    vmax=None, ax=None, s=10, alpha=1., zorder=1):

    try:
        kde = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=bw)
        z = kde.pdf([x, y])
    except ValueError:
        z = [0] * len(x)

    if ax is None:
        fig, ax = plt.subplots(figsize(4,4))

    sorted_idx = np.argsort(z)
    x, y, z = x[sorted_idx], y[sorted_idx], z[sorted_idx]

    # ax.scatter(x, y, c='', edgecolor='#c0c0c0', s=(s-1), zorder=2)
    ax.scatter(x, y, c=z, edgecolor='', s=s, cmap=cmap, vmin=vmin, 
        vmax=vmax, alpha=alpha, zorder=zorder, rasterized=True)



def plot_density(data, ax=None, color='red', arange=None, 
    alpha=1., zorder=1, fill=False, bw=10, neg=False, 
    mult=1.0, y_offset=0, flip=False, lw=1, label=None):

    from sklearn.neighbors import KernelDensity
    def _kde_sklearn(x, x_grid, bandwidth):
        kde_skl = KernelDensity(bandwidth=bandwidth)
        kde_skl.fit(x[:, np.newaxis])
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        pdf = np.exp(log_pdf)
        return pdf

    if ax is None:
        fig, ax = plt.subplots()

    if arange is None:
        arange = min(data), max(data), 1

    x = np.arange(arange[0], arange[1], arange[2])
    y = _kde_sklearn(data, x, bw) * mult
    d = scipy.zeros(len(y))
    fill_mask = y >= d

    if fill:
        if not flip:
            ax.fill_between(x, y+y_offset, 0, color=color,
                     alpha=alpha, linewidth=1, zorder=zorder)
        else:
            ax.fill_betweenx(x, y+y_offset, 0, color=color,
                     alpha=alpha, linewidth=1, zorder=zorder)
    else:
        if not flip:
            ax.plot(x, y+y_offset, color=color,
                 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
                 solid_joinstyle='round')
        else:
            ax.plot(y+y_offset, x, color=color,
                 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
                 solid_joinstyle='round')

    return y


def plot_violin(data, ax=None, color='red', arange=None, 
    alpha=1., zorder=1, fill=False, bw=10, neg=False, mult=1.0, y_offset=0):

    from sklearn.neighbors import KernelDensity
    def _kde_sklearn(x, x_grid, bandwidth):
        kde_skl = KernelDensity(bandwidth=bandwidth)
        kde_skl.fit(x[:, np.newaxis])
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        pdf = np.exp(log_pdf)
        return pdf

    if ax is None:
        fig, ax = plt.subplots()

    if arange is None:
        arange = min(data), max(data), 1

    x = np.arange(arange[0], arange[1], arange[2])
    y = _kde_sklearn(data, x, bw) * mult
    d = scipy.zeros(len(y))
    fill_mask = y >= d
    
    y_lower = -y + y_offset
    ax.fill_between(x, y+y_offset, y_lower, interpolate=True, color=color,
         alpha=alpha, linewidth=0, zorder=zorder)
    

    return y


def make_colormap(seq, name):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    new_cmap = mcolors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(name=name, cmap=new_cmap)
    return new_cmap


def register_nucleosomal_small_cmap():
    """Divergent color map to be used with nucleosomal and small factors
    heatmaps"""

    nuc_color = '#0CAFAF'
    small_color = '#DD7300'

    # manually create divergent color scheme with forced white in the center
    c = mcolors.ColorConverter().to_rgb
    colors = [c(nuc_color), 0.01, c(nuc_color), c('white'), 
              0.5, c('white'), 0.5, 
              c('white'), c(small_color), 0.99, c(small_color)]

    make_colormap(colors, 'chromatin_frags')

    # reverse
    colors = [c(small_color), 0.01, c(small_color), c('white'), 
              0.5, c('white'), 0.5, 
              c('white'), c(nuc_color), 0.99, c(nuc_color)]

    make_colormap(colors, 'chromatin_frags_r')
    print_fl("Registered 'chromatin_frags' and 'chromatin_frags_r' colormaps")
