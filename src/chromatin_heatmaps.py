
from src import plot_utils
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
from src.utils import print_fl

class ChromatinHeatmaps:

    def __init__(self, data_store):

        self.data_store = data_store
        self.color_bar_scales = [4, 4, 15, 15]
        self.orfs = data_store.orfs
        self.show_xlabels = True
        self.show_saved_plot = False

        self.num_times = 6
        self.col_spacing = 0.5
        self.row_spacing = 2
        self.plot_gene_names = False

        if data_store.is_antisense:
            self.num_columns = 4
        else:
            self.num_columns = 3

        plot_utils.apply_global_settings(linewidth=3)

    def plot_colorbars(self, write_path=None):

        fig, axs = plt.subplots(2, 1, figsize=(8,2))
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        fig.subplots_adjust(left=0.5)
        fig.patch.set_alpha(0.0)
        # fig.patch.set_facecolor('red')

        ax1, ax2 = tuple(axs)

        titles = ['$\\Delta$ Promoter occupancy /\n$\\Delta$ Disorganization',
                  'Log$_2$ fold-change\ntranscription rate']
        scale_cbars = [1, 1, 1]
        for i in range(len(axs)):
            ax = axs[i]
            title = titles[i]
            vlim = self.color_bar_scales[i*2]
            scale_cbar = scale_cbars[i*2]
            _make_fake_cbar(ax, vlim, title, scale=scale_cbar)
            plot_utils.format_spines(ax, lw=1.2)

        if write_path is not None:
            plt.savefig(write_path, transparent=False)

    def plot_heatmap(self, orf_groups=None, group_names=None, 
        group_colors=None, orf_names=None, head=None, tail=None, 
        ax=None, write_path=None, aspect_scale=10., fig_height=10.,
        lines=[], highlight_max=([], [], []), y_padding=None, group_spacing=0):

        plot_utils.apply_global_settings()

        self.aspect_scale = aspect_scale
        
        num_columns = self.num_columns
        num_times = self.num_times
        col_spacing = self.col_spacing
        row_spacing = self.row_spacing

        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
            fig.tight_layout(rect=[0.05, 0.03, 0.91, 0.95])
            fig.patch.set_alpha(0.0)
        else: fig = None

        if head is None and tail is None:
            plot_data = self.data_store.data
        elif head is not None:
            plot_data = self.data_store.data.head(head)
        elif tail is not None:
            plot_data = self.data_store.data.tail(tail)

        if orf_names is not None:
            plot_data = plot_data.loc[orf_names]

        self.aspect = aspect_scale/len(plot_data)

        if orf_groups is not None:
            num_groups = len(group_names)
            group_names = list(reversed(group_names))
            cur_row_height = 0
            tick_positions = []
            tick_labels = []

            border_padding = 0.2
            for r in range(0, num_groups):
                group_name = group_names[r]
                group_orfs = orf_groups[group_name]
                group_data = plot_data.loc[group_orfs]
                cur_height = len(group_data)
                y_start = cur_row_height + group_spacing

                if group_colors is not None:
                    plot_utils.plot_rect(ax, -16,
                        y_start-border_padding, num_times*num_columns+21,
                        cur_height+border_padding*2, color=group_colors[group_name], 
                        zorder=2, lw=12.0)

                cur_row_height = self.plot_group(ax, group_data, cur_height, 
                    y_start, [])

                if cur_height < 100 and self.plot_gene_names:
                    names = group_data.join(self.orfs[['name']])['name'].values
                    names = list(reversed(names))

                    tick_labels = tick_labels + list(names)
                    tick_positions = tick_positions + \
                        list(np.arange(cur_height) + y_start + 0.5)
                else:
                    tick_labels = []
                    tick_positions = []

                for i in range(len(tick_labels)):
                    ax.text((num_times*num_columns + 
                            (num_columns-1)*col_spacing+0.5),
                            tick_positions[i], tick_labels[i],
                            ha='left', va='center', fontsize=14,
                            fontdict={'fontname': 'Open Sans'})

                # group names
                ax.text(-1,
                    y_start+cur_height/2.0, group_name, 
                    ha='right', va='center', fontsize=14,
                    fontdict={'fontname': 'Open Sans'})

                # draw gene names manually when grouped
                tick_labels = []
                tick_positions = []
        else:

            cur_row_height = len(plot_data)
            height = len(plot_data)
            cur_row_height = 0
            tick_positions = []
            tick_labels = []

            y_start = cur_row_height

            cur_row_height = self.plot_group(ax, plot_data, height, 
                0, lines)

            if height < 100 and self.plot_gene_names:
                names = plot_data.join(self.orfs[['name']])['name'].values
                names = list(reversed(names))
                tick_labels = tick_labels + list(names)
                tick_positions = tick_positions + \
                    list(np.arange(height))

        x_padding = 0.25

        if y_padding is None:
            y_padding = len(plot_data)*1e-2

        ax.set_yticks(np.array(tick_positions)+0.5)
        ax.set_yticklabels(tick_labels, fontsize=26)

        if group_names is not None:

            xlim = [-col_spacing-16, 
                   (num_times*num_columns + (num_columns-1)*col_spacing)]

            xlim[0] = xlim[0]-1-x_padding
            xlim[1] = xlim[1]+5+x_padding

            ax.set_xlim(*xlim)
            ax.set_ylim(-y_padding, cur_row_height+y_padding)

            ax.tick_params(axis='x', length=0, pad=0)

        else:

            ax.set_xlim(-x_padding, (num_times*num_columns + 
                (num_columns-1)*col_spacing)+x_padding)
            ax.set_ylim(-y_padding, cur_row_height+y_padding)

            ax.tick_params(axis='x', labelsize=10, length=0, pad=15)
            ax.tick_params(axis='y', labelsize=18, length=0, pad=0)

        ax.yaxis.tick_right()

        if self.show_xlabels:
            positions = num_times/2 + \
                np.arange(self.num_columns)*(num_times + col_spacing)
            ax.set_xticks(positions)

            if group_names is None:
                ax.set_xticklabels(self.data_store.xlabels, fontsize=26)
            else:
                ax.set_xticklabels(self.data_store.xlabels, fontsize=12)

        else:
            ax.set_xticks([])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if write_path is not None:
            plt.savefig(write_path, transparent=True)
            print_fl("Writing %s" % write_path)

            # close plots
            if not self.show_saved_plot:
                if fig is not None:
                    plt.close(fig)
                plt.cla()
                plt.clf()

    def plot_group(self, ax, group_data, cur_height, cur_row_height, lines):

        num_times = self.num_times
        col_spacing = self.col_spacing
        row_spacing = self.row_spacing
        num_columns = self.num_columns
        height = len(group_data)

        for c in range(0, num_columns):
            i = c*num_times
            i_end = i+num_times
            x_start = c*num_times + c*col_spacing
            x_end = x_start+num_times
            cbar_scale = self.color_bar_scales[c]
            cols = group_data.columns[i:i_end]
            y_start = cur_row_height
            y_end = y_start + cur_height
            self.plot_heatmap_cell(group_data[cols], ax, x_start, x_end, 
                y_start, y_end, cbar_scale, lines)

        for y in lines:
            if y < 0: 
                y = y_end+y
                box_y_span = (y, y_end)
            else:
                box_y_span = (0, y)

            # draw background box for annotated zoom in region
            plot_utils.plot_rect(ax, -0, 
                    box_y_span[0], num_columns*num_times +\
                        col_spacing*(num_columns-1),
                    box_y_span[1]-box_y_span[0], 
                    color='#E6E7E8', zorder=1, lw=0)

        cur_row_height = y_end + row_spacing
        height += cur_height

        ax.patch.set_facecolor('red')

        return cur_row_height


    def plot_heatmap_cell(self, plot_data, ax, x_start, x_end, y_start, y_end,
        cbar_scale, lines):
        height = len(plot_data)
        im = ax.imshow(plot_data,
            aspect=self.aspect, 
            vmin=-cbar_scale, vmax=cbar_scale, cmap='RdBu_r',
            extent=[x_start, x_end, y_start, y_end], zorder=3,
            rasterized=True)

        x_span = (x_start, x_end)
        y_span = (y_start, y_end)

        # translate additinoal lines to draw to y-position
        updated_lines = []
        for y in lines:
            if y < 0: y = y_end+y
            updated_lines.append(y)
        ax.patch.set_alpha(0.0)

        _draw_grid(ax, x_span, y_span, height, self.num_times, updated_lines)

def _draw_grid(ax, x_span, y_span, rows, columns, annot_lines):

    (x_start, x_end) = x_span
    (y_start, y_end) = y_span

    lines = []
    white_lines = []
    # draw rows
    if y_span[1]-y_span[0] < 100:
        for i in range(rows):
            _add_hline(white_lines, x_span, y_start+i)

    # draw columns
    for i in range(columns):
        _add_vline(white_lines, x_start+i, y_span)

    # draw box around feature
    _add_vline(lines, x_start, y_span)
    _add_vline(lines, x_end, y_span)
    _add_hline(lines, x_span, y_start)
    _add_hline(lines, x_span, y_end)

    for line in annot_lines:
        _add_hline(lines, x_span, line)

    # draw lines
    lc = mc.LineCollection(white_lines, color='white', linewidth=0.25, zorder=5)
    ax.add_collection(lc)
    lc = mc.LineCollection(lines, color='black', linewidth=1.5, zorder=100)
    ax.add_collection(lc)


def _add_hline(lines, xs, y):
    lines.append(((xs[0], y), (xs[1], y)))


def _add_vline(lines, x, ys):
    lines.append(((x, ys[0]), (x, ys[1])))


def _make_fake_cbar(ax, vlim, title, scale=1, cmap='RdBu_r'):
    """Make fake cbar by drawing gradient box"""
    
    Y = np.linspace(-vlim, vlim, 100)

    data = np.array(Y).reshape(len(Y),1).T
    ax.imshow(data, cmap=cmap,
        extent=[-10, 10, 0, 1.], vmin=-vlim, vmax=vlim)
    ax.set_yticks([0.5])
    ax.set_yticklabels([title])

    plot_utils.format_ticks_font(ax, fontsize=14)
    ax.tick_params(axis='y', labelsize=16, length=0, pad=30)

    ax.set_xticks(np.linspace(-10, 10, 5))

    ticklabel_values = np.linspace(-vlim, vlim, 5) * scale
    ticklabels = ['%.0f' % t for t in ticklabel_values]
    ticklabels[0] = '<' + ticklabels[0]
    ticklabels[-1] = '>' + ticklabels[-1]
    ax.set_xticklabels(ticklabels)

    # ax.set_title(title)

