
import numpy as np
import pandas
from matplotlib import collections as mc
from pipeline.plot_utils import plot_rect
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from pipeline.plot_utils import plot_scatter_density, plot_aux_line
from matplotlib.colors import ListedColormap
from matplotlib import cm
from scipy.stats import norm
from pipeline.nuc_kernel.nucleosome_linkages import get_nuc, get_p123, get_l321
from pipeline.nuc_kernel.nucleosome_linkages import get_transcript_nucleosomes
from pandas.core.series import Series


class NucleosomeEntities:

    def __init__(self, span, chrom, cross_corr, kernel_width = 50.0, 
        overlap_dist = 180, nuc_signal_min = 0.005,
        times = [0.0, 7.5, 15, 30, 60, 120], nucleosomes=None):

        self.nuc_signal_min = nuc_signal_min
        self.overlap_dist = overlap_dist
        self.span = span
        self.chrom = chrom
        self.times = times

        kernel_width_2 = kernel_width / 2

        if nucleosomes is not None:
            mask = ((nucleosomes.chr == chrom) &
                    (nucleosomes.mid > span[0]) & 
                    (nucleosomes.mid < span[1]))
            self.peaks = nucleosomes[mask]

        self.cross_correlation = cross_corr[(cross_corr.chr == chrom) & 
            (cross_corr.mid >= span[0] - kernel_width_2) & 
            (cross_corr.mid <= span[1] + kernel_width_2)].copy()\
            .sort_values(['time', 'mid'])

    def peak_line_color(self, x):
        if x < 0.025: color = self.color_corr(x)
        else: color = '#f47777'
        return color

    def color_corr(self, x):
        greys = plt.get_cmap('Greys')
        newcolors = greys(np.linspace(0.00, 0.8, 255))
        greys = ListedColormap(newcolors, name='nucleosome_cross')

        min_corr = 0
        max_corr = 0.3

        scale = (x - min_corr)/(max_corr - min_corr)
        scale = max(min(scale, 1), 0)
        return greys(scale)

    def color_gen_corr(self, x):
        return self.color_corr(x)

    def calculate_peaks(self):
        self.peaks = pandas.DataFrame()
        for time in self.times:
            cur_peaks = self.calculate_peaks_time(time)
            self.peaks = self.peaks.append(cur_peaks).reset_index(drop=True)

    def calculate_peaks_time(self, time):
        nuc_signal_min = self.nuc_signal_min
        nuc_window = self.overlap_dist
        nuc_window_2 = nuc_window / 2
        cur_cross = self.cross_correlation
        cur_cross = cur_cross[cur_cross.time == time].copy()
        cur_cross = cur_cross[cur_cross.cross >= nuc_signal_min].sort_values('cross', 
            ascending=False).reset_index(drop=True)
        peaks = pandas.DataFrame()

        while len(cur_cross) > 0:
            cur_peak = cur_cross.loc[0]
            near_peaks = cur_cross[(cur_cross.mid > cur_peak.mid - nuc_window_2) & 
                (cur_cross.mid < cur_peak.mid + nuc_window_2)]
            cur_peak = cur_peak[['mid',
             'cross',
             'time',
             'chr']].copy()
            cur_peak['start'] = near_peaks.mid.min()
            cur_peak['end'] = near_peaks.mid.max()
            peaks = peaks.append(cur_peak)
            cur_cross = cur_cross.drop(near_peaks.index.values)
            cur_cross = cur_cross.reset_index(drop=True)

        if len(peaks) == 0: return peaks

        peaks = peaks.rename(columns={'mid':'peak', 'cross':'peak_cross'})
        return peaks

    def find_nearest_index(self, cur_peak, peaks, time, search_padding=20):
        """Find nearest peak from another time. return the index of the peak"""
        cur_peaks = peaks[(peaks.time == time) & 
                          (cur_peak.start <= peaks.end + search_padding) & 
                          (peaks.start <= cur_peak.end - search_padding) & 
                          (peaks.time == time)].copy()
        if len(cur_peaks) == 0: return None
        cur_peaks['dist'] = np.abs(cur_peaks.mid - cur_peak.mid)
        cur_peaks = cur_peaks.sort_values('dist')

        return peaks.loc[cur_peaks[0:1].index[0]]

    def create_peak(self, peak, time):
        """Create a peak by projecting the boundary of a 
        peak at another time onto a time"""
        cc = self.cross_correlation
        cc = cc[(cc.time == time) & 
                (cc.mid >= peak.start) & 
                (cc.mid <= peak.end)].sort_values('cross', ascending=False)
        new_peak = peak.copy()

        # get the largest cross correlation, may be a subnucleosomal peak
        max_cross = cc.loc[cc.cross.idxmax()]

        new_peak.time = time
        new_peak['generated'] = True
        new_peak['cross'] = max_cross.cross
        new_peak['mid'] = max_cross.mid

        return new_peak

    def find_linkages_fwd_bck(self, last_peak, linkages, times, 
        working_peaks, link_idx):
        if len(times) == 0: return linkages, []

        del_indices = []

        for time in times:
            cur_peak = self.find_nearest_index(last_peak, working_peaks, time)
            
            # didnt find a peak, create a new one
            if cur_peak is None:
                last_peak = self.create_peak(last_peak, time)
            # found a peak, add to deletion index
            else: 
                last_peak = cur_peak
                last_peak['generated'] = False
                del_indices.append(last_peak.name)

            last_peak['link'] = link_idx
            linkages = linkages.append(last_peak)

        return linkages, del_indices

    def find_linkages_peak(self, cur_peak, link_idx, working_peaks, 
        times=[0.0, 7.5, 15, 30, 60, 120]):
        ""
        linkages = pandas.DataFrame()
        
        cur_peak['link'] = link_idx
        cur_peak['generated'] = False
        idx_time = np.where(np.array(times) == cur_peak.time)[0][0]
        linkages = linkages.append(cur_peak)

        # find peaks backwards
        linkages, del_1 = self.find_linkages_fwd_bck(cur_peak, linkages, 
            np.array(times)[np.arange(idx_time-1, -1, -1)], working_peaks, link_idx)

        # find peaks forward
        linkages, del_2 = self.find_linkages_fwd_bck(cur_peak, linkages, 
            np.array(times)[np.arange(idx_time+1, len(times))], working_peaks, link_idx)

        return linkages.sort_values(['link', 'time']), ([cur_peak.name] + del_1 + del_2)

    def find_linkages(self):
        """find peak with the largest signal across all times

        collect to peaks across earlier and later times

        if a peak does not exist earlier or later,  create one by casting 
        it from the previous high signal
        and finding a peak in that window"""

        working_peaks = self.peaks.sort_values('cross', ascending=False)

        # combine peaks across times
        linkages = pandas.DataFrame()
        all_linkages = pandas.DataFrame()

        # iterate through peaks with the highest signal
        while len(working_peaks) > 0:

            # get the highest peak available
            cur_peak = working_peaks.loc[working_peaks[0:1].index[0]].copy()
            link_idx = (cur_peak.chr.astype(int).astype(str) + "_" + 
                cur_peak.mid.astype(int).astype(str))
            
            # find linkages forward and backwards, create if necessary
            linkages, del_indices = self.find_linkages_peak(cur_peak, link_idx, 
                working_peaks)

            # drop peaks in already found linkages
            working_peaks = working_peaks.drop(del_indices)

            all_linkages = all_linkages.append(linkages)

        self.peaks = all_linkages


