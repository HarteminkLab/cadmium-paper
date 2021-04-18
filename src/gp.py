
import sys
sys.path.append('.')

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from src.chromatin_summary_plots import plot_distribution
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import pandas as pd
from chromatin_metrics_data import ChromatinDataStore

from src.timer import TimingContext, Timer
from sklearn.metrics import r2_score
from pandas.core.series import Series
from scipy.stats import pearsonr 
from src.datasets import read_orfs_data
from src.transformations import difference
from src.chromatin_metrics_data import pivot_metric, add_suffix
from src.datasets import read_orfs_data
from sklearn import preprocessing
from src.plot_utils import plot_density_scatter
from src.utils import print_fl
from src.colors import parula
from src.math_utils import convert_to_latex_sci_not
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from config import rna_dir, mnase_dir, paper_orfs, SUBSET_GPR_GENES


class GP:

    def __init__(self, name, times=[0, 7.5, 15, 30, 60, 120], sample_N=None, 
        results_path=None):

        print_fl("Loading %s" % name)

        self.name = name
        self.sample_N = sample_N
        self.times = times

        if results_path is None:
            self.design_matrix()
        else:
            self.load_results(results_path)

        self.l_scale = 1
        self.l_bounds = 1, 10

    def load_results(self, path):
        full_res = read_orfs_data(path).copy()
        Y = full_res[[]].copy()
        Y_pred = full_res[[]].copy()
        r2 = pd.DataFrame(index=self.times)
        r2['r2'] = 0.0

        for time in self.times:
            predicted = full_res['%.1f_predicted' % (time)]
            true = full_res['%.1f_true' % (time)]

            Y_pred.loc[:, time] = predicted
            Y.loc[:, time] = true

            r2.loc[time, 'r2'] = r2_score(true, predicted)

        self.Y = Y
        self.Y_predict = Y_pred
        self.r2 = r2


    def load_design_matrix(self, 
        incl_times=[0, 7.5, 15, 30, 60, 120],
        incl_prom=True, incl_gene=True,
        incl_occ=True, incl_cc=True,
        incl_small=True, incl_nuc=True,
        incl_shift=False, antisense=False):

        datastore = ChromatinDataStore()

        orfs = datastore.orfs
        orfs_idx = orfs.index.values

        # start with empty X
        X = orfs[[]].copy()

        # only use these regions for metrics
        regions = ['-200_0', '0_500']

        if antisense:
            strand_name = 'antisense'
        else:
            strand_name = 'sense'

        if incl_shift: 
            p1_positions = datastore.p1
            p2_positions = datastore.p2
            p3_positions = datastore.p3

            # no shift if nucleosome was not called
            p1_shift = add_suffix(difference(p1_positions), '_p1')
            p2_shift = add_suffix(difference(p2_positions), '_p2')
            p3_shift = add_suffix(difference(p3_positions), '_p3')

            shifts = orfs[[]].join(p1_shift)
            shifts = shifts.join(p2_shift)
            shifts = shifts.join(p3_shift).fillna(0.0)

            X = X.join(shifts)

        # occupancies
        if incl_occ:
            occupancies = orfs[[]]
            occupancy = datastore.occupancy

            # select designated columns
            occ_columns = ['-200_0_len_0_100', '0_500_len_0_100',
                           '-200_0_len_144_174','0_500_len_144_174']
            occupancy = occupancy[occ_columns]                           

            # add relevant columns to occupancy df
            for column in occupancy.columns.values:

                # select regions
                if column.startswith('0') and not incl_gene: continue
                if column.startswith('-') and not incl_prom: continue

                # select lengths
                if column.endswith('len_0_100') and not incl_small: continue
                if column.endswith('len_144_174') and not incl_nuc: continue
                
                pivoted_occ = pivot_metric(occupancy, column, '_%s_occ' % column)
                occupancies = occupancies.join(pivoted_occ, how='outer')

            # impute missing data with 0
            occupancies = occupancies.fillna(0.0)
            X = X.join(occupancies, how='outer')

        # cross correlation
        if incl_cc:
    
            from src.entropy import load_orf_entropies_by_cc_type

            cc_nuc = load_orf_entropies_by_cc_type('triple', strand_name)
            cc_small = load_orf_entropies_by_cc_type('small', strand_name)

            cc = orfs[[]]

            if incl_nuc:
                for col in cc_nuc.columns:

                    if col.startswith('0') and not incl_gene: continue
                    if col.startswith('-') and not incl_prom: continue

                    cur_pvt = pivot_metric(cc_nuc, col, '_%s_nuc_cc' % col)
                    cc = cc.join(cur_pvt, how='outer')
                
            if incl_small:
                for col in cc_small.columns:

                    if col.startswith('0') and not incl_gene: continue
                    if col.startswith('-') and not incl_prom: continue

                    cur_pvt = pivot_metric(cc_small, col, '_%s_small_cc' % col)
                    cc = cc.join(cur_pvt, how='outer')

            # impute missing data with mean cc "entropy"
            # applies to genes where antisense transcripts could not be called
            cc = cc.fillna(cc.mean())
            X = X.join(cc, how='outer')

        return X

    def design_matrix(self, 
        incl_times=[0, 7.5, 15, 30, 60, 120],
        incl_prom=True, incl_gene=True,
        incl_occ=True, incl_cc=True,
        incl_small=True, incl_nuc=True,
        incl_sense=True, incl_antisense=True,
        incl_shift=False, predict_TPM=True,
        include_TPM_0=True, scale=True, logfold=True):

        if self.name == 'Intercept': include_TPM_0 = False

        orfs = paper_orfs

        # TODO: Testing to see how GPR performs on good set of genes @ 120' only
        # more complex subsetting if we want to subset different genes per 
        # each time point
        if SUBSET_GPR_GENES:
            path = '%s/good_p1_nucs_gene_set_120.csv' % mnase_dir
            subset_idx = read_orfs_data(path).index.values
            orfs = orfs.loc[subset_idx]

            print_fl("Subsetting to well-positioned +1 nucleosomes, N=%d" % len(orfs))

        orfs_idx = orfs.index.values

        X = orfs[[]].copy()

        if incl_sense:
            sense_X = self.load_design_matrix(incl_times=incl_times,
                                              incl_prom=incl_prom,
                                              incl_gene=incl_gene,
                                              incl_occ=incl_occ,
                                              incl_cc=incl_cc,
                                              incl_small=incl_small,
                                              incl_nuc=incl_nuc,
                                              incl_shift=incl_shift)
            X = X.join(sense_X)

        if incl_antisense:
            antisense_X = self.load_design_matrix(incl_times=incl_times,
                                              incl_prom=incl_prom,
                                              incl_gene=incl_gene,
                                              incl_occ=incl_occ,
                                              incl_cc=incl_cc,
                                              incl_small=incl_small,
                                              incl_nuc=incl_nuc,
                                              incl_shift=False, # no antisense shift data
                                              antisense=True)
            X = X.join(antisense_X, lsuffix='_sense', rsuffix='_antisense')

        # load outcome
        # index = model.Y.index.values

        # predict absolute TPM level (log2)
        if predict_TPM:
            TPM = read_orfs_data('%s/sense_TPM.csv' % rna_dir).loc[orfs_idx]
            Y = np.log2(TPM+0.1)

        # predict log2 fold change
        else:
            xrate = read_orfs_data('%s/orf_xrates.csv' % rna_dir)
            xrate_logfold = read_orfs_data('%s/orf_xrates_log2fold.csv' % rna_dir)
            Y = xrate_logfold.loc[orfs_idx]

        # add TPM at time 0
        if include_TPM_0:
            X['0.0_TPM'] = np.log2(TPM[0].copy())

        if self.sample_N is not None:
            np.random.seed(123)
            orfs_idx = X.index
            orfs_idx = np.random.choice(orfs_idx, self.sample_N, 
                replace=False)

            X = X.loc[orfs_idx]
            Y = Y.loc[orfs_idx]

        # TODO: replace infinite values with 0
        X = X.replace([np.inf, -np.inf], 0.0)

        if len(X.columns) > 0:

            # logfold covariates
            if logfold:
                columns = X.columns
                # log transform cross correlation and occupancy columns
                logfold_cols = columns[columns.str.contains('_cc_') | columns.str.contains('_occ_')]
                X.loc[:, logfold_cols] = np.log2(X[logfold_cols] + 0.1)

            # scale covariates
            if scale:
                X.loc[:] = preprocessing.scale(X)
            
        X['intercept'] = 1

        self.X, self.Y = X, Y

        return X, Y


    def plot_covariates(self, time):

        X = self.X.copy()

        columns = X.columns
        cols_selected = columns[columns.str.startswith(str(time))]

        fig, axs = plt.subplots(5, 4, figsize=(15, 15))
        fig.tight_layout(rect=[0.3, 0.15, 0.95, 0.90])
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        axs = axs.flatten()

        for i in range(len(cols_selected)):
            col = cols_selected[i]
            ax = axs[i]
            
            col_data = X[col]

            ax.hist(col_data, bins=100)
            
            col_spl = col.split('_')
            title = col_spl[0] + '\n' + ' '.join(col_spl[1:])
            ax.set_title(title)


    def fit_cv(self, k=10, log=False, l_scale=1,
            l_bounds=(0.1, 100)):

        (self.last_models, self.mse, 
         self.r2, self.Y_predict) = fold_cross_validation(self.X, 
            self.Y, k=k, times=self.times, l_scale=l_scale,
            l_bounds=l_bounds, log=log)

        if log:
            print_fl("Fit %d-fold cross-validation with r2:" % k)

            for t in self.times:
                print_fl("%s:" % t)
                print_fl("\t%.5f" % (self.r2.loc[t]))

    def plot_correlation(self):
        fm = self
        X = fm.X
        Y = fm.Y

        X = X[select_time_columns(X.columns, 120.0)]

        data = pd.concat([X, Y], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.tight_layout(rect=[0.3, 0.15, 0.95, 0.90])

        im = ax.imshow(data.corr(), vmin=-1, vmax=1, cmap='RdBu_r')
        ticks = np.arange(0, len(data.columns), 6)+3
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        columns = []
        for c in data.columns[np.arange(0, len(data.columns), 6)].values:
            c = str(c).replace('0.0_', '')
            c = c.replace('_', ' ')
            c = c.replace('occ', '')
            c = c.replace('-200 0', 'Promoter')
            c = c.replace('0 500', 'Gene body')
            c = c.replace('len 0 100', 'small fragments')
            c = c.replace('nuc cc', 'nucleosomal cross correlation')
            c = c.replace('small cc', 'small fragments cross correlation')
            c = c.replace('0.0', 'Transcription log fold-change')
            c = c.replace('len 144 174', 'nucleosome fragments')
            columns.append(c)

        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.set_yticklabels(columns)
        ax.tick_params(length=0, width=0)
        cbar = plt.colorbar(im)
        cbar.ax.set_title('Correlation')

        for i in range(0, len(data.columns), 6):
            ax.axvline(i-0.5, color='white', linewidth=0.5)
            ax.axhline(i-0.5, color='white', linewidth=0.5)

def fit_gp(X, Y, l_scale=1., l_bounds=(1, 10)):
    """Use statsmodel to fit a linear model using OLS"""

    # scale parameters for regularization
    kernel = RBF(length_scale=l_scale, length_scale_bounds=l_bounds) \
        + WhiteKernel(0.0001)

    gp = GaussianProcessRegressor(kernel=kernel)
    fit = gp.fit(X, Y)

    return fit


def MSE(Y, Y_pred):
    return ((Y - Y_pred)**2).mean()


def get_fold_slice(X, Y, k, fold, time):

    N = len(X)
    fold_size = N/k

    end = (fold+1)*fold_size
    if (fold == k-1): end = N

    columns = X.columns

    # slice data
    test_slice = slice((fold*fold_size), end)
    test_orfs = X.index[test_slice]
    train_orfs = X.loc[~X.index.isin(test_orfs)].index
    
    X_train = X.loc[train_orfs]
    Y_train = Y.loc[train_orfs]

    X_test = X.loc[test_orfs]
    Y_test = Y.loc[test_orfs]

    cur_X_train = X_train[select_time_columns(columns, time)]
    cur_X_test = X_test[select_time_columns(columns, time)]

    return cur_X_train, Y_train[time], cur_X_test, Y_test[time]


def fold_cross_validation(X, Y, k=3, times=[0, 7.5, 15, 30, 60, 120],
    l_scale=1., l_bounds=(1, 10), time=False, log=False):

    np.random.seed(1)
    original_orfs = X.index.values
    shuffled_orfs_idx = X.index.values.copy()
    np.random.shuffle(shuffled_orfs_idx)

    Y_predict = pd.DataFrame(index=shuffled_orfs_idx)
    for t in times:
        Y_predict[t] = 0.

    N = len(X)
    fold_size = N/k

    timer = Timer()
    last_fold_models = {}

    for time in times:
        for fold in range(k):

            if log and fold % 1 ==  0: print_fl("%d/%d" % ((fold+1), k))

            X_train, Y_train, X_test, Y_test = get_fold_slice(X, Y, k, fold, time)
            test_orfs = X_test.index

            model = fit_gp(X_train.values, Y_train.values, l_scale, l_bounds)
            Y_pred = model.predict(X_test.values)

            r2 = r2_score(Y_test.values, Y_pred)

            Y_predict.loc[test_orfs, time] = Y_pred

            if log:
                print_fl(("\t%s - %s - r2 = %.3f" % 
                    (str(time), timer.get_time(), r2)))

            last_fold_models[time] = model

            if log:
                print_fl('')

    mse = MSE(Y.loc[shuffled_orfs_idx], Y_predict.loc[shuffled_orfs_idx])

    r2 = mse[[]].copy()
    for time in times:
        r2.loc[time] = r2_score(Y.loc[shuffled_orfs_idx][time], 
                                Y_predict.loc[shuffled_orfs_idx][time])

    return last_fold_models, mse, r2, Y_predict.loc[original_orfs]


def plot_res_distribution(model, predict_abs_TPM=True, selected_genes=[]):

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.90])
    plt.subplots_adjust(hspace=0.45, wspace=0.3 )
    axs = axs.flatten()
    times = model.times

    for i in range(len(times)):
        time = times[i]
        ax = axs[i]
        plot_res_distribution_time(model, time,
            predict_abs_TPM=predict_abs_TPM, ax=ax, 
            selected_genes=selected_genes)

    plt.suptitle("%s model predictions" % model.name, fontsize=30)


def plot_res_distribution_time(model, time, predict_abs_TPM=True, 
    ax=None, selected_genes=[], show_pearsonr=True, plot_aux='both', 
    show_r2=False, tight_layout=None):

    y = model.Y[time]
    x = model.Y_predict[time]

    if predict_abs_TPM:
        label = "log$_2$ transcript level, TPM"
        lims = (0, 15)
    else:
        label = "log$_2$ fold-change\ntranscription rate, TPM/min"
        lims = (-15, 15)

    if ax is None:

        title = ("%s model predictions\n %s min" % 
                 (model.name, str(time)))

        if pearsonr:
            cor, pval = pearsonr(x, y)
            pval = convert_to_latex_sci_not(pval)
            title = ("%s, Pearson's r=%.2f, p=%s" % 
                    (title, cor, pval))

        if show_r2:
            title = ("%s', $R^2$=%.2f" % 
                     (title, model.r2.loc[time]))
    else:
        title = ("%s', $R^2$=%.2f" % 
                 (str(time), model.r2.loc[time]))

    custom_formatting={}

    if time == 30:
        custom_formatting = {
            'HSP26': {'ha': 'right', 'va': 'bottom'},
            'RPS7A': {'ha': 'left', 'va': 'bottom'},
            'CKB1': {'ha': 'right', 'va': 'top'}
        }

    elif time == 120:
        custom_formatting = {
            'HSP26': {'ha': 'right', 'va': 'top'}
        }

    model = plot_distribution(x, y, 
                              'Predicted %s' % label,
                              'True %s' % label,
                              highlight=selected_genes,
                              title=title,
                              xlim=lims,
                              ylim=lims, 
                              xstep=2,
                              ystep=2,
                              pearson=False, 
                              plot_minor=False,
                              highlight_format=custom_formatting,
                              tight_layout=tight_layout,
                              ha='right', plot_aux=plot_aux, ax=ax)

def select_time_columns(columns, time, 
    times=[0, 7.5, 15, 30, 60, 120], snap_shot_time=True):
    """
    snap_shot_time: True
        Select columns with selected time and 0.0

    otherwse:
        Select time columns up to and including relevant time
    """

    i = times.index(time)
    cur_times = times[:i+1]

    if snap_shot_time:
        cur_times = ['0.0', '%.1f' % time]
    else:
        cur_times = ['%.1f' % c for c in cur_times]

    sel = [c.split('_')[0] in cur_times for c in columns]
    columns = list(columns[sel]) + ['intercept']
    return columns
