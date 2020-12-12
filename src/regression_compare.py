
import sys
sys.path.append('.')

import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ols import OLS
from src.timer import TimingContext
from src.utils import print_fl
from src import plot_utils
from src.gp import GP

class RegressionCompare:

    def __init__(self, reg_model=OLS):
        
        full_model = reg_model('Full')
        body_model = reg_model('Gene body')
        prom_model = reg_model('Promoter')
        intercept_model = reg_model('Intercept')
        cc_model = reg_model('Cross correlation')
        occ_model = reg_model('Occupancy')
        sm_model = reg_model('Small fragments')
        nuc_model = reg_model('Nucleosome fragments')

        sense_model = reg_model('Sense')
        antisense_model = reg_model('Antisense')

        body_model.design_matrix(incl_prom=False, incl_gene=True, incl_antisense=False)
        prom_model.design_matrix(incl_prom=True, incl_gene=False, incl_antisense=False)
        intercept_model.design_matrix(incl_prom=False, incl_gene=False, incl_antisense=False)
        cc_model.design_matrix(incl_cc=True, incl_occ=False, incl_antisense=False)
        occ_model.design_matrix(incl_cc=False, incl_occ=True, incl_antisense=False)
        sm_model.design_matrix(incl_small=False, incl_antisense=False)
        nuc_model.design_matrix(incl_nuc=False, incl_antisense=False)
        sense_model.design_matrix(incl_antisense=False)
        antisense_model.design_matrix(incl_sense=False)

        self.full_model = full_model
        self.body_model = body_model
        self.prom_model = prom_model
        self.intercept_model = intercept_model
        self.cc_model = cc_model
        self.occ_model = occ_model
        self.sm_model = sm_model
        self.nuc_model = nuc_model
        self.sense_model = sense_model
        self.antisense_model = antisense_model

        self.models = [intercept_model, 
                       sm_model, nuc_model, 
                       prom_model, body_model,
                       cc_model, occ_model, 
                       sense_model, antisense_model,
                       full_model]

    def fit(self, k=10):

        with TimingContext() as timing:
            for model in self.models:
                print_fl("Fitting %s" % model.name)
                model.fit_cv(log=False, k=k)
                print_fl("  " + timing.get_time())

        times = model.times

        df = pd.DataFrame(index=times)
        for model in self.models:
            df[model.name] = model.mse
        self.mse = df

        df = pd.DataFrame(index=times)
        for model in self.models:
            df[model.name] = model.r2
        self.r2 = df


    def plot_compare(self, metric='r2', rename={}):
        
        fig, ax = plt.subplots(figsize=(8,6))
        fig.tight_layout(rect=[0.05, 0.35, 0.95, 0.90])

        x = np.arange(6)
        times = self.models[0].times

        models = self.models
        model_names = [model.name for model in self.models]

        tab10 = plt.get_cmap('tab10')
        colors = ['#555555', 
            tab10(3), tab10(3),
            tab10(0), tab10(0), 
            tab10(1), tab10(1), 
            tab10(4), tab10(4),
            tab10(2)]

        markers = ['o', 
                   'v', '^',
                   '<', '>',
                   'P', 'D',
                   'o', 'o',
                   's']

        fill_styles = [
        'none',
        'full','none',
        'full','none',
        'full','none',
        'full','none',
        'full'
        ]

        # TODO: call general plot function
        # plot_compare()
        

def plot_compare(data, markers, colors, fill_styles, metric='r2', 
    rename={}, show_legend=False):
            
    plot_utils.apply_global_settings()

    model_names = data.columns

    fig, ax = plt.subplots(figsize=(9,4))
    fig.tight_layout(rect=[0.05, 0.05, 0.65, 0.9])

    x = np.arange(6)
    times=[0, 7.5, 15, 30, 60, 120]
    spacing = 0.09
    n_models = len(model_names)

    for i in range(n_models):

        model_name = model_names[i]
        model_data = data[model_name]
        label = model_name
        if label in rename.keys():
            label = rename[label]

        x_pos = x - spacing*(n_models-1)/2. + spacing*i
        
        ax.plot(x_pos, model_data, marker=markers[i],
            color=colors[i], label=label.replace('Promoter occupancy', 
                'Small fragment occupancy'), markersize=7, alpha=1.,
            fillstyle=fill_styles[i], linewidth=0,
            zorder=10)

    if metric == 'mse':
        ax.set_ylim(1, 8.5)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title("Model evaluation, MSE")

        for i in np.arange(0, 10, 1):
            ax.axhline(y=i, linewidth=0.1, linestyle='solid', 
                color='#303030', zorder=1)
    else:
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        ax.set_ylim(-0.05, 1.0)
        ax.set_ylabel('Coefficient of determination, $R^2$', fontsize=16)
        ax.set_title("Model evaluation, $R^2$", fontsize=20)

        for i in np.arange(0, 1.0, 0.1):
            ax.axhline(y=i, linewidth=0.1, linestyle='solid', 
                color='#303030', zorder=1)

    ax.set_xticks(np.arange(0, len(times)))
    ax.set_xticklabels(['%s\'' % t for t in times])

    ax.set_xlim(.5, 5.5)
    ax.tick_params('x', labelsize=16, length=0, width=0, pad=10)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1.), fontsize=14, frameon=False)

    for i in range(6):
        ax.axvline(x=i+0.5, linewidth=1.0, color='#303030')

def load_results(parent_dir):
    results = pd.DataFrame()
    for path in os.listdir(parent_dir):
        if path.endswith('.csv') and path.startswith('res_'):
            full_path = "%s/%s" % (parent_dir, path)
            cur = pd.read_csv(full_path).rename(columns={'Unnamed: 0': 'time'})
            name = path.replace('.csv', '').replace('res_', '')
            cur['model'] = name
            results = results.append(cur)

    results = results.pivot(index='time', columns='model', values='r2')

    results = results[['Intercept', 'RNA only', 'Promoter occupancy', 
                       'Nucleosome disorganization', 
                       'Combined chromatin', 
                       'Full']].copy()
    return results

def plot_compare_r2(parent_dir, show_legend=False):

    results = load_results(parent_dir)

    tab10 = plt.get_cmap('tab10')
    colors = ['#555555', 
                tab10(3), tab10(5),
                tab10(0),
                tab10(4),
                tab10(2)]

    markers = ['o', 
               'd',
               '<', '>',
               'D',
               'o',
               's']

    fill_styles = [
            'none',
            'none','none',
            'none','full',
            'full','none',
            'full',
            'full'
            ]

    plot_compare(results, markers, colors, fill_styles, 
                 rename={'Combined chromatin': 
                 "Small fragment occupancy +\nNucleosome disorganization"},
                 show_legend=show_legend)
    

def main():

    print_fl("Loading models")
    gp_compare = RegressionCompare(reg_model=GP)
    gp_compare.fit(k=10)

    gp_compare.plot_compare(metric='r2')
    plt.savefig('output/gp/r2.pdf', transparent=True)

    gp_compare.plot_compare(metric='mse')
    plt.savefig('output/gp/mse.pdf', transparent=True)

    gp_compare.full_model.plot_fit()
    plt.savefig('output/gp/full.pdf', transparent=True)

    gp_compare.full_model.plot_fit(120)
    plt.savefig('output/gp/full_120.pdf', transparent=True)

    gp_compare.mse.T.to_csv('output/gp/model_mse.csv', float_format='%.4f')
    gp_compare.r2.T.to_csv('output/gp/model_r2.csv', float_format='%.4f')

if __name__ == '__main__':
    main()





