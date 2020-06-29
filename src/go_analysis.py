
from src.chromatin_metrics_data import ChromatinDataStore
from src.gene_ontology import GeneOntology
from src.datasets import read_orfs_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import plot_utils
from src.utils import get_std_cutoff, print_fl

class GOChromatinAnalysis:

    def __init__(self, datastore=None, agg_fun=np.max, N=300, filepath=None):

        if filepath is None:

            self.store = datastore
            # promoter small fragments sorted by min
            prom_data = self.store.promoter_sm_occupancy_delta
            prom_data = prom_data.loc[agg_fun(prom_data, axis=1).sort_values(ascending=False).index]
            self.promoter_sm_occupancy_delta = prom_data

            # disorganization sorted by min
            dis_data = self.store.gene_body_disorganization_delta
            dis_data = dis_data.loc[agg_fun(dis_data, axis=1).sort_values(ascending=False).index]
            self.gene_body_disorganization_delta = dis_data
            self.agg_fun = agg_fun

            # chromatin data sorted by both
            self.chromatin_data = self.store.chromatin_data
            self.gene_ontology = GeneOntology()

            self.orfs = datastore.orfs
        else:
            self.terms_res = pd.read_csv(filepath).set_index('term')

        self.N = N


    def subset_genes(self, head=None, tail=None):

        if head is not None: 
            subset_func = pd.Series.head
            N = head
        elif tail is not None:  
            subset_func = pd.Series.tail
            N = tail

        self.N = N

        data = self.agg_fun(self.gene_body_disorganization_delta, axis=1)
        self.cur_disorg_orfs = subset_func(data.sort_values(ascending=False), N).index.values
        #self.cur_disorg_orfs = get_std_cutoff(data, cutoff).index.values

        data = self.agg_fun(self.promoter_sm_occupancy_delta, axis=1)
        self.cur_promoter_orfs = subset_func(data.sort_values(ascending=False), N).index.values
        # self.cur_promoter_orfs = get_std_cutoff(data, cutoff).index.values

        data = self.agg_fun(self.chromatin_data, axis=1)
        self.cur_chromatin_orfs = subset_func(data.sort_values(ascending=False), N).index.values
        # self.cur_chromatin_orfs = get_std_cutoff(data, cutoff).index.values

        # self.std_cutoff = cutoff

        print_fl("Disorganization ORFs: %d" % len(self.cur_disorg_orfs))
        print_fl("Promoter ORFs:        %d" % len(self.cur_promoter_orfs))
        print_fl("Chromatin ORFs:       %d" % len(self.cur_chromatin_orfs))


    def run_go(self, fdr_sig=0.01):
        self.fdr_sig = fdr_sig
        (self.promoters_sig_results, 
         self.promoters_results) = self._run_go_index(self.cur_promoter_orfs)
        (self.disorg_sig_results,
         self.disorg_results) = self._run_go_index(self.cur_disorg_orfs)
        (self.chrom_sig_results,
         self.chrom_results) = self._run_go_index(self.cur_chromatin_orfs)
        

    def _run_go_index(self, index):
        gene_ontology = self.gene_ontology
        names = self.orfs.loc[index]['name'].values
        gene_ontology.run_go(names, sig=self.fdr_sig)
        return gene_ontology.results_sig_df, gene_ontology.results_df, 

    def collect_counts(self):
        disorg_results = self.disorg_sig_results
        prom_results = self.promoters_sig_results
        chrom_results = self.chrom_sig_results

        d = set(disorg_results['name'].values)
        p = set(prom_results['name'].values)
        c = set(chrom_results['name'].values)

        # list of terms
        go_terms = d.union(p).union(c)

        df = pd.DataFrame(index=go_terms)

        keys = ['PRSM', 'DORG', 'BOTH']
        results = [prom_results, disorg_results, chrom_results]

        for i in range(len(keys)):
            key = keys[i]
            result = results[i]
            cur = result.set_index('name').join(df)
            df.loc[cur.index, key] = -np.log2(cur['fdr_bh'])/np.log2(10)

        df = df.fillna(0).astype(float)

        # remove high-level terms
        drop_items = {'molecular_function', 
                      'cytoplasm', 
                      'cellular_component',
                      'biological_process',
                      'nucleolus',
                      'cytoplasmic vesicle'}
        df = df.drop(set(df.index.values).intersection(drop_items))
        df = df.reset_index().rename(columns={'index': 'term'}).set_index('term')

        self.terms_res = df

        return df

    def plot_bar(self, activated_genes=True):

        if not activated_genes:
            title_cat = 'decrease'
        else:
            title_cat = 'increase'

        title = ("Greatest %s in various\nchromatin scores, N=300" % title_cat)

        plot_utils.apply_global_settings(30)

        # df = self.collect_counts()
        df = self.terms_res

        sorted_idx = df.max(axis=1).sort_values(ascending=True).index
        df = df.loc[sorted_idx]

        df = df.tail(8)

        prom_sm_vals = df['PRSM'].values
        disog_vals = df['DORG'].values
        both = df['BOTH'].values

        y = np.arange(len(prom_sm_vals)) 
        height = 0.225
        spacing = 0.05

        fig, ax = plt.subplots(figsize=(11, 14))
        fig.tight_layout(rect=[0.35, 0.15, 0.90, 0.87])
        fig.patch.set_alpha(0.0)

        if activated_genes:
            reds = plt.get_cmap('Reds')
            colors = [reds(0.5),reds(0.25), reds(0.8)]
            edgecolors = [reds(0.8), reds(0.6), reds(0.8),]
        else:
            blues = plt.get_cmap('Blues')
            colors = [blues(0.5),blues(0.3), blues(0.9)]
            edgecolors = [blues(0.9),blues(0.7), blues(0.9)]

        prom_y = y + (height+spacing)
        dis_y = y
        both_y = y - (height+spacing)

        rects1 = ax.barh(prom_y, prom_sm_vals, height, 
            label='Promoter occupancy', 
            color=colors[0], 
            alpha=1)

        for i in range(len(dis_y)):
            ax.barh(y=dis_y[i], width=disog_vals[i], height=height, 
                label='Nucleosome\ndisorganization' if i == 0 else None, 
                color=colors[2])

        for i in range(len(both_y)):
            ax.barh(both_y[i], both[i], height, 
                label='Combined' if i == 0 else None, 
                color=colors[1],
                facecolor=colors[1],
                edgecolor=edgecolors[1],
                hatch='\\\\',
                 alpha=1,
                 linewidth=2)

        group_vals = [prom_sm_vals, disog_vals, both]
        group_ys = [prom_y, dis_y, both_y]

        # determine scale to offset labels
        max_val = df.max().max()
        inc = max_val / 100.

        for g in range(3):
            vals = group_vals[g]
            ys = group_ys[g]
            for i in range(len(vals)):
                val = vals[i] + inc
                if val > inc:
                    ax.text(val, ys[i], ("10$^{-%0.1f}$" % val), 
                        va='center', fontsize=14, 
                        fontdict={'family':'Open Sans'})

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.legend(loc=4, bbox_to_anchor=(0.5, -0.25),
         frameon=False, fontsize=18)
        ax.set_yticks(np.arange(len(df)))

        for y in np.arange(1, len(prom_sm_vals)):
            ax.axhline(y=(y-0.5), color='#D0D0D0', linewidth=1)

        terms = df.index.values
        terms = [t[0:1].upper() + t[1:] for t in terms]

        new_terms = []
        for t in terms:

            if t == "Maturation of SSU-rRNA from tricistronic rRNA transcript (SSU-rRNA, 5.8S rRNA, LSU-rRNA)":
                t = "Maturation of SSU-rRNA"

            t_spl = t.split(' ')
            if len(t) > 30: 
                new_terms.append(' '.join(t_spl[:2]) + '\n' + ' '.join(t_spl[2:]))
            else: new_terms.append(t)
        terms = new_terms

        ax.set_yticklabels(terms)

        plot_utils.format_ticks_font(ax)
        plot_utils.format_ticks_font(ax, which='y', fontsize=12)

        max_fdr = self.terms_res.max().max()

        if activated_genes:
            ticks = np.arange(0, 6, 1)
        else:    
            ticks = np.arange(0, 100, 20)
            ax.set_xlim(0, round(max_fdr+15))

        ax.set_xticks(ticks)
        ax.set_title(title, fontsize=30)
        ax.set_xticklabels(-ticks)
        ax.set_xlabel('log$_{10}$ FDR', fontsize=20)

        ax.tick_params(axis='y', labelsize=18, length=0, pad=20)
        ax.tick_params(axis='x', labelsize=16, pad=10)

