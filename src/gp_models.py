
import sys
sys.path.append('.')

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import print_fl
from src.timer import Timer
from src.tasks import TaskDriver, child_done
from config import WATCH_TMP_DIR, USE_SLURM, gp_dir, SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV
from src.slurm import submit_sbatch
from src.gp import GP


def full_model(sample_N=None):
    full_model = GP('Full', sample_N=sample_N)
    full_model.design_matrix(incl_shift=True)
    return full_model

def antisense_model(sample_N=None):
    antisense_model = GP('Antisense', sample_N=sample_N)
    antisense_model.design_matrix(incl_shift=False, incl_antisense=True, 
        incl_sense=False)
    return antisense_model

def body_model(sample_N=None):
    body_model = GP('Gene body', sample_N=sample_N)
    body_model.design_matrix(incl_prom=False, incl_gene=True, incl_antisense=False)
    return body_model

def prom_model(sample_N=None):
    prom_model = GP('Promoter', sample_N=sample_N)
    prom_model.design_matrix(incl_prom=True, incl_gene=False, incl_antisense=False)
    return prom_model

def intercept_model(sample_N=None):
    intercept_model = GP('Intercept', sample_N=sample_N)
    intercept_model.design_matrix(incl_prom=False, incl_gene=False, incl_antisense=False)
    return intercept_model

def rna_only_model(sample_N=None):
    intercept_model = GP('RNA only', sample_N=sample_N)
    intercept_model.design_matrix(incl_prom=False, incl_gene=False, incl_antisense=False)
    return intercept_model

def shift_model(sample_N=None):
    shift_model = GP('Nucleosome shift', sample_N=sample_N)
    shift_model.design_matrix(incl_shift=True,
        incl_prom=False, incl_gene=False, incl_antisense=False)
    return shift_model

def combined_model(sample_N=None):
    combined_model = GP('Combined chromatin', sample_N=sample_N)

    sm_model = small_promoter_model(sample_N)
    disorg_model = gene_disorg_model(sample_N)

    X1 = sm_model.X
    X2 = disorg_model.X

    same_cols = set(X1.columns).intersection(set(X2.columns))

    for col in same_cols:
        X1 = X1.drop(col, axis=1)

    combined_model.X = X1.join(X2)
    return combined_model

def small_promoter_model(sample_N=None):
    small_prom_model = GP('Promoter occupancy', sample_N=sample_N)
    small_prom_model.design_matrix(incl_prom=True, incl_gene=False, 
        incl_cc=False, incl_occ=True,
        incl_small=True, incl_nuc=False,
        incl_antisense=False)
    return small_prom_model

def gene_disorg_model(sample_N=None):
    gene_disorg_model = GP('Nucleosome disorganization', sample_N=sample_N)
    gene_disorg_model.design_matrix(incl_prom=False, incl_gene=True, 
        incl_cc=True, incl_occ=False,
        incl_small=False, incl_nuc=True,
        incl_antisense=False)
    return gene_disorg_model

def sense_model(sample_N=None):
    sense_model = GP('Sense', sample_N=sample_N)
    sense_model.design_matrix(incl_antisense=False)
    return sense_model

def cc_model(sample_N=None):
    cc_model = GP('Cross correlation', sample_N=sample_N)
    cc_model.design_matrix(incl_cc=True, incl_occ=False, incl_antisense=False)
    return cc_model

def occ_model(sample_N=None):
    occ_model = GP('Occupancy', sample_N=sample_N)
    occ_model.design_matrix(incl_cc=False, incl_occ=True, incl_antisense=False)
    return occ_model


def sm_model(sample_N=None):
    sm_model = GP('Small fragments', sample_N=sample_N)
    sm_model.design_matrix(incl_small=False, incl_antisense=False)
    return sm_model

def nuc_model(sample_N=None):
    nuc_model = GP('Nucleosome fragments', sample_N=sample_N)
    nuc_model.design_matrix(incl_nuc=False, incl_antisense=False)
    return nuc_model

def get_model_funs():

    models_dic = {
        'Full': full_model,
        # 'Gene body': body_model,
        # 'Promoter': prom_model,
        'Intercept': intercept_model,
        'RNA only': rna_only_model,
        # 'Nucleosome shift': shift_model,
        'Combined chromatin': combined_model,
        'Promoter occupancy': small_promoter_model,
        'Nucleosome disorganization': gene_disorg_model,
        # 'Sense': sense_model,
        # 'Cross correlation': cc_model,
        # 'Occupancy': occ_model,
        # 'Small fragments': sm_model,
        # 'Nucleosome fragments': nuc_model,
        # 'Antisense': antisense_model
    }

    return models_dic


def run_models(save_dir, timer):

    task_name = 'gp'

    # launch gp models
    print_fl("Loading models...", end='')
    models = get_model_funs()

    print_fl("Running %d models..." % len(models), end='')
    driver = TaskDriver(task_name, WATCH_TMP_DIR, len(models.keys()), timer=timer)
    driver.print_driver()

    for name, model in models.items():
        if not USE_SLURM:
            run_model(name, save_dir)
            child_done(task_name, WATCH_TMP_DIR, name)
            pass
        else:
            exports = ("MODEL=%s,SLURM_WORKING_DIR=%s,CONDA_PATH=%s,CONDA_ENV=%s" % \
                      (name.replace(' ', '_'),
                       SLURM_WORKING_DIR, CONDA_PATH, CONDA_ENV))
            script = 'scripts/4_analysis/gp.sh'
            submit_sbatch(exports, script, WATCH_TMP_DIR)

    driver.wait_for_tasks()
    print_fl()


def run_model(name, save_dir, predict_TPM=True):

    timer = Timer()

    print_fl("Loading %s model" % name)
    print_fl("Predicting TPM: %s" % predict_TPM)

    sample_N = None

    model_fun = get_model_funs()[name]
    model = model_fun(sample_N=sample_N)

    folds = 10
    print_fl("Fitting %d folds.." % folds)
    model.fit_cv(log=True, k=folds)

    # save models to disk
    res = model.Y.join(model.Y_predict,
     lsuffix='_true', rsuffix='_predicted')
    res.to_csv('%s/%s_results.csv' % (save_dir, name))

    res = pd.DataFrame({'r2':model.r2, 'mse':model.mse})
    res.to_csv('%s/res_%s.csv' % (save_dir, name))
    timer.print_time()


def main():
    name = sys.argv[1].replace('_', ' ')
    predict_TPM = sys.argv[2] == 'True'
    run_model(name, gp_dir, predict_TPM)
    child_done('gp', WATCH_TMP_DIR, name)

if __name__ == '__main__':
    main()


