
import numpy as np


def MSE(Y, Y_pred):
    return ((Y - Y_pred)**2).mean()


def fold_cross_validation(X, Y, time, folds=10):

    np.random.seed(123)
    v = X.index.values.copy()
    np.random.shuffle(v)

    Y_predict = pandas.DataFrame(index=v.copy())
    Y_predict['TPM'] = 0.

    N = len(X)
    fold_size = N/folds

    for fold in range(folds):

        end = (fold+1)*fold_size
        if (fold == folds-1): end = N

        test_slice = slice((fold*fold_size), end)

        test_orfs = X.index[test_slice]
        train_orfs = X.loc[~X.index.isin(test_orfs)].index
        
        X_train = X.loc[train_orfs]
        Y_train = Y.loc[train_orfs].values

        X_test = X.loc[test_orfs]
        Y_test = Y.loc[test_orfs][time].values

        model = fit(np.matrix(X_train), np.array(Y_train), plot=False)
        Y_pred = model.predict(sm.add_constant(np.matrix(X_test)))
        Y_predict.loc[test_orfs, 'TPM'] = Y_pred

    mse = MSE(Y.loc[v][time], Y_predict.loc[v]['TPM'])

    return mse, Y_predict.loc[Y.index]