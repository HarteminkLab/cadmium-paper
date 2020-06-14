
# import numpy as np
# from matplotlib import pyplot as plt

# import numpy as np
# from scipy import stats 
# import pandas as pd
# from scipy.spatial.distance import cdist


# class GP():

#     def __init__(self, X, y, X_predict, Y_outcome, nu=100, tau=0.2):
#         self.X = X
#         self.y = y
#         self.X_predict = X_predict
#         self.Y_outcome = Y_outcome
#         self.nu = nu
#         self.tau = tau

#     def predict(self):
#         X = self.X
#         y = self.y
#         X_predict = self.X_predict
#         y_predict = gp_predict(X, y, X_predict, nu=self.nu, tau=self.tau, 
#             plt_kernel=True)
#         self.y_pred = y_predict

#     def plot(self):
#         X = self.X
#         y = self.Y_outcome
#         X_predict = self.X_predict
#         Y_outcome = self.Y_outcome
#         Y_predict = self.y_pred

#         plt.figure(figsize=(4,4))
#         plt.scatter(Y_outcome, Y_predict)
#         plt.xlabel("Observed")
#         plt.ylabel("Predicted")

#         plt.figure(figsize=(4,4))
#         plt.scatter(X_predict[:, 5], Y_predict, label='prediction')
#         plt.scatter(X_predict[:, 5], Y_outcome, label='observed')
#         plt.legend()
#         plt.xlabel("X_5")
#         plt.ylabel("logfold xRate")


# def kernel(X_i,X_j,nu,tau):
#     """GP exponential kernel"""
#     nu, tau = float(nu), float(tau)
#     dist = cdist(X_j, X_i, 'euclidean')
#     return nu*np.exp(-1/(2*tau**2)*dist**2)


# def gp_predict(X, Y, X_star, nu=1, tau=50, plt_kernel=False, ret_posterior=False):
#     """Predict new X values with given X values trained on Gaussian Process regression. 
#     Exponential kernel."""

#     diag = 1 # for computational stability

#     n = len(X)
#     n_star = len(X_star)

#     K = kernel(X, X, nu, tau)
#     K_star_none = kernel(X_star, X, nu,tau)
#     K_star_star = kernel(X_star, X_star, nu, tau)
#     K_none_star = K_star_none.T

#     Kprecision = np.linalg.inv(K+diag*np.eye(n))

#     mu_star = np.dot(Y, np.matmul(Kprecision, K_star_none))
#     Sigma_star = K_star_star - np.matmul(K_none_star, np.matmul(Kprecision, K_star_none))
    
#     Y_pred = stats.multivariate_normal.rvs(mu_star, Sigma_star)
    
#     if plt_kernel:
#         plt.imshow(K)
#         plt.title("Kernel nu={}, tau={}".format(nu, tau))
#         plt.colorbar()

#     if ret_posterior:
#         return Y_pred, mu_star, Sigma_star

#     return Y_pred



