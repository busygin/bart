import numpy as np
from bart import compute_bart


class ComputeBart(object):
    def __init__(self, pb=0.5, alpha=0.95, beta=2.0, nd=1000, lambda_=1.0, burn=100, m=200, nc=100, nu=3, kfac=2.0):
        self.bart_regressor = compute_bart()
        self.bart_regressor.set_mcmc_params(pb_=pb, alpha_=alpha, beta_=beta)
        self.bart_regressor.set_run_params(nd_=nd, lambda_=lambda_, burn_=burn, m_=m, nc_=nc, nu_=nu, kfac_=kfac)

    def fit_and_predict(self, X, y, X1):
        X_ = X if X.flags['C_CONTIGUOUS'] else np.ascontiguousarray(X)
        y_ = y if y.flags['C_CONTIGUOUS'] else np.ascontiguousarray(y)
        X1_ = X1 if X1.flags['C_CONTIGUOUS'] else np.ascontiguousarray(X1)
        y1 = np.empty(shape=X1.shape[0], dtype=float)

        self.bart_regressor.set_insample_matrix(X_)
        self.bart_regressor.set_insample_target(y_)
        self.bart_regressor.set_outsample_matrix(X1_)
        self.bart_regressor.set_outsample_target(y1)
        self.bart_regressor.fit()

        return y1
