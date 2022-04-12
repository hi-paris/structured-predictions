from time import time

import numpy as np
import torch


class IOKR(object):
    """
        Main class implementing IOKR + OEL
    """

    def __init__(self, path_to_candidates=None):

        # OEL
        self.oel_method = None
        self.oel = None

        # Output Embedding Estimator
        self.L = None
        self.input_gamma = None
        self.input_kernel = None
        self.linear = False
        self.output_kernel = None
        self.Omega = None
        self.n_anchors_krr = -1

        # Data
        self.n_tr = None
        self.X_tr = None
        self.Y_tr = None
        self.UY_tr = None
        self.K_x = None
        self.K_y = None

        # vv-norm
        self.vv_norm = None

        # Decoding
        self.decode_weston = False
        self.path_to_candidates = path_to_candidates

    def fit(self, X_s, Y_s, L=None, input_gamma=None,
            input_kernel=None, linear=False,
            output_kernel=None, Omega=None, oel_method=None,
            K_X_s=None, K_Y_s=None, verbose=0):
        """
            Fit OEE (Output Embedding Estimator = KRR) and OEL with supervised/unsupervised data
        """

        # Saving
        self.X_tr = X_s.clone()
        self.Y_tr = Y_s.clone()
        self.L = L
        self.input_gamma = input_gamma
        self.input_kernel = input_kernel
        self.linear = linear
        self.output_kernel = output_kernel
        self.oel_method = oel_method

        # Training OEE
        t0 = time()

        # Gram computation
        if not self.linear:
            if K_X_s is None:
                self.K_x = self.input_kernel.compute_gram(self.X_tr)
            else:
                self.K_x = K_X_s

        if K_Y_s is None:
            self.K_y = self.output_kernel.compute_gram(self.Y_tr)
        else:
            self.K_y = K_Y_s

        # KRR computation (standard or Nystrom approximated)
        if not self.linear:
            self.n_tr = self.K_x.shape[0]
            if Omega is None:
                if self.n_anchors_krr == -1:
                    M = self.K_x + self.n_tr * self.L * torch.eye(self.n_tr)
                    self.Omega = torch.inverse(M)
                else:
                    n_anchors = self.n_anchors_krr
                    idx_anchors = torch.from_numpy(np.random.choice(self.n_tr, n_anchors, replace=False)).int()
                    K_nm = self.K_x[:, idx_anchors]
                    K_mm = self.K_x[np.ix_(idx_anchors, idx_anchors)]
                    M = K_nm.T @ K_nm + self.n_tr * self.L * K_mm
                    self.Omega = K_nm @ torch.inverse(M)
                    self.X_tr = self.X_tr[idx_anchors]
            else:
                self.Omega = Omega
        else:
            m = self.input_kernel.model_forward(self.X_tr).shape[1]
            M = self.input_kernel.model_forward(self.X_tr).T @ self.input_kernel.model_forward(
                self.X_tr) + m * self.L * torch.eye(m)
            self.Omega = torch.inverse(M) @ self.input_kernel.model_forward(self.X_tr).T

        # vv-norm computation
        if self.n_anchors_krr == -1:
            if not self.linear:
                self.vv_norm = torch.trace(self.Omega @ self.K_x @ self.Omega @ self.K_y)
            else:
                self.vv_norm = torch.trace(self.Omega.T @ self.Omega @ self.K_y)

        if verbose > 0:
            print(f'KRR training time: {time() - t0}', flush=True)

        # Training time
        t0 = time()

        if verbose > 0:
            print(f'Training time: {time() - t0}', flush=True)

    def sloss(self, K_x, K_y):

        """
            Compute the square loss (train MSE)
        """

        n_te = K_x.shape[1]
        if self.n_anchors_krr == -1:
            A = K_x.T @ self.Omega
        else:
            A = self.Omega @ K_x
            A = A.T

        # \|psi(y)\|^2
        norm_y = torch.diag(K_y)

        product_h_y = torch.einsum('ij, ji -> i', A, K_y)
        norm_h = torch.einsum('ij, jk, ki -> i', A, K_y, A.T)
        se = norm_h - 2 * product_h_y + norm_y

        mse = torch.mean(se)

        return mse

    def get_vv_norm(self):
        """
            Returns the vv norm of the estimator fitted
        """
        return self.vv_norm
