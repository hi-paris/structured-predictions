import numpy as np
from time import time


class IOKR:
    
    def __init__(self, L, input_kernel, output_kernel, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        Kx = self.input_kernel(self.X_tr, Y=self.X_tr)
        n = Kx.shape[0]
        self.M = np.linalg.inv(Kx + n * self.L * np.eye(n))
        self.output_kernel.fit(Y)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        Kx_te_tr = self.input_kernel(X, Y=self.X_tr)
        Ky_tr_c = self.output_kernel.transform(Y_c).T
        scores = Kx_te_tr.dot(self.M).dot(Ky_tr_c)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]
    
    def predict_scores(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        Kx_te_tr = self.input_kernel(X, Y=self.X_tr)
        Ky_tr_c = self.output_kernel.transform(Y_c).T
        scores = Kx_te_tr.dot(self.M).dot(Ky_tr_c)
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return scores

    def predict_topk(self, X, Y_c=None, k=5):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        Kx_te_tr = self.input_kernel(X, Y=self.X_tr)
        Ky_tr_c = self.output_kernel.transform(Y_c).T
        scores = Kx_te_tr.dot(self.M).dot(Ky_tr_c)
        idx_topk = np.argsort(scores, axis=1)[:, -k:]
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        n = X.shape[0]
        Y_pred_topk = np.empty((n, 0))
        for i in range(k):
            Y_pred_i = Y_c[idx_topk[:, -(i + 1)]].reshape((-1, 1)).copy()
            Y_pred_topk = np.hstack((Y_pred_topk, Y_pred_i))
        scores_topk = np.flip(np.sort(scores, axis=1), axis=1)[:, :k]
        
        return Y_pred_topk, scores_topk
    
    
    
class SIOKR:
    
    def __init__(self, L, input_kernel, output_kernel, R_x, mu_x=1e-8, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.R_x = R_x
        self.mu_x = mu_x
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        n = X.shape[0]
        m_x = self.R_x.size[0]
        self.Y_tr = Y
        KRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, X)
        RxKRxT = self.R_x.multiply_matrix_one_side(KRxT, right=False)
        B = KRxT.T.dot(KRxT) + n * self.L * RxKRxT
        B_inv = np.linalg.inv(B + self.mu_x * np.eye(m_x))
        self.M = B_inv.dot(KRxT.T)
        self.output_kernel.fit(Y)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        Kx_te_trRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, Y=self.X_tr)
        Ky_tr_c = self.output_kernel.transform(Y_c).T
        scores = Kx_te_trRxT.dot(self.M).dot(Ky_tr_c)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]
    
    def predict_scores(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        Kx_te_trRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, Y=self.X_tr)
        Ky_tr_c = self.output_kernel.transform(Y_c).T
        scores = Kx_te_trRxT.dot(self.M).dot(Ky_tr_c)
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return scores

    def predict_topk(self, X, Y_c=None, k=5):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        Kx_te_trRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, Y=self.X_tr)
        Ky_tr_c = self.output_kernel.transform(Y_c).T
        scores = Kx_te_trRxT.dot(self.M).dot(Ky_tr_c)
        idx_topk = np.argsort(scores, axis=1)[:, -k:]
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        n = X.shape[0]
        Y_pred_topk = np.empty((n, 0))
        for i in range(k):
            Y_pred_i = Y_c[idx_topk[:, -(i + 1)]].reshape((-1, 1)).copy()
            Y_pred_topk = np.hstack((Y_pred_topk, Y_pred_i))
        scores_topk = np.flip(np.sort(scores, axis=1), axis=1)[:, :k]
        
        return Y_pred_topk, scores_topk



class ISOKR:
    
    def __init__(self, L, input_kernel, output_kernel, R_y, mu_y=0, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.R_y = R_y
        self.mu_y = mu_y
        self.KYRyT = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        Kx = self.input_kernel(X, X)
        n = Kx.shape[0]
        m_y = self.R_y.size[0]
        Omega = np.linalg.inv(Kx + n * self.L * np.eye(n))
        self.KYRyT = self.R_y.multiply_Gram_one_side(Y, self.output_kernel, Y)
        RyKYRyT = self.R_y.multiply_matrix_one_side(self.KYRyT, right=False)
        RyKYRyT_inv = np.linalg.inv(RyKYRyT.copy() + self.mu_y * np.eye(m_y))
        self.M = Omega.dot(self.KYRyT).dot(RyKYRyT_inv)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
            RyKy_tr_c = self.KYRyT.T
        else:
            RyKy_tr_c = self.R_y.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c).T
        Kx_te_tr = self.input_kernel(X, self.X_tr)
        scores = Kx_te_tr.dot(self.M).dot(RyKy_tr_c)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]
    
    def predict_scores(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
            RyKy_tr_c = self.KYRyT.T
        else:
            RyKy_tr_c = self.R_y.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c).T
        Kx_te_tr = self.input_kernel(X, self.X_tr)
        scores = Kx_te_tr.dot(self.M).dot(RyKy_tr_c)
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return scores

    def predict_topk(self, X, Y_c=None, k=5):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
            RyKy_tr_c = self.KYRyT.T
        else:
            RyKy_tr_c = self.R_y.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c).T
        Kx_te_tr = self.input_kernel(X, self.X_tr)
        scores = Kx_te_tr.dot(self.M).dot(RyKy_tr_c)
        idx_topk = np.argsort(scores, axis=1)[:, -k:]
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        n = X.shape[0]
        Y_pred_topk = np.empty((n, 0))
        for i in range(k):
            Y_pred_i = Y_c[idx_topk[:, -(i + 1)]].reshape((-1, 1)).copy()
            Y_pred_topk = np.hstack((Y_pred_topk, Y_pred_i))
        scores_topk = np.flip(np.sort(scores, axis=1), axis=1)[:, :k]
        
        return Y_pred_topk, scores_topk



class SISOKR:
    
    def __init__(self, L, input_kernel, output_kernel, R_x, R_y, mu_x=1e-8, mu_y=0, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.R_x = R_x
        self.R_y = R_y
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.KYRyT = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        n = X.shape[0]
        m_x = self.R_x.size[0]
        m_y = self.R_y.size[0]
        KXRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, X)
        RxKRxT = self.R_x.multiply_matrix_one_side(KXRxT, right=False)
        B = KXRxT.T.dot(KXRxT) + n * self.L * RxKRxT
        Omega = np.linalg.inv(B + self.mu_x * np.eye(m_x)).dot(KXRxT.T)
        self.KYRyT = self.R_y.multiply_Gram_one_side(Y, self.output_kernel, Y)
        RyKYRyT = self.R_y.multiply_matrix_one_side(self.KYRyT, right=False)
        RyKYRyT_inv = np.linalg.inv(RyKYRyT.copy() + self.mu_y * np.eye(m_y))
        self.M = Omega.dot(self.KYRyT).dot(RyKYRyT_inv)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
            RyKy_tr_c = self.KYRyT.T
        else:
            RyKy_tr_c = self.R_y.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c).T
        Kx_te_trRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, Y=self.X_tr)
        scores = Kx_te_trRxT.dot(self.M).dot(RyKy_tr_c)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]
    
    def predict_scores(self, X, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
            RyKy_tr_c = self.KYRyT.T
        else:
            RyKy_tr_c = self.R_y.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c).T
        Kx_te_trRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, Y=self.X_tr)
        scores = Kx_te_trRxT.dot(self.M).dot(RyKy_tr_c)
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return scores
    
    def predict_topk(self, X, Y_c=None, k=5):
        
        t0 = time()
        if Y_c is None:
            Y_c = self.Y_tr.copy()
            RyKy_tr_c = self.KYRyT.T
        else:
            RyKy_tr_c = self.R_y.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c).T
        Kx_te_trRxT = self.R_x.multiply_Gram_one_side(X, self.input_kernel, Y=self.X_tr)
        scores = Kx_te_trRxT.dot(self.M).dot(RyKy_tr_c)
        idx_topk = np.argsort(scores, axis=1)[:, -k:]
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        n = X.shape[0]
        Y_pred_topk = np.empty((n, 0))
        for i in range(k):
            Y_pred_i = Y_c[idx_topk[:, -(i + 1)]].reshape((-1, 1)).copy()
            Y_pred_topk = np.hstack((Y_pred_topk, Y_pred_i))
        scores_topk = np.flip(np.sort(scores, axis=1), axis=1)[:, :k]
        
        return Y_pred_topk, scores_topk