import numpy as np
from time import time


class IOKR:
    
    def __init__(self):
        self.X_tr = None
        self.Y_tr = None
        self.input_kernel = None
        self.output_kernel = None
        self.sy = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = 0
        
    def fit(self, X, Y, L, input_kernel, output_kernel):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        Kx = input_kernel(self.X_tr, Y=self.X_tr)
        n = Kx.shape[0]
        self.M = np.linalg.inv(Kx + n * L * np.eye(n))
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        t0 = time()
        Kx = self.input_kernel(X_te, Y=self.X_tr)
        Ky = self.output_kernel(self.Y_tr, Y=Y_c)
        scores = Kx.dot(self.M).dot(Ky)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return self.Y_tr[idx_pred]



class SIOKR:
    
    def __init__(self):
        self.X_tr = None
        self.Y_tr = None
        self.input_kernel = None
        self.output_kernel = None
        self.S = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = 0
        
    def fit(self, X, Y, S, L, input_kernel, output_kernel, mu=1e-8):
        
        t0 = time()
        self.S = S
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel          
        n = X.shape[0]
        s = S.size[0]
        self.Y_tr = Y
        SKST = S.multiply_Gram_both_sides(X, self.input_kernel)
        KST = S.multiply_Gram_one_side(X, self.input_kernel, X)
        B = KST.T.dot(KST) + n * L * SKST
        B_inv = np.linalg.inv(B + mu * np.eye(s))
        self.M = B_inv.dot(KST.T)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        t0 = time()
        K_te_trST = self.S.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        Ky = self.output_kernel(self.Y_tr, Y=Y_c)
        scores = (K_te_trST.dot(self.M)).dot(Ky)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return self.Y_tr[idx_pred]



class ISOKR:
    
    def __init__(self):
        self.X_tr = None
        self.Y_tr = None
        self.input_kernel = None
        self.output_kernel = None
        self.S = None
        self.KyST = None
        self.SKyST_inv = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = 0
        
    def fit(self, X, Y, S, L, input_kernel, output_kernel, mu=0):
        
        t0 = time()
        self.S = S
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        Kx = self.input_kernel(X, X)
        n = Kx.shape[0]
        s = S.size[0]
        self.M = np.linalg.inv(Kx + n * L * np.eye(n))
        self.KyST = S.multiply_Gram_one_side(Y, self.output_kernel, Y)
        self.SKyST_inv = np.linalg.inv(S.multiply_Gram_both_sides(Y, self.output_kernel) + mu * np.eye(s))
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        t0 = time()
        Kx = self.input_kernel(X_te, self.X_tr)
        SKy = self.S.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c, right=False)
        scores = Kx.dot(self.M).dot(self.KyST).dot(self.SKyST_inv).dot(SKy)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return self.Y_tr[idx_pred]



class SISOKR:
    
    def __init__(self):
        self.X_tr = None
        self.Y_tr = None
        self.input_kernel = None
        self.output_kernel = None
        self.S_in = None
        self.S_out = None
        self.M = None
        self.KyST = None
        self.SKyST_inv = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = 0
        
    def fit(self, X, Y, S_in, S_out, L, input_kernel, output_kernel, mu_in=1e-8, mu_out=0):
        
        t0 = time()
        self.S_in = S_in
        self.S_out = S_out
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        n = X.shape[0]
        s_in = S_in.size[0]
        s_out = S_out.size[0]
        SKST = S_in.multiply_Gram_both_sides(X, self.input_kernel)
        KST = S_in.multiply_Gram_one_side(X, self.input_kernel, X)
        B = KST.T.dot(KST) + n * L * SKST
        self.M = np.linalg.inv(B + mu_in * np.eye(s_in)).dot(KST.T)
        self.KyST = S_out.multiply_Gram_one_side(Y, self.output_kernel, Y)
        self.SKyST_inv = np.linalg.inv(S_out.multiply_Gram_both_sides(Y, self.output_kernel) + mu_out * np.eye(s_out))
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        if Y_c is None:
            Y_c = self.Y_tr.copy()
        t0 = time()
        KxST = self.S_in.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        SKy = self.S_out.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c, right=False)
        scores = KxST.dot(self.M).dot(self.KyST).dot(self.SKyST_inv).dot(SKy)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return self.Y_tr[idx_pred]