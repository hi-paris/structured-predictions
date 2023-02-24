import os
import time
import numpy as np

# sklearn
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from stpredictions.models.IOKR.Sketch import SubSample, pSparsified, Incomplete, Accumulation, Rademacher, Gaussian
from stpredictions.models.IOKR.SketchedIOKR import IOKR, SIOKR, ISOKR, SISOKR
from stpredictions.datasets.load_data import *

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Defining Gaussian kernel
def Gaussian_kernel(gamma):
    def Compute_Gram(X, Y=None):
        if Y is None:
            Y = X.copy()
        return rbf_kernel(X, Y, gamma=gamma)
    return Compute_Gram

# Load the bibtex dataset
x, y = load_bibtex_train_from_arff()
X_tr, Y_tr = x.todense(), y.todense()

x_test, y_test = load_bibtex_test_from_arff()
X_te, Y_te = x_test.todense(), y_test.todense()

Y_tr, Y_te = Y_tr.astype(int), Y_te.astype(int)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# X_tr, Y_tr, X_te, Y_te = load_bibtex()
n_tr = X_tr.shape[0]
n_te = X_te.shape[0]
input_dim = X_tr.shape[1]
label_dim = Y_tr.shape[1]

print(f'Train set size = {n_tr}')
print(f'Test set size = {n_te}')
print(f'Input dim. = {input_dim}')
print(f'Output dim. = {label_dim}')

# Selection of the hyper-parameters: rbf output kernel

Ls = np.logspace(-6, -4, 3)
sxs = np.logspace(1, 4, 4)
sys = np.logspace(1, 4, 4)
n_tr = X_tr.shape[0]
n_val = n_tr//5
f1_val_best = 0
clf = IOKR()
clf.verbose = 0


t0 = time.time()
for L in Ls:
    for sx in sxs:
        for sy in sys:
            input_kernel = Gaussian_kernel(gamma=1/(2 * sx))
            output_kernel = Gaussian_kernel(gamma=1/(2 * sy))
            clf.fit(X=X_tr[:-n_val], Y=Y_tr[:-n_val], L=L, input_kernel=input_kernel, output_kernel=output_kernel)
            Y_pred_val = clf.predict(X_te=X_tr[-n_val:])
            f1_val = f1_score(Y_pred_val, Y_tr[-n_val:], average='samples')
            if f1_val > f1_val_best:
                f1_val_best = f1_val
                sx_best = sx
                sy_best = sy
                L_best = L

print(f'Selection time: {time.time() - t0}')
print(f'Best selected parameters: L: {L_best} | sx: {sx_best} | sy: {sy_best}')

# Test rbf output kernel

clf.verbose = 1
input_kernel = Gaussian_kernel(gamma=1/(2 * sx_best))
output_kernel = Gaussian_kernel(gamma=1/(2 * sy_best))
clf.fit(X=X_tr, Y=Y_tr, L=L_best, input_kernel=input_kernel, output_kernel=output_kernel)
Y_pred_te = clf.predict(X_te=X_te)
f1_te = f1_score(Y_pred_te, Y_te, average='samples')
print(f'Test f1 score with selected parameters: {f1_te}')