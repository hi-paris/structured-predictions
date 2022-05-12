#!/usr/bin/env python
# coding: utf-8

import torch
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from stpredictions.models.DIOKR import cost, kernel, estimator, IOKR
from sklearn.metrics.pairwise import rbf_kernel
from stpredictions.datasets.load_data import load_from_arff, load_bibtex_test_from_arff, load_bibtex_train_from_arff
from scipy.linalg import block_diag
import numpy as np
from os.path import join
from stpredictions.models.DIOKR.utils import project_root

# import seaborn as sns
# plt.style.use('seaborn')


# Load data

x_train, y_train = load_bibtex_train_from_arff()
x_train, y_train = x_train.todense(), y_train.todense()


x_test, y_test = load_bibtex_test_from_arff()
x_test, y_test = x_test.todense(), y_test.todense()

x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()


n = x_train.shape[0]

gamma_input = 0.1
gamma_output = 1.0

kernel_output = kernel.Gaussian(gamma_output)
lbda = 0.01

batch_size_train = 256


cost_function = cost.sloss_batch


d_out = int(x_train.shape[1]/2)
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], d_out),
    torch.nn.ReLU(),
)

optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)

kernel_input = kernel.LearnableGaussian(
    gamma_input, model_kernel_input, optim_params)


iokr = IOKR()

diokr_estimator = estimator.DIOKREstimator(kernel_input, kernel_output,
                                           lbda, iokr=iokr, cost=cost_function)


diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test, n_epochs=20, solver='sgd', batch_size_train=batch_size_train)


# plt.figure()
# plt.title("Training loss evolution when learning the kernel")
# plt.plot(diokr_estimator.kernel_input.train_losses, label="train")
# #plt.plot(diokr_estimator.kernel_input.test_mse, label="test")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()


# plt.figure()
# plt.title("Test MSE evolution when learning the kernel")
# #plt.plot(diokr_estimator.kernel_input.train_losses, label="train")
# plt.plot(diokr_estimator.kernel_input.test_mse, label="test")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()


diokr_estimator.kernel_input.times[-1]/60


from sklearn.metrics import f1_score

Y_pred_train = diokr_estimator.predict(x_test=x_train)
Y_pred_test = diokr_estimator.predict(x_test=x_test)
f1_train = f1_score(Y_pred_train, y_train, average='samples')
f1_test = f1_score(Y_pred_test, y_test, average='samples')
print("Train f1 score:", f1_train,"/", "Test f1 score:", f1_test)


