#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
"""
Created on Sun Sep 13 15:45:26 2020

@author: samuel
"""
import numpy as np
import pandas as pd

df = pd.read_csv(
    '/home/samuel/Bureau/zip.train', sep=" ", header=None)

digits = df.to_numpy()

classes = digits[:, 0]
digits = digits[:, 1:-1]
# %%
bdd = []
X = []
y = []

for i in range(10):
    bdd.append(digits[classes == i][:100])
    X.append(digits[classes == i][:100][:128])
    y.append(digits[classes == i][:100][128:])
# %%
gamma = 0.01
kernel = ("gaussian", gamma)
# %%
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

n_train = 800

bdd_train = [None] * 10
bdd_test = [None] * 10

for i in range(10):
    # bdd_train.append(bdd[i][sample_without_replacement(n_population=100, n_samples=n_train//10)])
    bdd_train[i], bdd_test[i] = train_test_split(bdd[i], train_size=n_train // 10)

bdd_train = np.concatenate(bdd_train)
bdd_test = np.concatenate(bdd_test)

np.random.shuffle(bdd_train)
np.random.shuffle(bdd_test)

X_train = bdd_train[:, :128]
y_train = bdd_train[:, 128:]

X_test = bdd_test[:, :128]
y_test = bdd_test[:, 128:]

# %%

from stpredictions.models.OK3._classes import OK3Regressor, ExtraOK3Regressor
from stpredictions.models.OK3._forest import RandomOKForestRegressor, ExtraOKTreesRegressor

ok3 = OK3Regressor(kernel=kernel, max_leaf_nodes=50).fit(X_train, y_train)
extraok3 = ExtraOK3Regressor(kernel=kernel, max_leaf_nodes=50).fit(X_train, y_train)

okforest = RandomOKForestRegressor(kernel=kernel, max_leaf_nodes=50).fit(X_train, y_train)
extraokforest = ExtraOKTreesRegressor(kernel=kernel, max_leaf_nodes=50).fit(X_train, y_train)

# %%

y_pred1 = ok3.predict(X_test)
y_pred2 = extraok3.predict(X_test)
y_pred3 = okforest.predict(X_test)
y_pred4 = extraokforest.predict(X_test)

# %%
mse1 = np.mean(
    np.sum((y_test - y_pred1) ** 2, axis=1))  # gamma 0.01, maxleaf=50 ==> 70 ;  gamma 0.01, maxleaf=10 ==> 77
mse2 = np.mean(np.sum((y_test - y_pred2) ** 2, axis=1))
mse3 = np.mean(np.sum((y_test - y_pred3) ** 2, axis=1))
mse4 = np.mean(
    np.sum((y_test - y_pred4) ** 2, axis=1))  # gamma 0.01, maxleaf=50 ==> 55 ;  gamma 0.01, maxleaf=10 ==> 70

rbf_loss1 = 2 * (1 - np.exp(- gamma * mse1))
rbf_loss2 = 2 * (1 - np.exp(- gamma * mse2))
rbf_loss3 = 2 * (1 - np.exp(- gamma * mse3))
rbf_loss4 = 2 * (1 - np.exp(- gamma * mse4))

print("MSE 1 :", mse1)
print("MSE 2 :", mse2)
print("MSE 3 :", mse3)
print("MSE 4 :", mse4)

print("RBF loss 1 : ", rbf_loss1)
print("RBF loss 2 : ", rbf_loss2)
print("RBF loss 3 : ", rbf_loss3)
print("RBF loss 4 : ", rbf_loss4)

# %%

# import matplotlib.pyplot as plt

test_ex = 3

plt.imshow(X_test[test_ex].reshape(8, 16), cmap='gray')
plt.title("Input upper image")
plt.show()
plt.imshow(y_test[test_ex].reshape(8, 16), cmap='gray')
plt.title("True output lower image")
plt.show()
plt.imshow(y_pred[test_ex].reshape(8, 16), cmap='gray')
plt.title("Predicted output lower image")
plt.show()

plt.imshow(np.vstack((X_test[test_ex].reshape(8, 16),
                      y_test[test_ex].reshape(8, 16),
                      -np.ones((1, 16)),
                      X_test[test_ex].reshape(8, 16),
                      y_pred[test_ex].reshape(8, 16))),
           cmap='gray')
plt.title("Up : True image\nDown : Image with the predicted lower half")
# plt.imsave('/home/samuel/Bureau/prediction_ex_'+str(test_ex)+'.png', np.vstack((X_test[test_ex].reshape(8,16),
#                                                                                 y_test[test_ex].reshape(8,16),
#                                                                                 -np.ones((1,16)),
#                                                                                 X_test[test_ex].reshape(8,16),
#                                                                                 y_pred[test_ex].reshape(8,16))),
#            cmap='gray')

# %%

pixels_importances = ok3.feature_importances_

plt.imshow(pixels_importances.reshape(8, 16), cmap='gray')
plt.title("Image of pixels (features) importances")
plt.show()

'''