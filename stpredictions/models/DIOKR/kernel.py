import numpy as np
import torch
import copy

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_kernel(X, Y=None):
    """Compute linear Gram matrix between X and Y (or X)
    Parameters
    ----------
    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed
    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None
    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    if Y is None:
        Y = X

    K = X @ Y.T

    return K


def rbf_kernel(X, Y=None, gamma=None):
    """Compute rbf Gram matrix between X and Y (or X)
    Parameters
    ----------
    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed
    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None
    gamma: float
           Gamma parameter of the kernel (see sklearn implementation)
    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    K_tmp = X_norm + Y_norm - 2. * torch.mm(X, torch.t(Y))
    K_tmp *= -gamma
    K = torch.exp(K_tmp)

    return K


def gaussian_tani_kernel(X, Y=None, gamma=None):
    """Compute Gaussian Tanimoto Gram matrix between X and Y (or X)
    Parameters
    ----------
    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed
    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None
    gamma: float
           Gamma parameter of the kernel (see sklearn implementation)
    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_np, Y_np = X.data.numpy(), Y.data.numpy()

    scalar_products = X_np.dot(Y_np.T)
    X_norms = np.linalg.norm(X_np, axis=1) ** 2
    Y_norms = np.linalg.norm(Y_np, axis=1) ** 2
    nomi = scalar_products
    den = X_norms.reshape(-1, 1) + Y_norms.reshape(1, -1) - scalar_products
    K_t = np.divide(nomi, den, where=den != 0)
    K_gt = np.exp(- gamma * 2 * (1 - K_t))
    K_gt_torch = torch.from_numpy(K_gt).float()

    return K_gt_torch


def get_anchors_gaussian_rff(dim_input, dim_rff, gamma):
    return gamma * torch.randn(dim_input, dim_rff)


class Kernel(object):

    def __init__(self):
        pass


class Gaussian(Kernel):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        return rbf_kernel(X, Y, self.gamma)


class GaussianTani(Kernel):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        return gaussian_tani_kernel(X, Y, self.gamma)


class LearnableGaussian(Kernel):

    def __init__(self, gamma, model, optim_params):
        self.gamma = gamma
        self.is_learnable = True
        self.model = model
        self.optim_params = optim_params
        self.train_losses = []
        self.test_losses = []
        self.times = [0]

    def compute_gram(self, X, Y=None):
        if Y is None:
            return rbf_kernel(self.model.forward(X), Y=None, gamma=self.gamma)
        else:
            return rbf_kernel(self.model.forward(X), self.model.forward(Y), self.gamma)

    def append_train_loss(self, train_loss):
        self.train_losses.append(train_loss)

    def append_test_loss(self, test_loss):
        self.test_losses.append(test_loss)

    def append_time(self, time):
        self.times.append(time)

    def clone_kernel(self):
        clone_gamma = self.gamma
        clone_model = copy.deepcopy(self.model)
        clone_optim_params = self.optim_params
        return LearnableGaussian(clone_gamma, clone_model, clone_optim_params)

    def clear_memory(self):
        self.train_losses, self.test_mse, self.test_f1, self.times = [], [], [], [0]


class LearnableLinear(Kernel):

    def __init__(self, model, optim_params):
        self.is_learnable = True
        self.model = model
        self.optim_params = optim_params
        self.train_losses = []
        self.test_losses = []
        self.times = [0]

    def model_forward(self, X):
        return self.model.forward(X)

    def compute_gram(self, X, Y=None):
        if Y is None:
            return linear_kernel(self.model.forward(X), Y=None)
        else:
            return linear_kernel(self.model.forward(X), self.model.forward(Y))

    def clone_kernel(self):
        clone_model = copy.deepcopy(self.model)
        clone_optim_params = self.optim_params
        return LearnableLinear(clone_model, clone_optim_params)

    def append_train_loss(self, train_loss):
        self.train_losses.append(train_loss)

    def append_test_loss(self, test_loss):
        self.test_losses.append(test_loss)

    def append_time(self, time):
        self.times.append(time)

    def clear_memory(self):
        self.train_losses, self.test_mse, self.test_f1, self.times = [], [], [], [0]
