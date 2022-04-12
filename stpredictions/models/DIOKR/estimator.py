import torch.optim as optim
from scipy.linalg import block_diag

# from stpredictions.models.DIOKR.cost import *
# from stpredictions.models.DIOKR.kernel import *
from stpredictions.models.DIOKR.IOKR import *

dtype = torch.float


class DIOKREstimator(object):
    "DIOKR Class with fitting procedure using pytorch"

    def __init__(self, kernel_input, kernel_output, lbda, linear=False, iokr=None, Omega=None, cost=None, eps=None):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.lbda = lbda
        self.linear = linear
        self.iokr = iokr
        self.Omega = Omega
        self.cost = cost
        self.eps = eps

    def objective(self, x_batch, y_batch):
        "Computes the objectif function to be minimized, sum of the cost +regularization"

        self.iokr.fit(X_s=x_batch, Y_s=y_batch, L=self.lbda, input_kernel=self.kernel_input,
                      output_kernel=self.kernel_output, input_gamma=self.kernel_input.gamma)
        K_x = self.kernel_input.compute_gram(x_batch)
        K_y = self.kernel_output.compute_gram(y_batch)
        obj = self.iokr.sloss(K_x, K_y) + self.lbda * self.iokr.get_vv_norm()

        return obj

    def train_kernel_input(self, x_batch, y_batch, solver: str, t0):
        """
        One step of the gradient descent using Stochastic Gradient Descent during the fitting of the input kernel
        when using a learnable neural network kernel input
        """
        if solver not in ('sgd', 'adam'):
            raise ValueError(f"'solver' should be 'sgd' or 'adam',but not {solver}")
        if type(t0) not in (int, float):
            raise TypeError(f"'t0' should be an int, not a {type(t0)}")
        if solver == 'sgd':
            optimizer_kernel = optim.SGD(
                self.kernel_input.model.parameters(),
                lr=self.kernel_input.optim_params['lr'],
                momentum=self.kernel_input.optim_params['momentum'],
                dampening=self.kernel_input.optim_params['dampening'],
                weight_decay=self.kernel_input.optim_params['weight_decay'],
                nesterov=self.kernel_input.optim_params['nesterov'])

        if solver == 'adam':
            optimizer_kernel = optim.Adam(
                self.kernel_input.model.parameters(),
                lr=self.kernel_input.optim_params['lr'],
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)

        def closure_kernel():
            loss = self.objective(x_batch, y_batch)
            optimizer_kernel.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            # loss.backward(retain_graph=True)
            loss.backward()
            return (loss)

        loss = closure_kernel()
        self.kernel_input.append_time(time() - t0)
        optimizer_kernel.step(closure_kernel)

        mse_train = loss - self.lbda * self.iokr.get_vv_norm()

        return mse_train.item()

    def fit_kernel_input(self, x_train, y_train, x_test, y_test, n_epochs=50, solver='sgd', batch_size_train=64,
                         batch_size_test=None, verbose=True):
        """
        Fits the input kernel when using a learnable neural network kernel input using the method train_kernel_input
        at each epoch.
        """
        if solver not in ('sgd', 'adam'):
            raise ValueError(f"'solver' should be 'sgd' or 'adam',but not {solver}")
        if type(n_epochs) != int:
            raise TypeError(f"'n_epochs' should be an int, not a {type(n_epochs)}")
        if type(batch_size_train) != int:
            raise TypeError(f"'batch_size_train' should be an int, not a {type(batch_size_train)}")

        self.verbose = verbose

        if batch_size_train is None:
            batch_size_train = x_train.shape[0]

        if not hasattr(self.kernel_input, 'train_losses'):
            self.kernel_input.train_losses = []
            self.kernel_input.times = [0]

        if not hasattr(self.kernel_input, 'test_mse'):
            self.kernel_input.test_mse = []

        if not hasattr(self.kernel_input, 'test_f1'):
            self.kernel_input.test_f1 = []

        if batch_size_test is None:
            batch_size_test = x_test.shape[0]

        self.x_train = x_train
        self.y_train = y_train

        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test)

        t0 = time()

        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train)
        self.loader_train = loader_train

        for epoch in range(n_epochs):

            batch_losses_train = []

            len_trained = 0

            for batch_idx_train, (data_train, target_train) in enumerate(loader_train):

                batch_losses_train.append(self.train_kernel_input(data_train, target_train, solver, t0))

                len_trained += data_train.shape[0]

                if verbose:

                    if batch_idx_train % 10 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, len_trained, x_train.size()[0],
                            100. * len_trained / x_train.size()[0], batch_losses_train[-1]))

                batch_idx_train += 1

            self.kernel_input.train_losses.append(np.mean(np.asarray(batch_losses_train)))

            # self.iokr.fit(X_s=x_train, Y_s=y_train, L=self.lbda, input_kernel=self.kernel_input,
            #              output_kernel=self.kernel_output, input_gamma=self.kernel_input.gamma)

            batch_losses_test = []

            K_y = self.kernel_output.compute_gram(y_train, y_train)

            Omega_block_diag = np.empty((0, 0))

            for batch_idx_train, (data_train, target_train) in enumerate(loader_train):
                n_ba = data_train.shape[0]

                Omega = torch.inverse(
                    self.kernel_input.compute_gram(data_train) + n_ba * self.lbda * torch.eye(n_ba)).data.numpy()

                Omega_block_diag = block_diag(Omega_block_diag, Omega)

            Omega_block_diag = torch.from_numpy(Omega_block_diag).float()

            print(Omega_block_diag.shape)

            n_b = len(loader_train)

            for batch_idx_test, (data_test, target_test) in enumerate(loader_test):
                K_x_tr_te = self.kernel_input.compute_gram(x_train, data_test)
                K_y_tr_te = self.kernel_output.compute_gram(y_train, target_test)
                K_y_te_te = self.kernel_output.compute_gram(target_test)

                mse_test, _ = self.cost(Omega_block_diag, K_x_tr_te, K_y_tr_te, K_y_te_te, K_y, n_b)

                batch_losses_test.append(mse_test.item())

            self.kernel_input.test_mse.append(np.mean(batch_losses_test))

            if verbose:
                print('\nTest MSE of the whole model: {:.4f}\n'.format(mse_test))

            """
            if epoch % 25 == 0:
                
                K_x_tr_te = self.kernel_input.compute_gram(x_train, x_test)                    
                
                structured_loss, _ = F1_score(y_train, K_y, K_x_tr_te, y_test, y_train, self.kernel_output, Omega_block_diag, n_b)
                
                self.kernel_input.test_f1.append(structured_loss)
                
                if verbose:
                    print('\nTest F1 score of the whole model: {:.4f}\n'.format(structured_loss))
            """

    def predict(self, x_test, Y_candidates=None):
        """
        Model Prediction
        """
        if Y_candidates is None:
            Y_candidates = self.y_train

        t0 = time()

        K_y = self.kernel_output.compute_gram(self.y_train)

        Omega_block_diag = np.empty((0, 0))

        for batch_idx_train, (data_train, target_train) in enumerate(self.loader_train):
            n_ba = data_train.shape[0]

            Omega = torch.inverse(self.kernel_input.compute_gram(data_train)
                                  + n_ba * self.lbda * torch.eye(n_ba)).data.numpy()

            Omega_block_diag = block_diag(Omega_block_diag, Omega)

        Omega_block_diag = torch.from_numpy(Omega_block_diag).float()

        n_b = len(self.loader_train)

        K_x_tr_te = self.kernel_input.compute_gram(self.x_train, x_test)

        A = K_x_tr_te.T @ Omega_block_diag

        prod_h_c = A @ self.kernel_output.compute_gram(self.y_train, Y_candidates)
        norm_h = torch.diag(A @ K_y @ A.T)
        norm_c = torch.diag(self.kernel_output.compute_gram(Y_candidates))

        se = (1.0 / n_b ** 2) * norm_h.view(-1, 1) - (2.0 / n_b) * prod_h_c + norm_c.view(1, -1)
        scores = - se

        # Predict and compute Structured losses
        idx_pred = torch.argmax(scores, axis=1)
        Y_pred = Y_candidates[idx_pred]
        if self.verbose > 0:
            print(f'Decoding time: {time() - t0} in s')

        return Y_pred
