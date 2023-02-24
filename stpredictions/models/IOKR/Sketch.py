import numpy as np


class Sketch:
    """
    Class of sketch matrices
    """

    def __init__(self, size):
        """
        Initialise a sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.
        """
        self.size = size


class SubSample(Sketch):
    """
    Class of sub-sampling sketch matrices
    """

    def __init__(self, size, probs=None, replace=False):
        """
        Initialise a sub-sampling sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.

        probs: 1-D array-like of floats, optionnal
        Probabilies of sampling. Default is None, leading to Uniform sampling.

        replace: boolean, optionnal
        With or without replacement. Default is False, i.e. without replacement.
        """
        super(SubSample, self).__init__(size)
        self.indices = np.random.choice(self.size[1], self.size[0], replace=replace, p=probs)
        if probs is None:
            self.probs = (1.0 / self.size[1]) * np.ones(self.size[1])
        else:
            self.probs = probs


    def multiply_vector(self, x):
        """
        Multiply sketch matrix with vector x

        Parameters
        ----------
        x: 1-D array-like of size self.size[1]
        Vector to compute multiplication with.

        Returns
        -------
        res: 1-D array-like of size self.size[0]
        S.dot(x).
        """
        res = np.sqrt(1.0 / self.size[0]) * x[self.indices]
        res *= (1.0 / np.sqrt(self.probs[self.indices]))
        return res

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) if right.
        S.dot(K) otherwise.
        """
        if Y is None:
            Y = X.copy()

        if right:
            Y_sampled = Y[self.indices]
            res = np.sqrt(1.0 / self.size[0]) * kernel(X, Y_sampled)
            res *= (1.0 / np.sqrt(self.probs[self.indices]))
            return res

        else:
            X_sampled = X[self.indices]
            res = np.sqrt(1.0 / self.size[0]) * kernel(X_sampled, Y)
            res *= (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
            return res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = np.sqrt(1.0 / self.size[0]) * M[:, self.indices]
            res *= (1.0 / np.sqrt(self.probs[self.indices]))
            return res

        else:
            res = np.sqrt(1.0 / self.size[0]) * M[self.indices, :]
            res *= (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
            return res

    
    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        X_sampled = X[self.indices]
        res = (1.0 / self.size[0]) * kernel(X_sampled, X_sampled)
        res *= (1.0 / np.sqrt(self.probs[self.indices]))
        res *= (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
        return res


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = (1.0 / self.size[0]) * M[self.indices, self.indices]
        res *= (1.0 / np.sqrt(self.probs[self.indices]))
        res *= (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
        return res


class SubSampleRad(Sketch):
    """
    Class of sub-sampling with Rademacher variables on each line sketch matrices
    """

    def __init__(self, size, probs=None, replace=True):
        """
        Initialise a sub-sampling Rademacher sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.

        probs: 1-D array-like of floats, optionnal
        Probabilies of sampling. Default is None, leading to Uniform sampling.

        replace: boolean, optionnal
        With or without replacement. Default is True, i.e. with replacement.
        """
        super(SubSampleRad, self).__init__(size)
        self.indices = np.random.choice(self.size[1], self.size[0], replace=replace, p=probs)
        if probs is None:
            self.probs = (1.0 / self.size[1]) * np.ones(self.size[1])
        else:
            self.probs = probs
        self.rad = 2 * np.random.binomial(1, 0.5, self.size[0]) - 1


    def multiply_vector(self, x):
        """
        Multiply sketch matrix with vector x

        Parameters
        ----------
        x: 1-D array-like of size self.size[1]
        Vector to compute multiplication with.

        Returns
        -------
        res: 1-D array-like of size self.size[0]
        S.dot(x).
        """
        res = np.sqrt(1.0 / self.size[0]) * x[self.indices]
        res *= self.rad * (1.0 / np.sqrt(self.probs[self.indices]))
        return res

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) if right.
        S.dot(K) otherwise.
        """
        if Y is None:
            Y = X.copy()

        if right:
            Y_sampled = Y[self.indices]
            res = np.sqrt(1.0 / self.size[0]) * kernel(X, Y_sampled)
            res *= self.rad * (1.0 / np.sqrt(self.probs[self.indices]))
            return res

        else:
            X_sampled = X[self.indices]
            res = np.sqrt(1.0 / self.size[0]) * kernel(X_sampled, Y)
            res *= np.reshape(self.rad, (self.size[0], -1)) * (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
            return res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = np.sqrt(1.0 / self.size[0]) * M[:, self.indices]
            res *= self.rad * (1.0 / np.sqrt(self.probs[self.indices]))
            return res

        else:
            res = np.sqrt(1.0 / self.size[0]) * M[self.indices, :]
            res *= np.reshape(self.rad, (self.size[0], -1)) * (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
            return res

    
    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        X_sampled = X[self.indices]
        res = (1.0 / self.size[0]) * kernel(X_sampled, X_sampled)
        res *= self.rad * (1.0 / np.sqrt(self.probs[self.indices]))
        res *= np.reshape(self.rad, (self.size[0], -1)) * (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
        return res


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = (1.0 / self.size[0]) * M[self.indices, self.indices]
        res *= self.rad * (1.0 / np.sqrt(self.probs[self.indices]))
        res *= np.reshape(self.rad, (self.size[0], -1)) * (1.0 / np.sqrt(np.reshape(self.probs[self.indices], (self.size[0], -1))))
        return res


class Accumulation(Sketch):
    """
    Class of accumulation of Sub-Sample Rademacher sketch matrices
    """
    def __init__(self, size, m, probs=None, replace=True):
        """
        Initialise a sub-sampling Rademacher sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.

        probs: 1-D array-like of floats, optionnal
        Probabilies of sampling. Default is None, leading to Uniform sampling.

        replace: boolean, optionnal
        With or without replacement. Default is True, i.e. with replacement.
        """
        super(Accumulation, self).__init__(size)
        self.m = m
        self.sketches = []
        for i in range(m):
            self.sketches.append(SubSampleRad(size, probs, replace))

    
    def multiply_vector(self, x):
            """
            Multiply sketch matrix with vector x

            Parameters
            ----------
            x: 1-D array-like of size self.size[1]
            Vector to compute multiplication with.

            Returns
            -------
            res: 1-D array-like of size self.size[0]
            S.dot(x).
            """
            res = np.zeros(self.size[0])
            for k in range(self.m):
                res += self.sketches[k].multiply_vector(x)
            res /= np.sqrt(self.m)
            return res

        
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) if right.
        S.dot(K) otherwise.
        """
        if Y is None:
            Y = X.copy()
        
        if right:
            res = np.zeros((X.shape[0], self.size[0]))
            for k in range(self.m):
                res += self.sketches[k].multiply_Gram_one_side(X, kernel, Y, right)
            res /= np.sqrt(self.m)
            return res

        else:
            res = np.zeros((self.size[0], Y.shape[0]))
            for k in range(self.m):
                res += self.sketches[k].multiply_Gram_one_side(X, kernel, Y, right)
            res /= np.sqrt(self.m)
            return res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = np.zeros((M.shape[0], self.size[0]))
            for k in range(self.m):
                res += self.sketches[k].multiply_matrix_one_side(M, right)
            res /= np.sqrt(self.m)
            return res

        else:
            res = np.zeros((self.size[0], M.shape[1]))
            for k in range(self.m):
                res += self.sketches[k].multiply_matrix_one_side(M, right)
            res /= np.sqrt(self.m)
            return res

    
    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        res = np.zeros((self.size[0], self.size[0]))
        for k in range(self.m):
            for l in range(self.m):
                X_sampled_left = X[self.sketches[k].indices]
                X_sampled_right = X[self.sketches[l].indices]
                res_temp = (1.0 / self.size[0]) * kernel(X_sampled_left, X_sampled_right)
                res_temp *= np.reshape(self.sketches[k].rad, (self.size[0], -1)) * (1.0 / np.sqrt(np.reshape(self.sketches[k].probs[self.sketches[k].indices], (self.size[0], -1))))
                res_temp *= self.sketches[l].rad * (1.0 / np.sqrt(self.sketches[l].probs[self.sketches[l].indices]))
                res += res_temp
        res /= self.m
        return res


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = np.zeros((self.size[0], self.size[0]))
        for k in range(self.m):
            for l in range(self.m):
                res_temp = (1.0 / self.size[0]) * M[self.sketches[k].indices, self.sketches[l].indices]
                res_temp *= np.reshape(self.sketches[k].rad, (self.size[0], -1)) * (1.0 / np.sqrt(np.reshape(self.sketches[k].probs[self.sketches[k].indices], (self.size[0], -1))))
                res_temp *= self.sketches[l].rad * (1.0 / np.sqrt(self.sketches[l].probs[self.sketches[l].indices]))
                res += res_temp
        res /= self.m
        return res


class SJLT(Sketch):
    """
    Class of Sparse Johnson-Lindenstrauss Transform sketch matrices
    """

    def __init__(self, size, m=1):
        """
        Initialise a sub-sampling sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.

        m: int
        Number of non-zero elements in each column
        """
        super(SJLT, self).__init__(size)
        s, n = size[0], size[1]
        S = np.empty((0, n))
        ss = int(s / m)
        for i in range(m - 1):
            idx0 = np.random.choice(ss, n)
            idx1 = np.arange(n)
            coefs = (2 * np.random.binomial(1, 0.5, n) - 1)
            S_i = np.zeros((ss, n), dtype=float)
            S_i[idx0, idx1] = coefs
            S = np.vstack((S, S_i))
        r = s % m
        idx0 = np.random.choice(ss + r, n)
        idx1 = np.arange(n)
        coefs = (2 * np.random.binomial(1, 0.5, n) - 1)
        S_i = np.zeros((ss + r, n), dtype=float)
        S_i[idx0, idx1] = coefs
        S = np.vstack((S, S_i))
        S *= 1.0 / np.sqrt(m)
        self.S = S.copy()


    def multiply_vector(self, x):
            """
            Multiply sketch matrix with vector x

            Parameters
            ----------
            x: 1-D array-like of size self.size[1]
            Vector to compute multiplication with.

            Returns
            -------
            res: 1-D array-like of size self.size[0]
            S.dot(x).
            """
            return self.S.dot(x)

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) of shape (self.size[1], self.size[0]) if right.
        S.dot(K) of shape (self.size[0], self.size[1]) otherwise.
        """
        if Y is None:
            Y = X.copy()

        if right:
            K = kernel(X, Y)
            res = K.dot(self.S.T)
            """
            res = np.zeros((n, self.size[0]))
            for i in range(Y.shape[0]):
                col = kernel(X, Y[i].reshape(1, -1))
                line = self.S[:, i]
                res += np.reshape(col, (n, 1)).dot(np.reshape(line, (1, self.size[0])))
            """
            return res

        else:
            K = kernel(X, Y)
            res = self.S.dot(K)
            """
            res = np.zeros((self.size[0], n))
            for i in range(X.shape[0]):
                col = self.S[:, i]
                line = kernel(X[i].reshape(1, -1), Y)
                res += np.reshape(col, (self.size[0], 1)).dot(np.reshape(line, (1, n)))
            """
            return res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = M.dot(self.S.T)
            return res

        else:
            res = self.S.dot(M)
            return res


    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        res = self.multiply_Gram_one_side(X, kernel, right=True)
        return self.S.dot(res)


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = self.multiply_matrix_one_side(M, right=True)
        return self.S.dot(res)


class Rademacher(Sketch):
    """
    Class of Rademacher sketch matrices
    """

    def __init__(self, size):
        """
        Initialise a sub-sampling sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.
        """
        super(Rademacher, self).__init__(size)
        self.S = (1 / np.sqrt(size[0])) * (2 * np.random.binomial(1, 0.5, size) - 1)


    def multiply_vector(self, x):
            """
            Multiply sketch matrix with vector x

            Parameters
            ----------
            x: 1-D array-like of size self.size[1]
            Vector to compute multiplication with.

            Returns
            -------
            res: 1-D array-like of size self.size[0]
            S.dot(x).
            """
            return self.S.dot(x)

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) of shape (self.size[1], self.size[0]) if right.
        S.dot(K) of shape (self.size[0], self.size[1]) otherwise.
        """
        if Y is None:
            Y = X.copy()

        if right:
            K = kernel(X, Y)
            res = K.dot(self.S.T)
            """
            res = np.zeros((n, self.size[0]))
            for i in range(Y.shape[0]):
                col = kernel(X, Y[i].reshape(1, -1))
                line = self.S[:, i]
                res += np.reshape(col, (n, 1)).dot(np.reshape(line, (1, self.size[0])))
            """
            return res

        else:
            K = kernel(X, Y)
            res = self.S.dot(K)
            """
            res = np.zeros((self.size[0], n))
            for i in range(X.shape[0]):
                col = self.S[:, i]
                line = kernel(X[i].reshape(1, -1), Y)
                res += np.reshape(col, (self.size[0], 1)).dot(np.reshape(line, (1, n)))
            """
            return res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = M.dot(self.S.T)
            return res

        else:
            res = self.S.dot(M)
            return res


    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        res = self.multiply_Gram_one_side(X, kernel, right=True)
        return self.S.dot(res)


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = self.multiply_matrix_one_side(M, right=True)
        return self.S.dot(res)


class Gaussian(Sketch):
    """
    Class of Gaussian sketch matrices
    """

    def __init__(self, size):
        """
        Initialise a sub-sampling sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.
        """
        super(Gaussian, self).__init__(size)
        self.S = (1 / np.sqrt(size[0])) * np.random.normal(size=self.size)


    def multiply_vector(self, x):
            """
            Multiply sketch matrix with vector x

            Parameters
            ----------
            x: 1-D array-like of size self.size[1]
            Vector to compute multiplication with.

            Returns
            -------
            res: 1-D array-like of size self.size[0]
            S.dot(x).
            """
            return self.S.dot(x)

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) of shape (self.size[1], self.size[0]) if right.
        S.dot(K) of shape (self.size[0], self.size[1]) otherwise.
        """
        if Y is None:
            Y = X.copy()

        if right:
            K = kernel(X, Y)
            res = K.dot(self.S.T)
            """
            res = np.zeros((n, self.size[0]))
            for i in range(Y.shape[0]):
                col = kernel(X, Y[i].reshape(1, -1))
                line = self.S[:, i]
                res += np.reshape(col, (n, 1)).dot(np.reshape(line, (1, self.size[0])))
            """
            return res

        else:
            K = kernel(X, Y)
            res = self.S.dot(K)
            """
            res = np.zeros((self.size[0], n))
            for i in range(X.shape[0]):
                col = self.S[:, i]
                line = kernel(X[i].reshape(1, -1), Y)
                res += np.reshape(col, (self.size[0], 1)).dot(np.reshape(line, (1, n)))
            """
            return res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = M.dot(self.S.T)
            return res

        else:
            res = self.S.dot(M)
            return res


    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        res = self.multiply_Gram_one_side(X, kernel, right=True)
        return self.S.dot(res)


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = self.multiply_matrix_one_side(M, right=True)
        return self.S.dot(res)


class pSparsified(Sketch):
    """
    Class of Sp-Sparsified sketches implemented as product of Sub-Gaussian matrix and Sub-Sampling matrix
    """
    
    def __init__(self, size, p=None, type='Gaussian'):
        """
        Initialise a sub-sampling sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.

        p: float, optionnal
        Probability for an entry of the sketch matrix to being non-null.
        Default is 1/size[1].

        type: str, optionnal
        Type of the p-Sparse sketch matrix, either 'Gaussian' or 'Rademacher'.
        Default is 'Gaussian'
        """
        super(pSparsified, self).__init__(size)
        if p is None:
            p = 20 / self.size[1]
        self.p = p
        self.type = type
        B = np.random.binomial(1, self.p, self.size)
        idx1 = np.where(B!=0)[1]
        idx = np.argwhere(np.all(B[..., :] == 0, axis=0))
        B1 = np.delete(B, idx, axis=1)
        B1 = B1.astype(float)
        if type == 'Gaussian':
            self.SG = np.random.normal(size=B1.shape) * B1.copy()
        else:
            self.SG = (2 * np.random.binomial(1, 0.5, B1.shape) - 1) * B1.copy()
        self.indices = np.unique(idx1)


    def multiply_vector(self, x):
            """
            Multiply sketch matrix with vector x

            Parameters
            ----------
            x: 1-D array-like of size self.size[1]
            Vector to compute multiplication with.

            Returns
            -------
            res: 1-D array-like of size self.size[0]
            S.dot(x).
            """
            res = self.SG * x[self.indices]
            return (1 / np.sqrt(self.size[0] * self.p)) * res

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) of shape (self.size[1], self.size[0]) if right.
        S.dot(K) of shape (self.size[0], self.size[1]) otherwise.
        """
        if Y is None:
            Y = X.copy()
        
        if right:
            Y_sampled = Y[self.indices]
            res = kernel(X, Y_sampled).dot(self.SG.T)
            return (1 / np.sqrt(self.size[0] * self.p)) * res

        else:
            X_sampled = X[self.indices]
            res = self.SG.dot(kernel(X_sampled, Y))
            return (1 / np.sqrt(self.size[0] * self.p)) * res


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        if right:
            res = M[:, self.indices].dot(self.SG.T)
            return (1 / np.sqrt(self.size[0] * self.p)) * res

        else:
            res = self.SG.dot(M[self.indices, :])
            return (1 / np.sqrt(self.size[0] * self.p)) * res


    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        X_sampled = X[self.indices]
        res = self.SG.dot(kernel(X_sampled, X_sampled)).dot(self.SG.T)
        return (1 / self.size[0] * self.p) * res


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        res = self.SG.dot(M[np.ix_(self.indices, self.indices)]).dot(self.SG.T)
        return (1 / self.size[0] * self.p) * res


class Incomplete(Sketch):
    """
    Class of Incomplete Rademacher or Gaussian sketch matrices
    """
    
    def __init__(self, size, s2, type='Gaussian', probs=None):
        """
        Initialise a sub-sampling sketch matrix

        Parameters
        ----------
        size: tuple of ints
        Sketch matrix shape.

        s2: int
        Size of Sub-Sampling matrix

        probs: 1-D array-like of floats, optionnal
        Probabilies of sampling. Default is None, leading to Uniform sampling.
        """
        super(Incomplete, self).__init__(size)
        if type == 'Rademacher':
            self.S1 = Rademacher((size[0], s2))
        else:
            self.S1 = Gaussian((size[0], s2))
        self.S2 = []
        self.S2 = SubSample((s2, size[1]), probs=probs, replace=False)


    def multiply_vector(self, x):
            """
            Multiply sketch matrix with vector x

            Parameters
            ----------
            x: 1-D array-like of size self.size[1]
            Vector to compute multiplication with.

            Returns
            -------
            res: 1-D array-like of size self.size[0]
            S.dot(x).
            """
            s2 = self.S2.size[0]
            res = np.zeros(s2)
            res = self.S2.multiply_vector(x)
            return self.S1.multiply_vector(res)

    
    def multiply_Gram_one_side(self, X, kernel, Y=None, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        right: boolean, optionnal.
        If True, computation of K.dot(S.T) is performed.
        Else, S.dot(K).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) of shape (self.size[1], self.size[0]) if right.
        S.dot(K) of shape (self.size[0], self.size[1]) otherwise.
        """
        if Y is None:
            Y = X.copy()
        
        s2 = self.S2.size[0]

        if right:
            res = np.zeros((X.shape[0], s2))
            res = self.S2.multiply_Gram_one_side(X, kernel, Y, right)
            return self.S1.multiply_matrix_one_side(res, right)

        else:
            res = np.zeros((s2, Y.shape[0]))
            res = self.S2.multiply_Gram_one_side(X, kernel, Y, right)
            return self.S1.multiply_matrix_one_side(res, right)


    def multiply_matrix_one_side(self, M, right=True):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        right: boolean, optionnal.
        If True, computation of M.dot(S.T) is performed.
        Else, S.dot(M).
        Default is True.

        Returns
        -------
        res: 2-D array-like
        M.dot(S.T) of shape (M.shape[0], self.size[0]) if right.
        S.dot(M) of shape (self.size[0], M.shape[1]) otherwise.
        """
        s2 = self.S2.size[0]

        if right:
            res = np.zeros((M.shape[0], s2))
            res = self.S2.multiply_matrix_one_side(M, right)
            return self.S1.multiply_matrix_one_side(res, right)

        else:
            res = np.zeros((s2, M.shape[1]))
            res = self.S2.multiply_matrix_one_side(M, right)
            return self.S1.multiply_matrix_one_side(res, right)


    def multiply_Gram_both_sides(self, X, kernel):
        """
        Multiply on both sides sketch matrix with Gram matrix formed with X and a kernel

        Parameters
        ----------
        X: 2-D array-like of shape (self.size[1], n_features)
        Inputs on which Gram matrix is computed

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K with inputs X.

        Returns
        -------
        res: 2-D array-like of shape (self.size[0], self.size[0])
        S.dot(K.dot(S.T)).
        """
        s2 = self.S2.size[0]
        res = np.zeros((s2, s2))
        X_sampled_left = X[self.S2.indices]
        X_sampled_right = X[self.S2.indices]
        res_temp = (1.0 / s2) * kernel(X_sampled_left, X_sampled_right)
        res_temp *= (1.0 / np.sqrt(np.reshape(self.S2.probs[self.S2.indices], (s2, -1))))
        res_temp *= (1.0 / np.sqrt(self.S2.probs[self.S2.indices]))
        res = res_temp.copy()
        return self.S1.multiply_matrix_both_sides(res)


    def multiply_matrix_both_sides(self, M):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        M: 2-D array-like
        Matrix which is multiplied by S.

        Returns
        -------
        res: 2-D array-like
        S.dot(M.dot(S.T)) of shape (self.size[0], self.size[0]).
        """
        s2 = self.S2.size[0]
        res = np.zeros((s2, s2))
        res_temp = (1.0 / s2) * M[self.S2.indices, self.S2.indices]
        res_temp *= (1.0 / np.sqrt(self.S2.probs[self.S2.indices]).reshape((-1, 1)))
        res_temp *= (1.0 / np.sqrt(self.S2.probs[self.S2.indices]))
        res = res_temp.copy()
        return self.S1.multiply_matrix_both_sides(res)