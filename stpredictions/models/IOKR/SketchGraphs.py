import numpy as np


class SketchGraphs:
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


class SubSampleGraphs(SketchGraphs):
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
        super(SubSampleGraphs, self).__init__(size)
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

    
    def multiply_Gram_one_side(self, X, kernel, Y=None):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: GraphKernel.
        Compute Gram matrix K between X and Y.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) if right.
        S.dot(K) otherwise.
        """
        if Y is None:
            Y = X.copy()
            
        X_sampled = X[self.indices]
        kernel.fit(X_sampled)

        res = np.sqrt(1.0 / self.size[0]) * kernel.transform(Y)
        res *= (1.0 / np.sqrt(self.probs[self.indices]))

        return res
    

    def multiply_Gram_one_side_sum_kernel(self, X, kernel, Xgraph, kernelGraph, Y=None, Ygraph=None, t=0.5):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and the product
        kernel of a kernel on vector and a kernel on graphs.

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        X: 1-D array-like
        First input on which Gram matrix is computed

        Y: 1-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        kernel: GraphKernel.
        Compute Gram matrix K between X and Y.

        t: float, optionnal.
        Value between 0 and 1. Default is 0.5.


        Returns
        -------
        res: 2-D array-like
        K.dot(S.T).
        """
        if Y is None:
            Y = X.copy()

        if Ygraph is None:
            Ygraph = Xgraph.copy()

        X_sampled = X[self.indices]
            
        Xgraph_sampled = Xgraph[self.indices]
        kernelGraph.fit(Xgraph_sampled)

        res = np.sqrt(1.0 / self.size[0]) * (t * kernel(Y, X_sampled) + (1 - t) * kernelGraph.transform(Ygraph))
        res *= (1.0 / np.sqrt(self.probs[self.indices]))

        return res
    

    def multiply_Gram_one_side_prod_kernel(self, X, kernel, Xgraph, kernelGraph, Y=None, Ygraph=None):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and the product
        kernel of a kernel on vector and a kernel on graphs.

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        X: 1-D array-like
        First input on which Gram matrix is computed

        Y: 1-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        kernel: GraphKernel.
        Compute Gram matrix K between X and Y.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T).
        """
        if Y is None:
            Y = X.copy()

        if Ygraph is None:
            Ygraph = Xgraph.copy()

        X_sampled = X[self.indices]
            
        Xgraph_sampled = Xgraph[self.indices]
        kernelGraph.fit(Xgraph_sampled)

        res = np.sqrt(1.0 / self.size[0]) * kernel(Y, X_sampled) * kernelGraph.transform(Ygraph)
        res *= (1.0 / np.sqrt(self.probs[self.indices]))

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


class pSparsifiedGraphs(SketchGraphs):
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
        super(pSparsifiedGraphs, self).__init__(size)
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

    
    def multiply_Gram_one_side(self, X, kernel, Y=None):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and a kernel

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: GraphKernel.
        Compute Gram matrix K between X and Y.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) of shape (self.size[1], self.size[0]) if right.
        S.dot(K) of shape (self.size[0], self.size[1]) otherwise.
        """
        if Y is None:
            Y = X.copy()
            
        X_sampled = X[self.indices]
        kernel.fit(X_sampled)
        
        res = kernel.transform(Y).dot(self.SG.T)
        res = (1 / np.sqrt(self.size[0] * self.p)) * res
        
        return res
    

    def multiply_Gram_one_side_sum_kernel(self, X, kernel, Xgraph, kernelGraph, Y=None, Ygraph=None, t=0.5):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and the product
        kernel of a kernel on vector and a kernel on graphs.

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        X: 1-D array-like
        First input on which Gram matrix is computed

        Y: 1-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        kernel: GraphKernel.
        Compute Gram matrix K between X and Y.

        t: float.
        Value between 0 and 1. Default is 0.5.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) if right.
        """
        if Y is None:
            Y = X.copy()

        if Ygraph is None:
            Ygraph = Xgraph.copy()

        X_sampled = X[self.indices]
            
        Xgraph_sampled = Xgraph[self.indices]
        kernelGraph.fit(Xgraph_sampled)

        res = (t * kernel(Y, X_sampled) + (1 - t) * kernelGraph.transform(Ygraph)).dot(self.SG.T)
        res *= 1 / np.sqrt(self.size[0] * self.p)

        return res


    def multiply_Gram_one_side_prod_kernel(self, X, kernel, Xgraph, kernelGraph, Y=None, Ygraph=None):
        """
        Multiply sketch matrix with Gram matrix formed with X and Y and the product
        kernel of a kernel on vector and a kernel on graphs.

        Parameters
        ----------
        X: 2-D array-like
        First input on which Gram matrix is computed

        Y: 2-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        X: 1-D array-like
        First input on which Gram matrix is computed

        Y: 1-D array-like, optionnal.
        Second input on which Gram matrix is computed. Default is None,
        in this case Y=X.

        kernel: function of 2 2-D array-like variables.
        Compute Gram matrix K between X and Y.

        kernel: GraphKernel.
        Compute Gram matrix K between X and Y.

        Returns
        -------
        res: 2-D array-like
        K.dot(S.T) if right.
        """
        if Y is None:
            Y = X.copy()

        if Ygraph is None:
            Ygraph = Xgraph.copy()

        X_sampled = X[self.indices]
            
        Xgraph_sampled = Xgraph[self.indices]
        kernelGraph.fit(Xgraph_sampled)

        res = (kernel(Y, X_sampled) * kernelGraph.transform(Ygraph)).dot(self.SG.T)
        res *= 1 / np.sqrt(self.size[0] * self.p)

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
            res = M[:, self.indices].dot(self.SG.T)
            return (1 / np.sqrt(self.size[0] * self.p)) * res

        else:
            res = self.SG.dot(M[self.indices, :])
            return (1 / np.sqrt(self.size[0] * self.p)) * res


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