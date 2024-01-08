"""
This module implement the OK3 method. 
Handles the problem of learning functions with a structured output space, 
represented by vectors but whiwh is often not a vectorial space.
"""

import numbers
import warnings
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import time

import numpy as np
from scipy.sparse import issparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import r2_score

from sklearn.base import BaseEstimator
from sklearn.base import clone

from .base import StructuredOutputMixin

from sklearn.utils import Bunch
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from ._criterion import Criterion
from ._splitter import Splitter
from .kernel import Kernel
from ._tree import DepthFirstTreeBuilder
from ._tree import BestFirstTreeBuilder
from ._tree import Tree
from ._tree import _build_pruned_tree_ccp
from ._tree import ccp_pruning_path

from ._tree import DTYPE, DOUBLE
from .kernel import Gini_Kernel, MSE_Kernel, Mean_Dirac_Kernel, Linear_Kernel, Laplacian_Kernel, Gaussian_Kernel
from ._criterion import KernelizedMSE
from ._splitter import BestSplitter, RandomSplitter, BestSparseSplitter, RandomSparseSplitter

__all__ = ["OK3Regressor", "ExtraOK3Regressor"]


# =============================================================================
# Types and constants
# =============================================================================

#DTYPE = _tree.DTYPE
#DOUBLE = _tree.DOUBLE

# The criteria is the loss function (in the embedding Hilbert space) the tree 
# wants to minimise. Here we've implemented the classic variance reduction, called here "mse".
#CRITERIA = {"mse": _criterion.KernelizedMSE}
CRITERIA = {"mse": KernelizedMSE}

# This is the different types of kernels which can be used to compute similarities between vectorial representations of the outputs.
# Each one of them corresponds to a different embedding in an Hilbert space.
# There is two particular kernels : "gini_clf" and "mse_reg" which actually correspond 
# to the "mean-dirac" kernel and the linear kernel but by specifying that we want 
# (for a classification or a regression task) to perform an exact search of the output 
# instead of an approximate minimisation of the criterion among  alist of candidates.
KERNELS = {"gini_clf": Gini_Kernel, 
           "mse_reg": MSE_Kernel,
           "mean_dirac": Mean_Dirac_Kernel,
           "linear": Linear_Kernel,
           "laplacian": Laplacian_Kernel,
           "gaussian": Gaussian_Kernel}

# Les splitters sont des classes définissant les stratégies de recherche des splits (feature+threshold)
DENSE_SPLITTERS = {"best": BestSplitter,
                   "random": RandomSplitter}

SPARSE_SPLITTERS = {"best": BestSparseSplitter,
                    "random": RandomSparseSplitter}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseKernelizedOutputTree(StructuredOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for regression trees with a kernel in the output space.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, *,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 max_leaf_nodes,
                 random_state,
                 min_impurity_decrease,
                 min_impurity_split,
                 ccp_alpha=0.0, 
                 kernel):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.kernel = kernel
        self.leaves_preds = None

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves
    
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated", kernel=None, in_ensemble=False, Gram_y=None):
        """
        kernel : Optional input. If not provided, the kernel attribute of the
                 estimatoris used. If provided, the kernel attribute of the estimator 
                 is updated.
                 
                 Some possible values :
            
                "gini_clf" : y is a matrix of labels for multilabel classification.
                        shape (n_train_samples, n_labels)
                        We have to compute the corresponding gram matrix, 
                        equivalent to the use of a Classification Tree with the gini
                        index as impurity. Exact solution search is performed.
                 "mse_reg" : y is a matrix of real vectors for multiouput regression.
                        shape (n_train_samples, n_outputs)
                        We have to compute the corresponding gram matrix, 
                        equivalent to the use of a Regression Tree with the mse
                        as impurity. Exact solution search is performed.
                 "mean_dirac" : y is a matrix or vectors (vectorial representation of structured objects).
                        shape (n_train_samples, vector_length)
                        The similarity between two objects is computed with a mean dirac equality kernel.
                 "linear" : y is a matrix or vectors (vectorial representation of structured objects).
                        shape (n_train_samples, vector_length)
                        The similarity between two objects is computed with a linear kernel.
                 "gaussian" : y is a matrix or vectors (vectorial representation of structured objects).
                        shape (n_train_samples, vector_length)
                        The similarity between two objects is computed with a gaussian kernel.
        
        in_ensemble : boolean, default=False
                flag to set to true when the estimator is used with an ensemble method,
                if True, the Gram matrix of the outputs is also given (as K_y) and doesn't have to
                be computed by the tree --> avoid this heavy calculation for all trees.
        
        Gram_y : the output gram matrix, default=None
                Allows to avoid the Gram matrix calculation if we already have it (useful when in_ensemble=True)
        """

        if kernel is None:
            kernel = self.kernel
        else: # on peut mettre à jour l'attribut kernel de l'estimateur à travers le fit.
            self.kernel = kernel
        
        random_state = check_random_state(self.random_state)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
 
        if not isinstance(kernel, Kernel):
            params = ()
            if isinstance(kernel, tuple):
                kernel, params = kernel[0], kernel[1:]
            try:
                kernel = KERNELS[kernel](*params)
            except KeyError:
                print("Error : 'kernel' attribute (or its first element if it is a tuple) has to be either a Kernel class or a string which is a valid key for the dict 'KERNELS'.")
                raise


        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        
        if y.shape[1] == y.shape[0]:
            warnings.warn("The target parameter is a square matrix."
                          "Are you sure this is the matrix of outputs and "
                          "not a Gram matrix ? "
                          "If it is a Gram matrix, sorry but you have to "
                          "provide the vectorialized data representation "
                          "instead and provide a 'kernel' argument which "
                          "describes how to compute the Gram matrix you want."
                          "Or you can also provided the gram as K_y in addition to y.")
                    
        if "clf" in kernel.get_name():
            check_classification_targets(y)
        
        
        if Gram_y is not None:
            # une matrice de Gram est donnée, on l'utilise
            if not in_ensemble:
                # dangereux d'accepter des matrics de Gram calculées par l'utilisateur : elles peuvent comporter des erreurs
                warnings.warn("You passed a gram matrix of the outputs as an argument and"
                              "the tree isn't in an ensemble method : please be sure it corresponds "
                              "to the outputs presented in 'y' and the kernel given in 'kernel' "
                              "because we are not going to recompute another gram matrix.")
            if Gram_y.ndim == 1:
                raise ValueError("Gram_y must be a 2d numpy array of shape "
                                 "n_samples*n_samples (Gram matrix of the "
                                 "outputs of the learning set), got a 1d array")
    
            if Gram_y.shape[1] != Gram_y.shape[0]:
                raise ValueError("Gram_y must be a 2d numpy array of shape "
                                 "n_samples*n_samples (Gram matrix of the "
                                 "outputs of the learning set)")
            
            K_y = Gram_y
            
        else:
            start_computation = time.time()
            # compute the Gram matrix of the outputs
            K_y = kernel.get_Gram_matrix(y)
            print("Time to compute the training Gram matrix : " + str(time.time() - start_computation) + " s.")
                
                
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if "clf" in kernel.get_name():
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of outputs=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        
        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn("The min_impurity_split parameter is deprecated. "
                          "Its default value has changed from 1e-7 to 0 in "
                          "version 0.23, and it will be removed in 0.25. "
                          "Use the min_impurity_decrease parameter instead.",
                          FutureWarning)

            if min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        # TODO: Remove in v0.26
        if X_idx_sorted != "deprecated":
            warnings.warn("The parameter 'X_idx_sorted' is deprecated and has "
                          "no effect. It will be removed in v0.26. You can "
                          "suppress this warning by not passing any value to "
                          "the 'X_idx_sorted' parameter.", FutureWarning)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA[self.criterion](n_samples)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        self.tree_ = Tree(self.n_features_, n_samples)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth,
                                            self.min_impurity_decrease,
                                            min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           min_impurity_split)
        
        builder.build(self.tree_, X, K_y, sample_weight)
        # store the output training examples
        self.tree_.y = y

        self._prune_tree()
        
        # reset leaves_preds because the tree changed
        self.leaves_preds = None
        
        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def get_leaves_weights(self):
        """Gives the weighted training samples in each leaf
        
        Returns
        -------
        A (n_nodes, n_training_samples) array which gives for each node (line number)
        and for each training sample its weight in the node (O if the sample doesn't fall
        in the node, and a non-negative value depending on 'sample_weight' otherwise.)
        """
        check_is_fitted(self)
        return self.tree_.value
    
    def predict_weights(self, X, check_input=True):
        """Predict the output for X as weighted combinations of training outputs
        It is kind of the representation in the Hilbert space.
        
        Returns
        -------
        A (len(X), n_training_samples) array which gives for each test example (line number)
        and for each training sample its weight in the node (O if the sample doesn't fall
        in the same leaf as the test example, and a non-negative value depending on 'sample_weight' otherwise.)
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        
        return self.tree_.predict(X)
        
    def predict(self, X, candidates=None, check_input=True, return_top_k=1):
        """Predict structured objects for X.

        The predicted structured objects based on X are returned.
        Performs an argmin research algorithm amongst the possible outputs

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        
        candidates : array of shape (nb_candidates, vectorial_repr_len), default=None
            The candidates outputs for the minimisation problem of decoding the predictions
            in the Hilbert space. 
            If not given or None, it will be set to the output training matrix.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        
        return_top_k : int, default=1
            Indicate how many decoded outputs to return for each example (or for each leaf).
            Default is one : select the output that gives the  minimum "distance" with the 
            predicted value in the Hilbert space. 
            But it is useful to be able to return for example the 5 best candidates in order 
            to evaluate a top 5 accuracy metric.

        Returns
        -------
        output : 
            array of shape (n_samples, vectorial_repr_len) containing the vectorial 
            representations of the structured output objects (found in the set of candidates, 
            or if it is not given, among the training outputs).
        """
        kernel = self.kernel
        if isinstance(kernel, Kernel):
            kernel = kernel.get_name()
        elif isinstance(kernel, tuple):
            kernel = kernel[0]
        
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA[self.criterion](X.shape[0])
        
        if "reg" in kernel and return_top_k > 1:
            warnings.warn("On ne peut pas retourner plusieurs candidats d'outputs dans le cas d'une régression, veuillez plutôt choisir kernel=linear. "
                          "return_top_k va etre mis à 1.")
            return_top_k = 1
        
        if candidates is None:

            if self.leaves_preds is None:
                if not("clf" in kernel or "reg" in kernel):
                    warnings.warn("Vous n'avez pas renseigné de candidats ni excécuté de précédentes prédictions, "
                                  "le décodage se fait donc parmi l'ensemble de sorties d'entrapinement.\n"
                                  "Si ce n'est pas ce que vous souhaitez, veuillez renseigner un ensemble de candidats de sorties afin de décoder.")
            
            elif self.leaves_preds.shape[0] != self.tree_.node_count*return_top_k: 
                # cas ou leaves_preds n'est pas None mais return_top_k a changé depuis le précédent décodage
                if not("clf" in kernel or "reg" in kernel):
                    warnings.warn("Vous n'avez pas renseigné de candidats et les précédents décodages demandaient un nombre différent de propositions de candidats ('return_top_k'), "
                                  "le décodage se fait donc parmi l'ensemble de sorties d'entrapinement.\n"
                                  "Si ce n'est pas ce que vous souhaitez, veuillez renseigner un ensemble de candidats de sorties afin de décoder.")

            else:
                # le nombre de prédictions par feuille est le meme:
                X_leaves = self.apply(X)
                # X_leaves_indices est là pour aller chercher les bons indices 
                # dans self.leaves_preds. En effet chaque noeud a return_top_k 
                # indices d'affilé dans le tableau leaves_preds, 
                # de node_id*return_top_k à node_id*return_top_k+return_top_k
                X_leaves_indices = np.zeros(X_leaves.shape[0]*return_top_k, dtype=X_leaves.dtype)
                for k in range(return_top_k):
                    X_leaves_indices[k::return_top_k] = X_leaves*return_top_k+k
                return self.leaves_preds[X_leaves_indices]

        # on calcule la totalité des sorties avec cet ensemble de candidats (et on met à jour self.leaves_preds)
        self.decode_tree(candidates, return_top_k=return_top_k)
        # on utilise ces sorties pour calculer celles demandées
        return self.predict(X, return_top_k=return_top_k)

    def decode(self, X, candidates=None, check_input=True, return_top_k=1):
        """ Synonyme de predict """
        return self.predict(X=X, candidates=candidates, check_input=check_input, return_top_k=return_top_k)

    def decode_tree(self, candidates=None, return_top_k=1):
        """Decode each leaves predictions of the tree, AND store the array of the decoded outputs
        as an attribut of the estimator : self.leaves_preds.
        
        ATTENTION, les prédictions correspondant aux noeuds qui ne sont pas des feuilles n'ont aucu  sens : elles sont arbitraires.
        Elles n'ont volontairement pas été calculées pour question d'économie de temps.

        Parameters
        ----------
        
        candidates : array of shape (nb_candidates, vectorial_repr_len), default=None
            The candidates outputs for the minimisation problem of decoding the predictions
            in the Hilbert space. 
            If not given or None, it will be set to the output training matrix.

        return_top_k : int, default=1
            Indicate how many decoded outputs to return for each example (or for each leaf).
            Default is one : select the output that gives the  minimum "distance" with the 
            predicted value in the Hilbert space. 
            But it is useful to be able to return for example the 5 best candidates in order 
            to evaluate a top 5 accuracy metric.
        
        Returns
        -------
        leaves_preds : array-like of shape (node_count,vector_length)
            For each leaf, return the vectorial representation of the output in 'candidates'
            that minimizes the "distance" with the "exact" prediction in the Hilbert space.
            
            leaves_preds[i*return_top_k : (i+1)*return_top_k] is the non-ordered list od the 
            decoded outputs of the node i among candidates.
        """
        kernel = self.kernel
        
        if not isinstance(kernel, Kernel):
            params = ()
            if isinstance(kernel, tuple):
                kernel, params = kernel[0], kernel[1:]
            try:
                kernel = KERNELS[kernel](*params)
            except KeyError:
                print("Error : 'kernel' attribute (or its first element if it is a tuple) has to be either a Kernel class or a string which is a valid key for the dict 'KERNELS'.")
                raise


        check_is_fitted(self)

        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA[self.criterion](0)
        
        if candidates is not None:
            # candidates doit etre un ensemble : pas de répétition
            candidates = np.unique(candidates, axis=0)
            
            K_cand_train = kernel.get_Gram_matrix(candidates, self.tree_.y)
            sq_norms_cand = kernel.get_sq_norms(candidates)
        else: # recherche dans le learning set
            candidates, indices = np.unique(self.tree_.y, return_index=True, axis=0)
            K_cand_train = self.tree_.K_y[indices]
            sq_norms_cand = self.tree_.K_y[indices, indices]
        
        if return_top_k > 1 and return_top_k >= len(candidates):
            warnings.warn("Le nombre de prédictions demandées pour chaque entrée dépasse le nombre de sorties candidates, return_top_k va être réduit à sa valeur maximale.")
            return_top_k = len(candidates)-1
        
        if "reg" in kernel.get_name() and return_top_k > 1:
            warnings.warn("On ne peut pas retourner plusieurs candidats d'outputs dans le cas d'une régression, veuillez plutôt choisir kernel=linear. "
                          "return_top_k va etre mis à 1.")
            return_top_k = 1
        
        leaves_preds = self.tree_.decode_tree(K_cand_train = K_cand_train, sq_norms_cand=sq_norms_cand, criterion=criterion, kernel=kernel.get_name(), return_top_k=return_top_k)
                
        if not("reg" in kernel.get_name() or "clf" in kernel.get_name()):
            # les outputs récupérées sont alors des indices corrspondants au set de candidats
            # on traduit les indices renvoyés en représentations vectorielles
            leaves_preds = candidates[leaves_preds]
            
        
        if leaves_preds.shape[1] == 1:
            leaves_preds = leaves_preds.reshape(-1)
        
        # On stocke les sorties décodées de l'arbre pour pouvoir les sortir rapidement pour les prochains décodages.
        self.leaves_preds = leaves_preds
        
        return leaves_preds
    
    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)


    def score(self, X, y, candidates=None, metric="accuracy", sample_weight=None):
        """
        Calcule le score après décodage 
        
        Return either
        
            -the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        
        (IF self.kernel="mse_reg")
        
            -the mean accuracy of the predictions if metric="accuracy".
            (Requires that all labels match to count as positive in case of multilabel)
            
            -the mean hamming score of the predictions if metric="hamming"
            (Well suited for multilabel classification)
            
            -the mean top k accuracy score if metric="top_"+str(k)
            It works with all wanted value of k.

        It is possible to set the 'sample_weight' parameter for all these metrics.
        
        Note:
        -----
        All this score metrics are highly dependent on the candidates set because it
        deals with the decoded predictions (which are among this set).
        If you want to compute a score only based on the tree structure, you can
        use the following method 'r2_score_in_Hilbert'.
        

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True outputs for X.
        candidates : array-like of shape (nb_candidates, n_outputs)
            Possible decoded outputs for X
        metric : str, default="accuracy"
            The way to compute the score
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Chosen score between self.predict(X) and y.
        """
        kernel = self.kernel
        if isinstance(kernel, Kernel):
            kernel = kernel.get_name()
        elif isinstance(kernel, tuple):
            kernel = kernel[0]
        return_top_k = 1
        top_k_score = False
        if metric[:4] == "top_":
            try:
                return_top_k = int(metric[4:])
                top_k_score = True
            except ValueError:
                raise(ValueError("Pour calculer le score 'top k', veuillez renseigner un nombre juste après le 'top_'. Nous avons reçu '"+metric[4:]+"'."))
        
        y_pred = self.predict(X, candidates=candidates, return_top_k=return_top_k)
        
        if "reg" in kernel:
            return r2_score(y, y_pred, sample_weight=sample_weight)
        else:
            if metric == "accuracy":
                return accuracy_score(y, y_pred, sample_weight=sample_weight)
            elif metric == "hamming":
                return 1 - hamming_loss(y, y_pred, sample_weight=sample_weight)
            elif top_k_score:
                contains_true = [False]*len(y)
                for ex in range(len(y)):
                    for candidate in range(ex*return_top_k, (ex+1)*return_top_k):
                        if np.atleast_1d(y[ex] == y_pred[candidate]).all():
                            contains_true[ex] = True
                if sample_weight is not None:
                    score = np.sum(sample_weight[contains_true]) / np.sum(sample_weight)
                else:
                    score = np.sum(contains_true) / len(y)
                return score
            else:
                raise ValueError("La metric renseignée n'est pas prise en charge.")


    def r2_score_in_Hilbert(self, X, y, sample_weight=None):
        """
        Calcule le score R2 SANS décodage 
        
        Return the coefficient of determination R^2 of the prediction in the Hilbert space.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True outputs for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            R2 score of the predictions in the Hilbert space wrt. the embedded values of y.
        """
        
        kernel = self.kernel
        
        if not isinstance(kernel, Kernel):
            params = ()
            if isinstance(kernel, tuple):
                kernel, params = kernel[0], kernel[1:]
            try:
                kernel = KERNELS[kernel](*params)
            except KeyError:
                print("Error : 'kernel' attribute (or its first element if it is a tuple) has to be either a Kernel class or a string which is a valid key for the dict 'KERNELS'.")
                raise
        
        weights = self.predict_weights(X)
        
        K_train = self.tree_.K_y
        
        K_test_train = kernel.get_Gram_matrix(y, self.tree_.y)
        
        K_test_test = kernel.get_Gram_matrix(y)


        if sample_weight is not None:
            if len(sample_weight != len(y)):
                raise ValueError("sample_weights has to have the same length as y. "
                                 "y is len "+str(len(y))+", and sample_weight is len "+str(len(sample_weight)))
            sample_weight[sample_weight<0] = 0
            if np.sum(sample_weight) == 0:
                warnings.warn("all weights in sample_weight were set to 0 or bellow. It is unvalid so sample_weight will be ignored.")
                sample_weight = None

        res_sq_sums = np.diag(K_test_test) - 2 * np.diag(K_test_train @ (weights.T)) + np.diag(weights @ K_train @ (weights.T))
        
        if sample_weight is not None:
            tot_sq_sums = np.diag(K_test_test) - np.sum(K_test_test @ np.diag(sample_weight), axis=1) / np.sum(sample_weight)
        else:
            tot_sq_sums = np.diag(K_test_test) - np.sum(K_test_test, axis=1)/K_test_test.shape[1]
        
        
        if sample_weight is not None:
            res_sq_sum = np.sum(sample_weight*res_sq_sums) / np.sum(sample_weight)
            tot_sq_sum = np.sum(sample_weight*tot_sq_sums) / np.sum(sample_weight)
        else:
            res_sq_sum = np.mean(res_sq_sums)
            tot_sq_sum = np.mean(tot_sq_sums)
        
        r2 = 1 - ( res_sq_sum / tot_sq_sum )
        
        return r2

    
    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    def _prune_tree(self):
        """Prune tree using Minimal Cost-Complexity Pruning."""
        check_is_fitted(self)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if self.ccp_alpha == 0.0:
            return

        # build pruned tree
        pruned_tree = Tree(self.n_features_, self.tree_.K_y.shape[0])
        _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

        self.tree_ = pruned_tree

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """Compute the pruning path during Minimal Cost-Complexity Pruning.

        See :ref:`minimal_cost_complexity_pruning` for details on the pruning
        process.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        ccp_path : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            ccp_alphas : ndarray
                Effective alphas of subtree during pruning.

            impurities : ndarray
                Sum of the impurities of the subtree leaves for the
                corresponding alpha value in ``ccp_alphas``.
        """        
        est = clone(self).set_params(ccp_alpha=0.0)
        est.fit(X, y, sample_weight=sample_weight)
        return Bunch(**ccp_pruning_path(est.tree_))

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).
        """
        check_is_fitted(self)

        return self.tree_.compute_feature_importances()


# =============================================================================
# Public estimators
# =============================================================================

class OK3Regressor(BaseKernelizedOutputTree):
    """A decision tree regressor for the OK3 method.

    Parameters
    ----------
    criterion : {"mse"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float, (default=0)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    kernel : string, or tuple (string, params) or instance of the Kernel class, default="linear"
        The type of kernel to use to compare the output data. Changing this
        parameter changes also implicitely the nature of the Hilbert space
        in which the output data are embedded.
        The string describes the type of Kernel to use (defined in Kernel.py), 
        The optional params given are here to set particular parameters values
        for the chosen kernel type.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    tree_ : Tree
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.
    
    leaves_preds : array of shape (n_nodes, n_components),
        where n_nodes is the number of nodes of the grown tree and
        n_components is the number of values used to represent an output.
        
        This array stores for each leaf of the tree, the decoded predictions in Y.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] Pierre Geurts, Louis Wehenkel, Florence d’Alché-Buc. 
          "Kernelizing the output of tree-based methods."
          Proc.  of the 23rd International Conference on Machine Learning, 
          2006, United States.  pp.345–352,￿10.1145/1143844.1143888￿. 

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from ??? import OK3Regressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = OK3Regressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 ccp_alpha=0.0, 
                 kernel="linear"):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha, 
            kernel=kernel)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated", kernel=None, in_ensemble=False, Gram_y=None):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : deprecated, default="deprecated"
            This parameter is deprecated and has no effect.
            It will be removed in v0.26.
        
        kernel : string, or tuple (string, params) or instance of the Kernel class, default="linear"
            The type of kernel to use to compare the output data. Changing this
            parameter changes also implicitely the nature of the Hilbert space
            in which the output data are embedded.
            The string describes the type of Kernel to use (defined in Kernel.py), 
            The optional params given are here to set particular parameters values
            for the chosen kernel type.
            This parameter can be set also here in the fit method instead of __init__.
        
        Returns
        -------
        self : OK3Regressor
            Fitted estimator.
        """

        super().fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
            kernel=kernel, 
            in_ensemble=in_ensemble, 
            Gram_y=Gram_y)
        return self


class ExtraOK3Regressor(OK3Regressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Parameters
    ----------
    criterion : {"mse", "friedman_mse", "mae"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"auto", "sqrt", "log2"} or None, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float, (default=0)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        
    kernel : string, or tuple (string, params) or instance of the Kernel class, default="linear"
        The type of kernel to use to compare the output data. Changing this
        parameter changes also implicitely the nature of the Hilbert space
        in which the output data are embedded.
        The string describes the type of Kernel to use (defined in Kernel.py), 
        The optional params given are here to set particular parameters values
        for the chosen kernel type.

    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        Return impurity-based feature importances (the higher, the more
        important the feature).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    tree_ : Tree
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    ExtraTreeClassifier : An extremely randomized tree classifier.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.tree import ExtraTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> extra_tree = ExtraTreeRegressor(random_state=0)
    >>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
    ...     X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.33...
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 criterion="mse",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 max_leaf_nodes=None,
                 ccp_alpha=0.0, 
                 kernel="linear"):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state,
            ccp_alpha=ccp_alpha, 
            kernel=kernel)
