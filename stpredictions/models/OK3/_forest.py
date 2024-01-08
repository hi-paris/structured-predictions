"""
Forest of structured-output-trees-based ensemble methods.

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseOKForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``OKForestRegressor`` base class further implement the prediction logic 
  by computing an average of the predicted training weights of the 
  sub-estimators, and then decode with these predicted weights.
  It cannot just compute the mean of the predictions made by each tree 
  because structured outputs may not be summable.

- The ``RandomOKForestRegressor`` derived classes provide the user with 
  concrete implementations of the forest ensemble method using 
  deterministic ``OK3Regressor`` as sub-estimator implementations.

- The ``ExtraOKTreesRegressor`` derived classes provide the user with concrete
  implementations of the forest ensemble method using the extremely randomized
  OK trees ``ExtraOKTreeRegressor`` as sub-estimator implementations.

Structured outputs are represented as vectors of numerical data.
"""

import numbers
from warnings import warn
import threading

import time

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel, delayed
from pkg_resources import parse_version

import itertools

from .base import StructuredOutputMixin

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import r2_score

from sklearn.preprocessing import OneHotEncoder
from ._classes import OK3Regressor, ExtraOK3Regressor
from ._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array
from sklearn.exceptions import DataConversionWarning

from sklearn.ensemble._base import BaseEnsemble, _partition_estimators

# from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args

from .kernel import Kernel
from ._classes import KERNELS, CRITERIA

from ._criterion import Criterion, KernelizedMSE

# import line_profiler

__all__ = ["RandomOKForestRegressor",
           "ExtraOKTreesRegressor",
           "RandomOKTreesEmbedding"]

MAX_INT = np.iinfo(np.int32).max

def _joblib_parallel_args(**kwargs):
    """Set joblib.Parallel arguments in a compatible way for 0.11 and 0.12+

    For joblib 0.11 this maps both ``prefer`` and ``require`` parameters to
    a specific ``backend``.

    Parameters
    ----------

    prefer : str in {'processes', 'threads'} or None
        Soft hint to choose the default backend if no specific backend
        was selected with the parallel_backend context manager.

    require : 'sharedmem' or None
        Hard condstraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based even
        if the user asked for a non-thread based backend with
        parallel_backend.

    See joblib.Parallel documentation for more details
    """
    import joblib

    if parse_version(joblib.__version__) >= parse_version('0.12'):
        return kwargs

    extra_args = set(kwargs.keys()).difference({'prefer', 'require'})
    if extra_args:
        raise NotImplementedError('unhandled arguments %s with joblib %s'
                                  % (list(extra_args), joblib.__version__))
    args = {}
    if 'prefer' in kwargs:
        prefer = kwargs['prefer']
        if prefer not in ['threads', 'processes', None]:
            raise ValueError('prefer=%s is not supported' % prefer)
        args['backend'] = {'threads': 'threading',
                           'processes': 'multiprocessing',
                           None: None}[prefer]

    if 'require' in kwargs:
        require = kwargs['require']
        if require not in [None, 'sharedmem']:
            raise ValueError('require=%s is not supported' % require)
        if require == 'sharedmem':
            args['backend'] = 'threading'
    return args

def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return round(n_samples * max_samples)

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples,
                                              n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def _parallel_build_trees(tree, forest, X, y, Gram_y, sample_weight, tree_idx, n_trees,
                          verbose=0, n_samples_bootstrap=None):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples,
                                           n_samples_bootstrap)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False, in_ensemble=True, Gram_y=Gram_y)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False, in_ensemble=True, Gram_y=Gram_y)
    
    return tree


class BaseOKForest(StructuredOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of ok-trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None,
                 kernel="linear"):        

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples
        self.kernel = kernel

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           **_joblib_parallel_args(prefer="threads"))(
            delayed(tree.apply)(X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              **_joblib_parallel_args(prefer='threads'))(
            delayed(tree.decision_path)(X, check_input=False)
            for tree in self.estimators_)

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr
    
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of ok-trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        kernel = self.kernel
        
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

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
            warn("The target parameter is a square matrix."
                          "Are you sure this is the matrix of outputs and "
                          "not a Gram matrix ? ")
                    
        if "clf" in kernel.get_name():
            check_classification_targets(y)
        
        start_computation = time.time()
        # compute the Gram matrix of the outputs
        K_y = kernel.get_Gram_matrix(y)
        print("Time to compute the training Gram matrix : " + str(time.time() - start_computation) + " s.")
        
        self.n_outputs_ = y.shape[1]
        
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        if getattr(K_y, "dtype", None) != DOUBLE or not K_y.flags.contiguous:
            K_y = np.ascontiguousarray(K_y, dtype=DOUBLE)

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X, y, K_y, sample_weight, i, len(trees),
                    verbose=self.verbose, 
                    n_samples_bootstrap=n_samples_bootstrap)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y, K_y)

        return self

    @abstractmethod
    def _set_oob_score(self, X, y, K_y):
        """
        Calculate out of bag predictions and score."""

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   **_joblib_parallel_args(prefer='threads'))(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    
    It sums the predictions given by predict to the memory 'out'.
    It will be used to sum the predictions of weights among the training samples.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class OKForestRegressor(BaseOKForest, metaclass=ABCMeta):
    """
    Base class for forest of ok-trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None,
                 kernel="linear"):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            kernel=kernel)
    
    def predict_weights(self, X):
        """
        Predict weights (on the training samples) for X.

        The predicted weights of an input sample are computed as the
        mean predicted weights of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        A (X.shape[0], n_training_samples) array which gives for each test example (line number)
        and for each training sample its weight in the node (O if the sample doesn't fall
        in any of the same leaves as the test example, and a non-negative value depending on 'sample_weight' otherwise.)
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        
        # avoid storing the output of every estimator by summing them here
        weights = np.zeros((X.shape[0], self.estimators_[0].tree_.y.shape[0]), dtype=np.float64)
        
        # Parallel loop
        lock = threading.Lock()

        Parallel(n_jobs=n_jobs, verbose=self.verbose,
             **_joblib_parallel_args(require="sharedmem"))(
        delayed(_accumulate_prediction)(e.predict_weights, X, [weights], lock)
        for e in self.estimators_)

        weights /= len(self.estimators_)

        return weights



    
    def predict(self, X, candidates=None, return_top_k=1, precomputed_weights=None):
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
        X = self._validate_X_predict(X)
        
        criterion = self.estimators_[0].criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA[criterion](X.shape[0])

        if candidates is not None:
            # candidates doit etre un ensemble : pas de répétition
            candidates = np.unique(candidates, axis=0)
            
            K_cand_train = kernel.get_Gram_matrix(candidates, self.estimators_[0].tree_.y)
            sq_norms_cand = kernel.get_sq_norms(candidates)
        else: # recherche dans le learning set
            candidates, indices = np.unique(self.estimators_[0].tree_.y, return_index=True, axis=0)
            K_cand_train = self.estimators_[0].tree_.K_y[indices]
            sq_norms_cand = self.estimators_[0].tree_.K_y[indices, indices]
        
        if return_top_k > 1 and return_top_k >= len(candidates):
            warn("Le nombre de prédictions demandées pour chaque entrée dépasse le nombre de sorties candidates, return_top_k va être réduit à sa valeur maximale.")
            return_top_k = len(candidates)-1
        
        if "reg" in kernel.get_name() and return_top_k > 1:
            warn("On ne peut pas retourner plusieurs candidats d'outputs dans le cas d'une régression, veuillez plutôt choisir kernel=linear. "
                          "return_top_k va etre mis à 1.")
            return_top_k = 1
        
        if precomputed_weights is None:
            weights = self.predict_weights(X)
        else:
            weights = precomputed_weights
        
        if isinstance(criterion, KernelizedMSE):

            # Cas particulier de la classification : recherche EXHAUSTIVE
            if kernel.get_name() == "gini_clf":
                
                # rechercher la meilleure combinaison de labels parmis toutes celles possible
                                    
                y_train = self.estimators_[0].tree_.y
                n_outputs = y_train.shape[1]

                classes = []
                n_classes = []
                
                y_train_encoded = np.zeros((y_train.shape[0], n_outputs), dtype=int)
                
                for l in range(n_outputs):
                    classes_l, y_train_encoded[:, l] = np.unique(y_train[:, l], return_inverse=True)
                    classes.append(classes_l)
                    n_classes.append(classes_l.shape[0])
                
                n_classes = np.array(n_classes, dtype=np.intp)
                                
                out = np.zeros((X.shape[0]*return_top_k,n_outputs), dtype=np.intp)
    
                nb_candidates = 1
                for nb_classes in n_classes:
                    nb_candidates *= nb_classes
                # array to store the value of the criteria to minimize, for each training sample
                value = np.zeros((nb_candidates,), dtype=np.float64)
                
                recherche_exhaustive_equivalente = False
                
                # node k
                for test_ex in range(X.shape[0]):

                    if recherche_exhaustive_equivalente or return_top_k > 1: # n_outputs boucles sur les classes de chaque output imbriquées dans le product --> long
                        
                        for ind, candidate in enumerate(list(itertools.product(*classes))):
                            
                            # la valeur a minimiser est k(candidate,candidate) - 2 * moyenne_des_Kernel(candidate,train_exs_in_same_leaf)
                            # dans le cas de gini, k(candidate,candidate) est toujours égal à 1 peu importe candidate
                            # on peut donc plutôt maximiser la quantité somme_des_Kernel(candidate,train_exs_in_same_leaf)
                                                            
                            value[ind] = np.sum([ weights[test_ex,ex] * (y_train[ex] == candidate).mean() for ex in range(weights.shape[1])])
                        
                        ind_top_candidates = np.argpartition(value, - return_top_k)[- return_top_k:]
                            
                        top_candidates = list(itertools.product(*classes))[ind_top_candidates]
                        top_candidates = np.array(top_candidates, dtype=int)
                        
                        out[test_ex*return_top_k : (test_ex+1)*return_top_k] = top_candidates
                    
                    else:
                                                        
                        for l in range(n_outputs):

                            major_class = np.argmax( [ np.sum( weights[test_ex, np.where( y_train[:,l] == class_i )[0] ] ) for class_i in classes[l] ] )
                            
                            out[test_ex,l] = classes[l][ major_class ]

            # Cas particulier de la régression : Recherche EXACTE
            elif kernel.get_name() == "mse_reg": 
                
                # rechercher la meilleure combinaison de labels parmis toutes celles possible
                # avec un critère MSE et donc un kernel linéaire, 
                # la solution exacte (argmin_y [k(y,y) - 2 moyenne_i(k(y,y_leaf_i))]) peut être calculée : 
                # c'est la moyenne des sorties de chaque feuille
                #
                # On ne peut pas rechercher les k meilleurs candidats car l'ensemble de recherche de candidats pour la régression est infini (R^d)
                                    
                y_train = self.estimators_[0].tree_.y
                n_outputs = y_train.shape[1]
                
                out = weights @ y_train
                
            # Dans ce else, on a donc une matrice de Gram de candidats fournie
            else: # cas général : pas de classification ou de régression mais recherche de l'argmin dans l'ensemble de candidats fourni

                # on a comme candidats une matrice de Gram des des candidats contre les training (+contre soi meme).
                
                # on renvoie l'indce du candidat représentant le mieux la feuille (on ne check pas les training examples, ils sont à mettre dans les candidats)
                
                out = np.zeros((X.shape[0]*return_top_k,candidates.shape[1]), dtype=candidates.dtype)
                
                # array to store the value of the criteria to minimize, for each training sample
                value = np.zeros((len(candidates),), dtype=np.float64)
                
                for test_ex in range(X.shape[0]):

                    # parmi les candidats, calculer k[candidat,candidat] - 2/self.n_node_samples * sum_i=0^self.n_node_samples k[candidat,i]
                    for candidate in range(len(candidates)):
                        
                        value[candidate] = sq_norms_cand[candidate] - 2 * np.sum([weights[test_ex,ex] * K_cand_train[candidate,ex] for ex in range(weights.shape[1])])
                    
                    # choisir l'entrée ex* qui donnait la plus petite valeur
                    ind_top_candidates = np.argpartition(value, return_top_k)[:return_top_k]
                    
                    out[test_ex*return_top_k : (test_ex+1)*return_top_k] = candidates[ind_top_candidates]
                        
            if out.shape[1] == 1:
                out = out.reshape(-1)
                
            return out

            
        else:
            raise NotImplementedError('only the "KernelizedMSE" criterion is supported')
    

    
    def _set_oob_score(self, X, y, K_y, candidates=None, metric="accuracy"):
        """
        Compute out-of-bag R2 scores in self.oob_score (computed in the HS : no need to decode).
        
        AND also the score on the decoded outputs (in self.oob_decoded_score)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Train samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True outputs for X.
        Returns
        -------
        None
        """
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        weights = np.zeros((n_samples, n_samples))
        n_predictions = np.zeros((n_samples, n_samples))

        for estimator in self.estimators_:
            
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples, n_samples_bootstrap)

            p_estimator = estimator.predict_weights(
                X[unsampled_indices, :], check_input=False)

            weights[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1
        
        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        weights /= n_predictions
        # on enregistre LES POIDS : pas les prédictions décodées
        self.oob_prediction_ = weights
                    
        # TODO : vérifier si ce calcul est correct pour le R2 dans le HS des samples oob.

        K_train = K_y

        res_sq_sums = np.diag(K_train) - 2 * np.diag(K_train @ (weights.T)) + np.diag(weights @ K_train @ (weights.T))
        
        tot_sq_sums = np.diag(K_train) - np.sum(K_train, axis=1)/K_train.shape[1]
        
        
        res_sq_sum = np.mean(res_sq_sums)
        tot_sq_sum = np.mean(tot_sq_sums)
        
        r2 = 1 - ( res_sq_sum / tot_sq_sum )
        
        self.oob_score_ = r2
            
        # compute also the decoded score :
        # compute the same function as 'score' bellow
        
        score = self.score(X, y, candidates=candidates, metric=metric, precomputed_weights=weights)
        
        self.oob_decoded_score_ = score
    
    def score(self, X, y, candidates=None, metric="accuracy", sample_weight=None, precomputed_weights=None):
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
        
        y_pred = self.predict(X, candidates=candidates, return_top_k=return_top_k, precomputed_weights=precomputed_weights)
        
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
        
        K_train = self.estimators_[0].tree_.K_y
        
        K_test_train = kernel.get_Gram_matrix(y, self.estimators_[0].tree_.y)
        
        K_test_test = kernel.get_Gram_matrix(y)


        if sample_weight is not None:
            if len(sample_weight != len(y)):
                raise ValueError("sample_weights has to have the same length as y. "
                                 "y is len "+str(len(y))+", and sample_weight is len "+str(len(sample_weight)))
            sample_weight[sample_weight<0] = 0
            if np.sum(sample_weight) == 0:
                warn("all weights in sample_weight were set to 0 or bellow. It is unvalid so sample_weight will be ignored.")
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



class RandomOKForestRegressor(OKForestRegressor):
    """
    A random ok-forest regressor.

    A random forest is a meta estimator that fits a number of
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"mse"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion

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

        .. versionchanged:: 0.18
           Added float values for fractions.

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

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
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

    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    base_estimator_ : DecisionTreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    See Also
    --------
    OK3Regressor, ExtraOKTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    The default value ``max_features="auto"`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [1], whereas the former was more recently justified empirically in [2].

    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomOKForestRegressor(max_depth=2, random_state=0, kernel="linear")
    >>> regr.fit(X, y)
    RandomForestRegressor(...)
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-8.32987858]
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None,
                 kernel="linear"):
        super().__init__(
            base_estimator=OK3Regressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha", "kernel"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            kernel=kernel)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha


class ExtraOKTreesRegressor(OKForestRegressor):
    """
    An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"mse"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion
        
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

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
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

    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    base_estimator_ : ExtraTreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_ : int
        The number of features.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    See Also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.

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
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = ExtraOKTreesRegressor(n_estimators=100, random_state=0, kernel="gaussian").fit(
    ...    X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.2708...
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None,
                 kernel="linear"):
        super().__init__(
            base_estimator=ExtraOK3Regressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha", "kernel"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            kernel=kernel)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha


class RandomOKTreesEmbedding(BaseOKForest):
    """
    An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    max_depth : int, default=5
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
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

    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    sparse_output : bool, default=True
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`transform`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls the generation of the random `y` used to fit the trees
        and the draw of the splits for each feature at the trees' nodes.
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    base_estimator_ : DecisionTreeClassifier instance
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeClassifier instances
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    one_hot_encoder_ : OneHotEncoder instance
        One-hot encoder used to create the sparse embedding.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    Examples
    --------
    >>> from sklearn.ensemble import RandomTreesEmbedding
    >>> X = [[0,0], [1,0], [0,1], [-1,0], [0,-1]]
    >>> random_trees = RandomTreesEmbedding(
    ...    n_estimators=5, random_state=0, max_depth=1, kernel="gaussian").fit(X)
    >>> X_sparse_embedding = random_trees.transform(X)
    >>> X_sparse_embedding.toarray()
    array([[0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.]])
    """

    criterion = 'mse'
    max_features = 1

    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 sparse_output=True,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 kernel="linear"):
        super().__init__(
            base_estimator=ExtraOK3Regressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "kernel"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=None,
            kernel=kernel)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y, K_y):
        raise NotImplementedError("OOB score not supported by tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object

        """
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Fit estimator and transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        X = check_array(X, accept_sparse=['csc'])
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])
        if "clf" in self.kernel:
            warn("Un noyau opérant sur des valeurs discrètes de y n'est pas compatible avec RandomOKTreesEmbedding, le noyau va être choisi linéaire (régression classique)")
            self.kernel = "mse_reg"
        super().fit(X, y, sample_weight=sample_weight)

        self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
        return self.one_hot_encoder_.fit_transform(self.apply(X))

    def transform(self, X):
        """
        Transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        check_is_fitted(self)
        return self.one_hot_encoder_.transform(self.apply(X))