# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs

import numpy as np
cimport numpy as np
np.import_array()

from sklearn.tree._utils cimport log
from sklearn.tree._utils cimport safe_realloc
from sklearn.tree._utils cimport sizet_ptr_to_ndarray
from sklearn.tree._utils cimport WeightedMedianCalculator

# from kernel import Kernel

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        pass

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that stores values of the output Gramm matrix of the samples
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of the samples being considered
        samples : array-like, dtype=SIZE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node

        """

        pass

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """

        pass

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will save the weighted 
        samples[start:end] into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        # cdef double impurity_left
        # cdef double impurity_right
        # self.children_impurity(&impurity_left, &impurity_right)

        # return (- self.weighted_n_right * impurity_right
        #         - self.weighted_n_left * impurity_left)
        
        pass

    cdef double impurity_improvement(self, double impurity) nogil:
        """Compute the improvement in impurity

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity : double
            The initial impurity of the node before the split

        Return
        ------
        double : improvement in impurity after the split occurs
        """

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right / 
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left / 
                             self.weighted_n_node_samples * impurity_left)))














cdef class KernelizedRegressionCriterion(Criterion):
    r"""Abstract kernelized output regression criterion.

    This handles cases where the target is a structured object and the Gramm
    matrix (the matrix of the kernel evaluated at the output samples) is given
    as y. The impurity is evaluated by computing the variance of the target
    values (embedded in a larger Hilbert space) left and right of the split point.
    """

    def __cinit__(self, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sum_diag_Gramm = 0.0
        self.sum_total_Gramm = 0.0
        
        self.sum_diag_Gramm_left = 0.0
        self.sum_diag_Gramm_right = 0.0
        
        self.sum_total_Gramm_left = 0.0
        self.sum_total_Gramm_right = 0.0


    def __reduce__(self):
        return (type(self), (self.n_samples,), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t p
        cdef SIZE_t q
        cdef DOUBLE_t w_i = 1.0
        cdef DOUBLE_t w_j = 1.0

        self.sum_diag_Gramm = 0.0
        self.sum_total_Gramm = 0.0
        
        self.sum_diag_Gramm_left = 0.0
        self.sum_diag_Gramm_right = 0.0
        
        self.sum_total_Gramm_left = 0.0
        self.sum_total_Gramm_right = 0.0

        for p in range(start, end):
            i = samples[p]
            # with gil:
            #     print("print samples :",i)

            if sample_weight != NULL:
                w_i = sample_weight[i]

            self.weighted_n_node_samples += w_i
            
            self.sum_diag_Gramm += w_i * self.y[i,i]
            
            for q in range(start, end):
                j = samples[q]
                
                if sample_weight != NULL:
                    w_j = sample_weight[j]
                
                self.sum_total_Gramm += w_i * w_j * self.y[i,j]

        # Reset to pos=start
        self.reset()
        # with gil:
        #     print("print sum diag  :",self.sum_diag_Gramm)
        #     print("print sum total :",self.sum_total_Gramm)
        #     print("print weighted_n_node_samples :",self.weighted_n_node_samples)
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        
        self.sum_diag_Gramm_left = 0.0
        self.sum_diag_Gramm_right = self.sum_diag_Gramm
        
        self.sum_total_Gramm_left = 0.0
        self.sum_total_Gramm_right = self.sum_total_Gramm
        
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        
        self.sum_diag_Gramm_right = 0.0
        self.sum_diag_Gramm_left = self.sum_diag_Gramm
        
        self.sum_total_Gramm_right = 0.0
        self.sum_total_Gramm_left = self.sum_total_Gramm

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t p
        cdef SIZE_t q
        cdef DOUBLE_t w_i = 1.0
        cdef DOUBLE_t w_j = 1.0

        # Update statistics up to new_pos

        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w_i = sample_weight[i]
            
            self.sum_diag_Gramm_left += w_i * self.y[i,i]
            
            self.sum_diag_Gramm_right -= w_i * self.y[i,i]

            self.weighted_n_left += w_i
            
            self.weighted_n_right -= w_i
        
            for q in range(start, pos):
                j = samples[q]
                    
                if sample_weight != NULL:
                    w_j = sample_weight[j]
                
                self.sum_total_Gramm_left += 2 * w_i * w_j * self.y[i,j]
            
            for q in range(pos, new_pos):
                j = samples[q]
                    
                if sample_weight != NULL:
                    w_j = sample_weight[j]
                
                self.sum_total_Gramm_left += w_i * w_j * self.y[i,j]
                
                self.sum_total_Gramm_right -= w_i * w_j * self.y[i,j]
        
        for p in range(new_pos, end):
            i = samples[p]

            if sample_weight != NULL:
                w_i = sample_weight[i]
            
            for q in range(pos, new_pos):
                j = samples[q]
                    
                if sample_weight != NULL:
                    w_j = sample_weight[j]
                
                self.sum_total_Gramm_right -= 2 * w_i * w_j * self.y[i,j]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(self.start, self.end):
            
            k = self.samples[p]
            
            if self.sample_weight != NULL:
                w = self.sample_weight[k]
            
            dest[k] = w / self.weighted_n_node_samples















cdef class KernelizedMSE(KernelizedRegressionCriterion):
    """Mean squared error impurity criterion.
    
        var = \sum_i^n (phi(y_i) - phi(y)_bar) ** 2
            = (\sum_i^n phi(y_i) ** 2) - n_samples * phi(y)_bar ** 2

        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double impurity

        impurity = self.sum_diag_Gramm / self.weighted_n_node_samples - self.sum_total_Gramm / (self.weighted_n_node_samples)**2

        return impurity

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double proxy_impurity_left = self.sum_diag_Gramm_left - self.sum_total_Gramm_left / self.weighted_n_left
        cdef double proxy_impurity_right = self.sum_diag_Gramm_right - self.sum_total_Gramm_right / self.weighted_n_right

        return (- proxy_impurity_left - proxy_impurity_right)

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        cdef double sum_diag_Gramm_left = self.sum_diag_Gramm_left
        cdef double sum_diag_Gramm_right = self.sum_diag_Gramm_right
        
        cdef double sum_total_Gramm_left = self.sum_total_Gramm_left
        cdef double sum_total_Gramm_right = self.sum_total_Gramm_right

        impurity_left[0] = sum_diag_Gramm_left / self.weighted_n_left - sum_total_Gramm_left / (self.weighted_n_left)**2
        impurity_right[0] = sum_diag_Gramm_right / self.weighted_n_right - sum_total_Gramm_right / (self.weighted_n_right)**2


