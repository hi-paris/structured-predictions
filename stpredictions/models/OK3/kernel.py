import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances
from abc import abstractmethod

__all__ = ["Kernel", 
           "Mean_Dirac_Kernel", 
           "Linear_Kernel", 
           "Gaussian_Kernel", 
           "Laplacian_Kernel", 
           "Gini_Kernel", 
           "MSE_Kernel"]

class Kernel:
    
    @abstractmethod
    def __init__(self, name):
        
        self.name = name
        
    def evaluate(self, obj1, obj2):

        pass

    def get_Gram_matrix(self, objects_1, objects_2=None):
        
        pass
    
    def get_sq_norms(self, objects):
        
        pass
    
    def get_name(self):
        
        return self.name



##### Kernel classes

class Mean_Dirac_Kernel(Kernel):
    
    def __init__(self):
        
        super().__init__(name="mean_dirac")
    
    def evaluate(self, y1, y2):
        
        return np.sum(y1==y2) / len(y1)
    
    def get_Gram_matrix(self, list_y_1, list_y_2=None):
        
        return 1 - pairwise_distances(list_y_1, list_y_2, metric="hamming")
    
    def get_sq_norms(self, list_y):
        
        return np.ones(len(list_y)) # = (y==y).mean()


class Gini_Kernel(Mean_Dirac_Kernel):
    """
    Identique au 'Mean_Dirac_Kernel', mais permet de signaler que le décodage 
    ne se fait pas parmi un candidates set mais est une recherche exhaustive.
    """
    
    def __init__(self):
        
        super().__init__()
        self.name = "gini_clf"
    


class Linear_Kernel(Kernel):
    
    def __init__(self):
        
        super().__init__(name="linear")
    
    def evaluate(self, y1, y2):
        
        return np.atleast_1d(y1) @ np.atleast_1d(y2)
    
    def get_Gram_matrix(self, list_y_1, list_y_2=None):
        
        return pairwise_kernels(list_y_1, list_y_2, metric="linear")

    def get_sq_norms(self, list_y):
        
        return np.sum(list_y**2, axis=1)


class MSE_Kernel(Linear_Kernel):
    """
    Identique au 'Linear_Kernel', mais permet de signaler que le décodage 
    ne se fait pas parmi un candidates set mais est une solution exacte.
    """
    
    def __init__(self):
        
        super().__init__()
        self.name = "mse_reg"



class Laplacian_Kernel(Kernel):

    def __init__(self, gamma=1):
        
        name = "laplacian"
        if gamma != 1:
            name += "_" + str(gamma)
        super().__init__(name=name)
        self.gamma = gamma
    
    def evaluate(self, y1, y2):
        
        return np.exp(-self.gamma * np.sum(np.abs(y1 - y2)))
    
    def get_Gram_matrix(self, list_y_1, list_y_2=None):
        
        return pairwise_kernels(list_y_1, list_y_2, metric="laplacian", gamma=self.gamma)

    def get_sq_norms(self, list_y):
        
        return np.ones(len(list_y)) # = exp(-gamma*0)



class Gaussian_Kernel(Kernel):

    def __init__(self, gamma=1):
        
        name = "gaussian"
        if gamma != 1:
            name += "_" + str(gamma)
        super().__init__(name=name)
        self.gamma = gamma
    
    def evaluate(self, y1, y2):
        
        return np.exp(- self.gamma * np.sum((y1 - y2)**2))
    
    def get_Gram_matrix(self, list_y_1, list_y_2=None):
        
        return pairwise_kernels(list_y_1, list_y_2, metric="rbf", gamma=self.gamma)

    def get_sq_norms(self, list_y):
        
        return np.ones(len(list_y)) # = exp(-gamma*0)
    


# Attention, noyau calculé naïvement : pas optimisé.
        
# kernel for permutations
# les permutations sigma de Sn sont représentées par des vecteurs y de 
# même taille n où y[i] = p <=> sigma(i) = p
class Mallows_Kernel(Kernel):

    def __init__(self, gamma=1):
        
        name = "mallows"
        if gamma != 1:
            name += "_" + str(gamma)
        super().__init__(name=name)
        self.gamma = gamma
    
    def evaluate(self, y1, y2):
        
        kendall_tau_dist = 0
        for i in range(len(y1)-1):
            for j in range(i+1, len(y1)):
                
                kendall_tau_dist += ((y1[i] < y1[j]) and (y2[i] > y2[j])) + ((y1[i] > y1[j]) and (y2[i] < y2[j]))
        
        return np.exp(-self.gamma * kendall_tau_dist)
    
    def get_Gram_matrix(self, list_y_1, list_y_2=None):
        
        K = np.zeros((len(list_y_1), len(list_y_2)), dtype=np.double)
        for i in range(len(list_y_1)):
            for j in range(i, len(list_y_2)):
                K[i,j] = self.evaluate(list_y_1[i], list_y_2[j])
        return K

    def get_sq_norms(self, list_y):
        
        return np.ones(len(list_y)) # = exp(-gamma*0)




