
import numpy as np

class StructuredObject:
    
    def __init__(self):
        
        self.index = 0
        self.vectorial_representation = np.zeros(0)
        self.name = ''
        
    def is_equal_to(self, other):
        
        if self.index == other.index or self.vectorial_representation == other.vectorial_representation:
            return True
    
    def similarity_with(self, other, kernel):
        
        return kernel.evaluate(self, other)
    
    def get_name(self):
        
        return self.name


"""
class LabeledGraph(StructuredObject):
    
    
    
class SimpleVector(StructuredObject):
    
    
    
class Permutation(StructuredObject):
    
    def __init__(self, n_elements):
        
        self.n_elements = n_elements
        
"""
