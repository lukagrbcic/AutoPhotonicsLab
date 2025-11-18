import numpy as np
from scipy.stats import qmc

class lhsSampler:
    
    def __init__(self, sample_size, lb, ub):
        
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        
    def gen_LHS_samples(self, seed=111):
        
        n = self.sample_size
        d = len(self.ub)
        
        if seed == None:
            sampler = qmc.LatinHypercube(d=d)
        else: 
            print ('using seed', seed)
            sampler = qmc.LatinHypercube(d=d, seed=seed)
            
        samples = sampler.random(n=n)
        sample_set = qmc.scale(samples, self.lb, self.ub)
        
        return sample_set
    
    
    
    
        
