import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

np.random.seed(random.randint(0, 10223))


class randomSampler:
    
    def __init__(self,  X, y, sample_size, lb, ub):

        self.X = X
        self.y = y
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub

        self.model = RandomForestRegressor().fit(self.X, self.y)

        
    def gen_random_samples(self):
        
        n = self.sample_size
        d = len(self.lb)
        sample_set = np.array([np.random.uniform(self.lb, self.ub, d) for i in range(n)])
        
        return sample_set, self.model
    
        
