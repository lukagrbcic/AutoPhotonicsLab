import numpy as np
from indago import PSO
import random
from sklearn.ensemble import RandomForestRegressor

class modelSampler:
    
    def __init__(self, X, y, sample_size, lb, ub):
        
        self.X = X
        self.y = y
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub

        self.model = RandomForestRegressor().fit(self.X, self.y)

    def gen_samples(self):
        print ('running original code!')
        
        X = []
        f = []
        for i in range(self.sample_size):
            
            def get_values(x):               
      
                preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
                
                return -np.sum(np.std(preds, axis=0))

            optimizer = PSO()
            optimizer.evaluation_function = get_values 
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = 100
            result = optimizer.optimize()
            min_x = result.X 
            min_f = result.f
                            
            X.append(min_x)
            f.append(min_f)
                                
        X = np.array(X)
        f = np.array(f)
        
        X = X[np.argsort(f)]

        return X, self.model
            
            
        
        
