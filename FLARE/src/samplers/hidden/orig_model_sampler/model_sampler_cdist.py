
import numpy as np
from indago import PSO
import random
from joblib import Parallel, delayed

np.random.seed(random.randint(0, 10223))

class modelSampler:
    
    def __init__(self, model, sample_size, lb, ub, algorithm, function='uncertainty'):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function     

    def get_values(self, x):
        preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
        return -np.sum(np.std(preds, axis=0))

    def optimize_single_sample(self):
        optimizer = PSO()
        optimizer.evaluation_function = self.get_values 
        optimizer.lb = self.lb
        optimizer.ub = self.ub
        optimizer.max_evaluations = 100
        result = optimizer.optimize()
        return result.X, result.f

    def get_samples(self):
        # Parallelize the optimization process
        results = Parallel(n_jobs=-1)(delayed(self.optimize_single_sample)() for _ in range(self.sample_size))
        
        X, f = zip(*results)
        X = np.array(X)
        f = np.array(f)
        
        # Sort by the evaluation function value
        sorted_indices = np.argsort(f)
        X = X[sorted_indices]
        
        # Ensure diversity by selecting the most diverse points
        selected_indices = self.select_diverse_points(X, self.sample_size)
        X = X[selected_indices]
        
        return X
    
    def select_diverse_points(self, X, n_samples):
        if n_samples >= len(X):
            return np.arange(len(X))
        
        selected_indices = [0]  # Start with the first point
        distances = np.linalg.norm(X - X[0], axis=1)
        
        for _ in range(1, n_samples):
            next_index = np.argmax(distances)
            selected_indices.append(next_index)
            distances = np.minimum(distances, np.linalg.norm(X - X[next_index], axis=1))
        
        return selected_indices
