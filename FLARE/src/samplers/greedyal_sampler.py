import numpy as np
from indago import PSO
import random
from sklearn.ensemble import RandomForestRegressor
from _PSO import PSO

class greedyALSampler:
    
    def __init__(self, X, y, sample_size, lb, ub):
        
        self.X = X
        self.y = y
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub

        self.model = RandomForestRegressor().fit(self.X, self.y)

    def gen_GAL_samples(self):
        print ('running original code!')
        
        X = []
        f = []
        for i in range(self.sample_size):
            
            def get_values(X_population):
                """
                Batched objective for PSO.
                X_population: np.ndarray shape (n_particles, n_features)
                Returns: np.ndarray shape (n_particles,)
                """
                X_population = np.atleast_2d(X_population)


                all_preds = np.array([
                    tree.predict(X_population) for tree in self.model.estimators_
                ])

                stds_across_trees = np.std(all_preds, axis=0)  # shape: (n_particles, horizon_length)

                particle_fitness = -np.sum(stds_across_trees, axis=1)  # shape: (n_particles,)

                assert particle_fitness.shape == (X_population.shape[0],), \
                    f"Shape mismatch: got {particle_fitness.shape} expected {(X_population.shape[0],)}"

                return particle_fitness

            opt = PSO(function=get_values, lb=self.lb, ub=self.ub, swarm_size=10, max_evals=100, device='cpu')
            min_x, _, min_f = opt.search()


            min_x = np.ravel(min_x)

            X.append(min_x)
            f.append(min_f[0])
                                
        X = np.array(X)
        f = np.array(f)
        
        X = X[np.argsort(f)]

        return X, self.model
            
            
        
        
