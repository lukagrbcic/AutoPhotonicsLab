
import numpy as np
from indago import PSO
import random

np.random.seed(random.randint(0, 10223))

class modelSampler:

    def __init__(self, model, sample_size, lb, ub, algorithm, function='uncertainty'):

        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function

    def get_samples(self):

        X = []
        f = []
        max_possible_distance = np.linalg.norm(np.array(self.ub) - np.array(self.lb))  # Precompute for normalization

        for i in range(self.sample_size):

            def get_values(x):

                # Compute uncertainty (standard deviation of ensemble predictions)
                preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
                uncertainty = np.sum(np.std(preds, axis=0))

                # Compute diversity term (normalized minimum distance to existing points)
                if len(X) > 0:
                    existing_points = np.array(X)
                    distances = np.linalg.norm(existing_points - x, axis=1)
                    min_distance = np.min(distances)
                    normalized_distance = min_distance / (max_possible_distance + 1e-5)
                else:
                    normalized_distance = 0.0

                # Combine uncertainty and diversity with weighting
                # The objective is to maximize (uncertainty + normalized_distance)
                # Return negative for minimization in PSO
                return -(uncertainty + normalized_distance)

            optimizer = PSO()
            optimizer.evaluation_function = get_values
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = 100  # Consider increasing this for better convergence

            result = optimizer.optimize()
            min_x = result.X
            min_f = result.f

            X.append(min_x)
            f.append(min_f)

        X = np.array(X)
        f = np.array(f)

        # Sort by f (lowest values correspond to highest objective values)
        X = X[np.argsort(f)]

        return X
