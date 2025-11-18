import numpy as np
import math

class blackBoxEvaluator:
    def __init__(self, function_name='rastrigin', dim=5,
                 test_data_size=1000, seed=42,
                 duplicate_outputs=True, n_objectives=2):
        self.function_name = function_name.lower()
        self.dim = dim
        self.test_data_size = test_data_size
        self.seed = seed
        self.duplicate_outputs = duplicate_outputs  # Applies to Rastrigin
        self.n_objectives = n_objectives            # For DTLZ and possibly other MOO funcs

    # --------------------
    # Functions
    # --------------------
    def _rastrigin(self, X):
        """Rastrigin function with optional duplicated outputs."""
        n_features = X.shape[1]
        A = 10
        y = A * n_features + np.sum(X**2 - A * np.cos(2 * math.pi * X), axis=1)
        y = y.reshape(-1, 1)
        if self.duplicate_outputs:
            y = np.hstack([y] * 2)  # duplicate into 2 identical columns
        return y

    def _zdt1(self, X):
        """ZDT1 multi-objective function: always returns (n_samples, 2)."""
        f1 = X[:, 0]
        g = 1 + 9 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    def _dtlz1(self, X):
        """
        DTLZ1 scalable multi-objective function.
        Returns (n_samples, n_objectives).
        Domain: [0, 1]^dim
        """
        M = self.n_objectives
        n = X.shape[1]
        k = n - M + 1
        g = 100 * (k + np.sum((X[:, M-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, M-1:] - 0.5)),
                              axis=1))
        F = []
        for m in range(1, M + 1):
            f = 0.5 * (1 + g)
            for i in range(1, M - m + 1):
                f *= X[:, i-1]
            if m > 1:
                f *= (1 - X[:, M - m])
            F.append(f)
        return np.column_stack(F)  # shape: (n_samples, M)

    # --------------------
    # Helpers
    # --------------------
    def get_bounds(self):
        if self.function_name == 'rastrigin':
            lb = np.ones(self.dim) * -5.12
            ub = np.ones(self.dim) * 5.12
        elif self.function_name in ['zdt1', 'dtlz1']:
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)
        else:
            raise ValueError(f"Bounds not defined for {self.function_name}")
        return lb, ub

    def get_test_data(self):
        lb, ub = self.get_bounds()
        rng = np.random.default_rng(self.seed)
        X_test = rng.uniform(low=lb, high=ub, size=(self.test_data_size, self.dim))
        y_test = self.evaluate(X_test)
        return X_test, y_test

    def evaluate(self, X):
        X = np.atleast_2d(X)
        if self.function_name == 'rastrigin':
            return self._rastrigin(X)
        elif self.function_name == 'zdt1':
            return self._zdt1(X)
        elif self.function_name == 'dtlz1':
            return self._dtlz1(X)
        else:
            raise ValueError(f"Function '{self.function_name}' not implemented.")




#evaluator = blackBoxEvaluator(function_name='rastrigin', dim=5, test_data_size=10)
#lb, ub = evaluator.get_bounds()
#X_test, y_test = evaluator.get_test_data()
#test_data = (X_test, y_test)

#print (X_test, y_test)
