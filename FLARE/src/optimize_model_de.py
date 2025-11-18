from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neural_network import MLPRegressor
from ensemble_regressor import EnsembleRegressor
from scipy.optimize import differential_evolution
import sys



from indago import PSO

import pandas as pd

import numpy as np
import warnings
warnings.filterwarnings("ignore")

class optimizeDE:
    
    def __init__(self, algorithm, X, y, n_iter=10, cv=3, ensemble_size=10):
        
        self.algorithm = algorithm
        self.X = X
        self.y = y
        self.n_iter = n_iter

        

    def get_param_bounds(self):
        """Define the bounds for differential evolution"""
        if 'rf' in self.algorithm[0]:
            param_bounds = [
                (50, 450),     # n_estimators
                (1, 30),       # max_depth
                (2, 20),       # min_samples_split
                (1, 20),       # min_samples_leaf
                (0.1, 1.0),    # max_features
                (0, 1)         # bootstrap (0=False, 1=True)
            ]
            
        return param_bounds

    def objective_function(self, params, X=None, y=None):

        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        if 'rf' in self.algorithm[0]:

           # avg_error = []
           # for i in range(2):
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=23)
    
            n_estimators = int(params[0])
            max_depth = int(params[1])
            min_samples_split = int(params[2])
            min_samples_leaf = int(params[3])
            max_features = params[4]
            bootstrap = bool(round(params[5]))
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
            ).fit(X_train, y_train)
            
            
            # tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
            # q95, q5 = np.percentile(tree_preds, [95, 5], axis=0)
            # iqr = q95 - q5
            
 
            # mean_iqr = np.mean(np.mean(iqr, axis=0))
                            
            # err = mean_iqr
            
        
            y_pred = model.predict(X_test)
            
            err = -np.sqrt(mean_squared_error(y_test, y_pred))
                
            return -err
        
    def search_de(self):
        """Perform hyperparameter optimization using differential evolution"""
        result = differential_evolution(
            func=self.objective_function,
            bounds=self.get_param_bounds(),
            strategy='best1bin',
            maxiter=self.n_iter,
            popsize=10,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            #seed=42,
            disp=True,
            workers=-1
        )
        
        return result
    
    def get_hyperparameters(self):
        if self.algorithm[0] == 'rf':
            self.ensemble_size = 1
            
            result = self.search_de()
            
            best_params = result.x
            
            n_estimators = int(best_params[0])
            max_depth = int(best_params[1])
            min_samples_split = int(best_params[2])
            min_samples_leaf = int(best_params[3])
            max_features = best_params[4]
            bootstrap = bool(round(best_params[5]))
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap
            )
            
            return model
        
        
        
        
        
        
        
        
        
        
        
        
    