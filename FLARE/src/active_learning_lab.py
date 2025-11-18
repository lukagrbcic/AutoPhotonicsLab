import numpy as np
import sys
from scipy.stats import qmc


sys.path.insert(0, 'samplers')
sys.path.insert(1, 'models')

# import check_accuracy as ca
from generate_samples import samplers
import optimize_model as opt
import optimize_model_de as opt_de
from ensemble_regressor import EnsembleRegressor
import warnings
warnings.filterwarnings("ignore")


class activeLearnerLab:
    
    
    def __init__(self, X_init, y_init,
                       lb, ub,
                       batch_size,
                       sampler,
                       init_size=None,
                       verbose=0,
                       ):
        
        self.X_init = X_init
        self.y_init = y_init
        self.lb = lb
        self.ub = ub
        self.init_size = init_size
        self.batch_size = batch_size
        self.sampler = sampler
        self.verbose = verbose

        self.model = None
        

    def initialize(self):

        X = samplers('lhs', self.init_size, self.lb, self.ub).generate_samples()
        
        return X
        
    def get_samples(self, X_new, y_new, sampled_points=[]):
               
        X, model = samplers(self.sampler, self.batch_size,
                     self.lb, self.ub, sampled_points, X_new, y_new).generate_samples()
        
        return X, model
        
    def run(self):
        print ('run')
        
        X = self.X_init
        y = self.y_init
        
        X_new, model = self.get_samples(X, y, sampled_points=X)

        
        return X_new, model


        
        
        
        
        
        
        
        
        
        
        
