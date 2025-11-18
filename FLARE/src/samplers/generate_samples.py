import lhs_sampler as lhs
import random_sampler as rnd
import greedyfp_sampler as gfp
import bc_sampler as bc
import model_sampler as ms
import greedyal_sampler as gal
import numpy as np


class samplers:
    
    def __init__(self, sampler, batch_size, lb, ub, sampled_points=[], X_=[], y_=[]):
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.lb = lb
        self.ub = ub
        self.sampled_points = sampled_points
        self.X_ = X_
        self.y_ = y_
    
    def generate_samples(self, seed=None):
        
        if self.sampler == 'lhs':
            
            X = lhs.lhsSampler(self.batch_size, self.lb, self.ub).gen_LHS_samples()

            return X
            
        elif self.sampler == 'random':
            
            X, model = rnd.randomSampler(self.X_, self.y_, self.batch_size, self.lb, self.ub).gen_random_samples()

            return X, model

        elif self.sampler == 'greedyfp':
            
            X, model = gfp.greedyFPSampler(self.X_, self.y_, self.batch_size, self.lb, self.ub).gen_GFP_samples()

            return X, model

        elif self.sampler == 'bc':
            
            X, model = bc.bcSampler(self.X_, self.y_, self.batch_size, self.lb, self.ub).gen_BC_samples()

            return X, model
        
        elif self.sampler.split('_')[0] == 'model':
            
            X, model = ms.modelSampler(self.X_, self.y_, self.batch_size,
                                self.lb, self.ub).gen_samples()
            
            return X, model


        elif self.sampler.split('_')[0] == 'greedyAL':

            X, model = gal.greedyALSampler(self.X_, self.y_, self.batch_size,
                                self.lb, self.ub).gen_GAL_samples()

            return X, model
        
    
    
    
        
