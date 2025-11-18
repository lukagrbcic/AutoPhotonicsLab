import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import torch





class PSO:
    
    def __init__(self, function, lb, ub, swarm_size, max_evals,
                 c1=1., c2=1., w=0.72,  initial_X=[], initial_fX=[], device='cpu'):
        
        self.function = function
        self.lb = lb
        self.ub = ub
        self.swarm_size = swarm_size
        self.max_evals = max_evals
        self.c1 = c1 
        self.c2 = c2 
        self.w = w
        self.initial_X = initial_X
        self.initial_fX = initial_fX
        self.device = device
        
        if self.device == 'cuda':
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        else:
            self.device = torch.device('cpu')
        
        self.dtype = torch.float32
                
        self.lower = torch.as_tensor(self.lb, device=self.device, dtype=self.dtype)
        self.upper = torch.as_tensor(self.ub, device=self.device, dtype=self.dtype)
        self.initial_X = torch.as_tensor(self.initial_X, device=self.device, dtype=self.dtype)
        self.D = self.lower.shape[0]
        
        
        """checks"""
        
        if self.lb.shape != self.ub.shape:
            raise ValueError("Lower and upper boundaries are not the same shape!")
            
        if self.initial_X.shape[0] > self.swarm_size:
            raise ValueError("Initial design vector is larger than the population size!")
            
    def _evaluate(self, X):
        
        f = torch.as_tensor(self.function(X.cpu().numpy()), device=self.device, dtype=self.dtype)       
        
        return f.unsqueeze(1)

    def _initialize(self):
        
        if len(self.initial_X) == 0:
            
            X = torch.rand((self.swarm_size, *self.lower.shape), device=self.device, dtype=self.dtype) * (self.upper - self.lower) + self.lower
                   
        else:
            
            X = torch.clamp(self.initial_X, min=self.lower, max=self.upper)
            
            if X.shape[0] < self.swarm_size:
                X = torch.vstack((X, torch.rand((self.swarm_size-X.shape[0], *self.lower.shape), \
                                               device=self.device, dtype=self.dtype) * (self.upper - self.lower) + self.lower))            
        
        return X
    
    def _personal_best(self, X, fX, P=None, fP=None):
        
        if P is None:
            
            P = X.clone()
            fP = fX.clone()
        
        else:
            
            indx_ = torch.where(fX <= fP)[0]
            P[indx_] = X[indx_]
            fP[indx_] = fX[indx_]
        
        return P, fP
        
    def _velocity_update(self, X, fX, P, G, V):
        
        r1 = torch.rand((self.swarm_size, 1), device=self.device, dtype=self.dtype)
        r2 = torch.rand((self.swarm_size, 1), device=self.device, dtype=self.dtype)


        V = self.w*V + self.c1*r1*(P - X) + self.c2*r2*(G-X)
        
        return V
        
    def _position_update(self, X, V):
        
        X = X + V
        
        X = torch.clamp(X, self.lower, self.upper)
        
        return X
    
    @torch.no_grad() 
    def search(self):
        
        X = self._initialize()
        if len(self.initial_fX) == 0:
            fX = self._evaluate(X)
        else:
            fX = self.initial_fX
        P, fP = self._personal_best(X, fX)     
        V = torch.rand((self.swarm_size, *self.lower.shape), \
                       device=self.device, dtype=self.dtype) * (self.upper - self.lower) + self.lower


        evals = fX.shape[0]
        
        while evals <= self.max_evals - fX.shape[0]:
            #print ('Evals:', evals, '->', 'Best solution fitness:', fP[torch.argmin(fP).item()].item())

            G = P[torch.argmin(fP)]
            V = self._velocity_update(X, fX, P, G, V)
            X = self._position_update(X, V)
            
            fX = self._evaluate(X)
            P, fP = self._personal_best(X, fX, P, fP)
            
            evals = evals + fX.shape[0]

            
        f_best_indx = torch.argmin(fP).item()

        
        return X[f_best_indx].cpu().numpy(), P.cpu().numpy(), fP[f_best_indx].cpu().numpy()
        









