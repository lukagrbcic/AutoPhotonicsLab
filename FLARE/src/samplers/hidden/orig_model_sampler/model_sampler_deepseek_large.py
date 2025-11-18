
import numpy as np
from indago import PSO
import random
from scipy.spatial import KDTree
from scipy.stats import qmc

np.random.seed(random.randint(0, 1000000))

class modelSampler:
    def __init__(self, model, sample_size, lb, ub, algorithm, 
                 function='uncertainty', 
                 penalty_weight=20000.0,  # Increased base penalty
                 penalty_exponent=4.0,    # Stronger distance-based scaling
                 uncertainty_aggregation='variance_sum',
                 inertia_weight=0.75,      # Slightly lower inertia for better convergence
                 cognitive_coefficient=2.8,
                 social_coefficient=2.8,
                 swarm_size=5000,         # Larger swarm for thorough exploration
                 max_evaluations=10000,    # More evaluations per PSO run
                 diversity_perturbation=0.3,  # Larger perturbation step
                 post_processing_candidates=100,  # More candidates for perturbation
                 threshold_factor=0.15,    # Moderate spacing threshold
                 min_distance_threshold=0.03):  # Minimum allowed distance
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function
        self.penalty_weight = penalty_weight
        self.penalty_exponent = penalty_exponent
        self.max_distance = np.linalg.norm(ub - lb)
        self.swarm_size = swarm_size
        self.max_evaluations = max_evaluations
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.uncertainty_aggregation = uncertainty_aggregation
        self.diversity_perturbation = diversity_perturbation
        self.threshold_factor = threshold_factor
        self.min_distance_threshold = min_distance_threshold  # Absolute minimum distance

    def get_samples(self):
        X = []
        f = []
        for i in range(self.sample_size):
            def get_values(x):
                preds = np.concatenate([m.predict([x]) for m in self.model.estimators_])
                
                # Enhanced uncertainty calculation
                std = np.std(preds, axis=0)
                if self.uncertainty_aggregation == 'variance_sum':
                    uncertainty = np.sum(std**2)
                else:
                    uncertainty = np.max(std)
                
                # Advanced diversity penalty with cubic scaling
                if len(X) > 0:
                    distances = np.linalg.norm(np.array(X) - x, axis=1)
                    min_distance = np.min(distances)
                    min_distance = max(min_distance, self.min_distance_threshold)
                else:
                    min_distance = self.max_distance
                
                # Progress-based penalty scaling with inverse cubic distance
                progress = (i + 1) / self.sample_size
                current_penalty = self.penalty_weight * (progress ** self.penalty_exponent)
                penalty = current_penalty / (min_distance ** 3)  # Cubic distance penalty
                
                return - (uncertainty - penalty)  # Minimize this to maximize (uncertainty - penalty)

            optimizer = PSO()
            optimizer.evaluation_function = get_values
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = self.max_evaluations
            optimizer.swarm_size = self.swarm_size
            optimizer.inertia_weight = self.inertia_weight
            optimizer.cognitive_coefficient = self.cognitive_coefficient
            optimizer.social_coefficient = self.social_coefficient

            # Enhanced initialization with collision avoidance and uncertainty bias
            def lhs_jitter(n, dims, lb, ub, jitter=0.05, existing_points=None):
                sampler = qmc.LatinHypercube(d=dims, seed=random.randint(0, 1000000))
                sample = sampler.random(n)
                sample += np.random.uniform(-jitter, jitter, sample.shape)
                sample = np.clip(lb + (ub - lb) * sample, lb, ub)
                
                if existing_points is not None and len(existing_points) > 0:
                    tree = KDTree(existing_points)
                    distances, _ = tree.query(sample)
                    # Remove points too close to existing samples
                    valid = distances > self.min_distance_threshold
                    sample = sample[valid]
                    # Re-sample if needed
                    while len(sample) < n:
                        new_sample = sampler.random(n - len(sample))
                        new_sample += np.random.uniform(-jitter, jitter, new_sample.shape)
                        new_sample = np.clip(lb + (ub - lb) * new_sample, lb, ub)
                        new_distances, _ = tree.query(new_sample)
                        valid_new = new_distances > self.min_distance_threshold
                        sample = np.vstack([sample, new_sample[valid_new]])
                        sample = sample[:n]
                
                # Bias towards high-uncertainty regions
                if len(existing_points) > 0:
                    uncertainties = [np.std(np.concatenate([m.predict([pt]) for m in self.model.estimators_]), axis=0).sum() 
                                    for pt in sample]
                    top_indices = np.argsort(uncertainties)[-int(n*0.7):]
                    sample = sample[top_indices]
                    sample = np.concatenate([sample, sampler.random(n - len(sample))])
                    sample = np.clip(lb + (ub - lb)*sample, lb, ub)
                
                return sample[:n]

            existing_points = X if X else None
            optimizer.particles = lhs_jitter(self.swarm_size, len(self.lb), self.lb, self.ub, existing_points=existing_points)

            result = optimizer.optimize()
            X.append(result.X)
            f.append(result.f)

        X = np.array(X)
        f = np.array(f)
        sorted_indices = np.argsort(f)
        
        # Advanced post-processing with gradient-based perturbations
        X = X[sorted_indices]
        threshold = max(self.threshold_factor * self.max_distance, self.min_distance_threshold)
        
        tree = KDTree(X)
        min_dists = tree.query(X, k=2)[0][:,1]  # Distance to nearest neighbor
        
        for j in range(len(X)):
            if min_dists[j] < threshold:
                current_point = X[j]
                candidates = []
                
                # Compute uncertainty gradient
                def compute_uncertainty(x):
                    preds = np.concatenate([m.predict([x]) for m in self.model.estimators_])
                    std = np.std(preds, axis=0)
                    return np.sum(std**2) if self.uncertainty_aggregation == 'variance_sum' else np.max(std)
                
                delta = 1e-5
                gradient = np.zeros_like(current_point)
                for d in range(len(current_point)):
                    x_plus = current_point.copy()
                    x_plus[d] += delta
                    x_minus = current_point.copy()
                    x_minus[d] -= delta
                    grad_d = (compute_uncertainty(x_plus) - compute_uncertainty(x_minus)) / (2*delta)
                    gradient[d] = grad_d
                
                for _ in range(self.post_processing_candidates):
                    # Find nearest neighbors
                    dists, idx = tree.query([current_point], k=4)
                    neighbors = X[idx[0][1:]]  # Exclude self
                    
                    # Compute repulsion direction
                    repulsion_dir = np.zeros_like(current_point)
                    for neighbor in neighbors:
                        vec = current_point - neighbor
                        norm = np.linalg.norm(vec) + 1e-6
                        repulsion_dir += vec / (norm ** 3)  # Stronger repulsion
                    
                    # Combine with gradient direction
                    combined_dir = repulsion_dir + gradient
                    if np.linalg.norm(combined_dir) == 0:
                        combined_dir = np.random.normal(0, 1, len(current_point))
                    combined_dir /= np.linalg.norm(combined_dir)
                    
                    # Add exploration noise
                    noise = np.random.normal(0, 0.15, current_point.shape)
                    combined_dir += noise
                    combined_dir /= np.linalg.norm(combined_dir)
                    
                    step_size = self.diversity_perturbation * dists[0][1]  # Scale with nearest distance
                    candidate = current_point + combined_dir * step_size
                    candidate = np.clip(candidate, self.lb, self.ub)
                    
                    # Recompute metrics
                    new_unc = compute_uncertainty(candidate)
                    new_dists = tree.query([candidate], k=len(X))[0]
                    new_min_dist = max(np.min(new_dists[1:]), self.min_distance_threshold)
                    new_penalty = self.penalty_weight / (new_min_dist ** 3)
                    
                    score = new_unc - new_penalty
                    candidates.append((score, candidate))
                
                if candidates:
                    best_score = max(c[0] for c in candidates)
                    best_candidate = [c[1] for c in candidates if c[0] == best_score][0]
                    X[j] = best_candidate
        
        return X
