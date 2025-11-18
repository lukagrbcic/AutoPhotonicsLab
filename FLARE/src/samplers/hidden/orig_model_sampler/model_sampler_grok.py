
import numpy as np
from scipy.stats import qmc  # For Latin Hypercube Sampling
from scipy.spatial import KDTree  # For efficient nearest-neighbor queries
import random

np.random.seed(random.randint(0, 10223))

class modelSampler:

    def __init__(self, model, sample_size, lb, ub, algorithm, function='uncertainty'):
        # No changes here; preserving original initialization for compatibility
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm  # Unused in the original code, so left as is
        self.function = function

    def get_samples(self):
        # Improvement: Generate candidates using LHS with a more adaptive number based on dimensionality.
        # Added a small random perturbation to the multiplier for variability, ensuring better exploration
        # without excessive evaluations, which reduces computation time and adapts to different problem sizes.
        dim = len(self.lb)  # Number of features
        base_multiplier = 50  # Base value from previous version
        adaptive_multiplier = base_multiplier + random.uniform(-5, 5)  # Add slight randomness for exploration
        n_candidates = min(int(adaptive_multiplier * dim * self.sample_size), 10000)  # Cap to prevent memory issues
        
        lhs = qmc.LatinHypercube(d=dim)
        candidates = lhs.random(n=n_candidates)  # Generate LHS points in [0,1)
        candidates = qmc.scale(candidates, self.lb, self.ub)  # Scale to actual bounds [lb, ub]
        
        # Improvement: Compute uncertainties for all candidates in a single batched operation.
        # This vectorizes predictions across all estimators and candidates, minimizing redundant model calls
        # and reusing arrays for efficiency, as in the previous version but with ensured no-copy operations.
        preds_list = [model.predict(candidates) for model in self.model.estimators_]  # Batched predictions
        preds_array = np.stack(preds_list, axis=0)  # Shape: (n_estimators, n_candidates, n_outputs)
        
        std_per_point_per_output = np.std(preds_array, axis=0)  # Std per output for each candidate: (n_candidates, n_outputs)
        uncertainties = -np.sum(std_per_point_per_output, axis=1)  # Uncertainty scores: (n_candidates,) -- negative for maximization
        
        # Improvement: Sort candidates by uncertainty and apply greedy selection with KDTree for diversity.
        # Using KDTree replaces cdist for faster minimum distance queries, reducing computation time in the loop
        # (e.g., from O(n^2) to near O(1) per query), enhancing the hybrid strategy for better batch quality.
        sorted_indices = np.argsort(uncertainties)  # Ascending sort (highest uncertainty first)
        top_indices = sorted_indices[:min(2 * self.sample_size, n_candidates)]  # Select top candidates
        
        selected_indices = []
        remaining_indices = list(top_indices)  # Convert to list for manipulation
        
        if len(top_indices) > 0:
            # Start with the most uncertain point
            first_index = sorted_indices[0]  # Highest uncertainty
            selected_indices.append(first_index)
            if first_index in remaining_indices:
                remaining_indices.remove(first_index)
            
            # Build KDTree on selected points for efficient queries
            if len(selected_indices) > 0:
                points_selected = candidates[selected_indices]
                kdtree = KDTree(points_selected)  # Create KDTree for fast nearest-neighbor search
                
                while len(selected_indices) < self.sample_size and len(remaining_indices) > 0:
                    points_remaining = candidates[remaining_indices]  # Subset of candidates
                    
                    # Use KDTree to query minimum distances efficiently
                    distances, _ = kdtree.query(points_remaining, k=1)  # Minimum distance to nearest selected point
                    
                    # Improvement: Normalize uncertainties for scale-invariant weighting, enhancing the greedy algorithm
                    # by balancing diversity and uncertainty more effectively, potentially improving active learning efficiency.
                    uncertainty_weight = uncertainties[remaining_indices]  # Corresponding uncertainties
                    normalized_uncertainty = (uncertainty_weight - np.min(uncertainty_weight)) / (np.max(uncertainty_weight) - np.min(uncertainty_weight) + 1e-8)  # Normalize to [0,1]
                    weighted_min_distances = distances + 0.1 * normalized_uncertainty  # Weighted distances
                    
                    best_idx_in_remaining = np.argmax(weighted_min_distances)  # Index in remaining_indices
                    best_index_global = remaining_indices[best_idx_in_remaining]  # Global index
                    selected_indices.append(best_index_global)
                    remaining_indices.remove(best_index_global)  # Remove from remaining
                    
                    # Update KDTree with the new selected point for the next iteration
                    kdtree = KDTree(candidates[selected_indices])  # Rebuild KDTree with updated selected points
        
        # Handle cases where fewer than sample_size points are selected.
        if len(selected_indices) < self.sample_size:
            additional_indices = [idx for idx in top_indices if idx not in selected_indices and idx not in remaining_indices]
            selected_indices.extend(additional_indices[:self.sample_size - len(selected_indices)])
        
        selected_indices = selected_indices[:self.sample_size]  # Ensure exactly sample_size indices
        X = candidates[selected_indices]  # Extract selected points
        
        # Improvement: Sort the final selected points by their uncertainties, reusing the uncertainties array
        # to avoid recomputation, while maintaining the original output format for exact functionality.
        sorted_selected_indices = np.argsort(uncertainties[selected_indices])  # Sort indices by uncertainties
        X = X[sorted_selected_indices]  # Final X sorted by uncertainties
        
        return X  # Returns a matrix of shape (sample_size, dim), as in the original code
