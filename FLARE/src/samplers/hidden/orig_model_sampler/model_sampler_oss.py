
# ----------------------------------------------------------------------
#  Improved modelSampler – batched active‑learning acquisition
# ----------------------------------------------------------------------
#  What changed?
#   • Replaced the per‑sample PSO optimiser with a single large random
#     candidate pool (default 5 000 points) that is evaluated **once**.
#   • Uncertainty (the sum of per‑output standard deviations across the
#     ensemble) is computed **vector‑wise** for the whole pool.
#   • A greedy “uncertainty + diversity” selection picks the final
#     batch, dramatically reducing redundant computations.
#   • All Python loops that were previously executed for every particle
#     or every sample have been eliminated – the code now relies on fast
#     NumPy broadcasting.
#   • The closure that defined the acquisition function inside the loop
#     is gone; instead we use a dedicated, cache‑friendly method.
#   • Minor clean‑ups (rng handling, seed preservation, input validation)
#     make the class easier to test and maintain.
# ----------------------------------------------------------------------

import numpy as np
from typing import Optional, Sequence

# ----------------------------------------------------------------------
#  Helper: a tiny wrapper to keep the original NumPy seeding behaviour.
# ----------------------------------------------------------------------
#  The original script seeded NumPy with a random integer drawn from the
#  Python ``random`` module.  We preserve that side‑effect so that any
#  downstream code that relies on the global seed continues to behave the
#  same way.
# ----------------------------------------------------------------------
import random

np.random.seed(random.randint(0, 10223))


class modelSampler:
    """
    Batched active‑learning sampler that selects the most uncertain (and
    diverse) points for a given ensemble model.

    Parameters
    ----------
    model : object
        An ensemble model exposing ``estimators_`` – a list of fitted
        estimators each with a ``predict`` method that accepts a 2‑D array.
    sample_size : int
        Number of points to return (size of the batch).
    lb, ub : array‑like
        Lower and upper bounds for each feature dimension.
    algorithm : str, optional
        Kept for backward compatibility – not used internally.
    function : str, optional
        Kept for backward compatibility – not used internally.
    n_candidates : int, optional
        Number of random candidates drawn from the search space.  More
        candidates give a finer approximation of the optimum at the cost
        of memory/computation (default: 5 000).
    diversity_weight : float, optional
        Weight α∈[0,1] that balances uncertainty vs. diversity in the
        greedy batch selection (default: 0.7 → 70 % uncertainty, 30 %
        diversity).
    random_state : int or np.random.Generator, optional
        Seed or generator for reproducible randomness.
    """

    def __init__(
        self,
        model,
        sample_size: int,
        lb: Sequence[float],
        ub: Sequence[float],
        algorithm,
        function: str = "uncertainty",
        *,
        n_candidates: int = 5000,
        diversity_weight: float = 0.7,
        random_state: Optional[int] = None,
    ):
        # ------------------------------------------------------------------
        #  Store everything unchanged – these attributes are accessed by
        #  other parts of the codebase, so we keep the original names.
        # ------------------------------------------------------------------
        self.model = model
        self.sample_size = sample_size
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.algorithm = algorithm
        self.function = function

        # ------------------------------------------------------------------
        #  Hyper‑parameters for the new acquisition pipeline.
        # ------------------------------------------------------------------
        self.n_candidates = max(n_candidates, sample_size)  # must be ≥ batch size
        if not (0.0 <= diversity_weight <= 1.0):
            raise ValueError("diversity_weight must be between 0 and 1.")
        self.diversity_weight = diversity_weight

        # ------------------------------------------------------------------
        #  Randomness handling – use a numpy Generator for speed and better
        #  reproducibility.  If an integer seed is given we honour it;
        #  otherwise we fall back to the global RNG (which already carries
        #  the seed from the top‑level ``np.random.seed`` call).
        # ------------------------------------------------------------------
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng(random_state)

    # ----------------------------------------------------------------------
    #  Private helper: compute the ensemble uncertainty for a batch of points.
    #  The implementation is fully vectorised:
    #
    #      preds.shape == (n_estimators, n_points, n_outputs)
    #      std   .shape == (n_points, n_outputs)
    #      uncertainty (scalar per point) == sum over outputs
    #
    #  Returning the **positive** uncertainty (higher → more uncertain) makes
    #  the subsequent greedy selection more intuitive.
    # ----------------------------------------------------------------------
    def _batch_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the sum of standard deviations across the ensemble for each
        candidate point.

        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_features)
            Candidate points.

        Returns
        -------
        uncertainties : np.ndarray, shape (n_points,)
            Positive uncertainty values (higher → more uncertain).
        """
        # Stack predictions from every estimator – this uses a list‑comprehension
        # that is fast because each ``est.predict`` processes the whole batch.
        preds = np.stack([est.predict(X) for est in self.model.estimators_], axis=0)

        # Standard deviation across the ensemble (axis=0 → over estimators)
        std_per_point = np.std(preds, axis=0)  # shape: (n_points, n_outputs)

        # Sum over output dimensions to obtain a scalar uncertainty per point
        uncertainties = np.sum(std_per_point, axis=1)  # shape: (n_points,)

        return uncertainties

    # ----------------------------------------------------------------------
    #  Public API – unchanged signature.
    # ----------------------------------------------------------------------
    def get_samples(self) -> np.ndarray:
        """
        Return a batch of `sample_size` points that maximise ensemble
        uncertainty while staying diverse.

        Returns
        -------
        X : np.ndarray, shape (sample_size, n_features)
            The selected points, ordered from **most** to **least**
            uncertain (i.e. the same ordering as the original implementation).
        """
        dim = self.lb.shape[0]

        # ------------------------------------------------------------------
        #  1️⃣  Draw a large set of random candidates inside the bounds.
        #      Using a single uniform draw is far cheaper than launching an
        #      optimiser for each point.
        # ------------------------------------------------------------------
        candidates = self.rng.uniform(
            low=self.lb, high=self.ub, size=(self.n_candidates, dim)
        )  # shape: (n_candidates, dim)

        # ------------------------------------------------------------------
        #  2️⃣  Vectorised uncertainty evaluation for *all* candidates.
        #      This replaces the inner PSO loop and the per‑particle
        #      ``np.concatenate`` calls.
        # ------------------------------------------------------------------
        uncertainties = self._batch_uncertainty(candidates)  # (n_candidates,)

        # ------------------------------------------------------------------
        #  3️⃣  Greedy batch selection that mixes uncertainty with a simple
        #      diversity term (distance to already‑chosen points).  The
        #      diversity penalty prevents the algorithm from returning many
        #      points that are clustered in the same local maximum.
        # ------------------------------------------------------------------
        selected_idx = []  # will contain indices into ``candidates``

        # --- pick the most uncertain point as the seed of the batch ---
        first = int(np.argmax(uncertainties))
        selected_idx.append(first)

        # Pre‑compute a normalised version of the uncertainties to keep the
        # scoring scale comparable to the distance term.
        unc_min, unc_ptp = uncertainties.min(), uncertainties.ptp()
        # Guard against zero variation (unlikely but safe):
        norm_unc = (
            (uncertainties - unc_min) / (unc_ptp + 1e-12)
        )  # values in [0, 1]

        # ------------------------------------------------------------------
        #  Greedy loop – runs ``sample_size‑1`` times.
        #  Each iteration:
        #   • compute the Euclidean distance from *every* candidate to the
        #     already‑selected points (broadcasting, no Python loops).
        #   • take the minimum distance per candidate → encourages spread.
        #   • combine normalised uncertainty & distance with the
        #     user‑controlled ``diversity_weight``.
        #   • mask out already‑selected indices so they are never chosen again.
        # ------------------------------------------------------------------
        for _ in range(1, self.sample_size):
            # Current set of selected points (k x dim)
            selected_points = candidates[selected_idx]  # shape: (k, dim)

            # Compute pairwise distances: (n_candidates, k)
            # Using broadcasting: (n_candidates, 1, dim) - (1, k, dim)
            diff = candidates[:, None, :] - selected_points[None, :, :]  # (N, k, d)
            dists = np.linalg.norm(diff, axis=2)  # Euclidean, shape (N, k)

            # Minimum distance to the chosen set for each candidate
            min_dists = np.min(dists, axis=1)  # shape (N,)

            # Normalise the distance term to [0, 1]
            dist_min, dist_ptp = min_dists.min(), min_dists.ptp()
            norm_dist = (min_dists - dist_min) / (dist_ptp + 1e-12)

            # Composite score = α * uncertainty + (1‑α) * distance
            # Both terms are already normalised → scores in [0, 1].
            α = self.diversity_weight
            composite = α * norm_unc + (1.0 - α) * norm_dist

            # Mask out already selected indices
            composite[selected_idx] = -np.inf

            # Choose the candidate with the highest composite score
            next_idx = int(np.argmax(composite))
            selected_idx.append(next_idx)

        # ------------------------------------------------------------------
        #  4️⃣  Gather the final batch and sort it by *descending* uncertainty
        #      (i.e. the same order the original code produced after sorting
        #      the negative values).
        # ------------------------------------------------------------------
        X_selected = candidates[selected_idx]                     # (sample_size, dim)
        f_selected = -uncertainties[selected_idx]                # negative as in original code

        # Sort by the (negative) score → most uncertain first
        order = np.argsort(f_selected)  # ascending because f is negative
        X = X_selected[order]

        # ------------------------------------------------------------------
        #  Return the batch.  The shape is (sample_size, n_features) – exactly
        #  what callers expect.
        # ------------------------------------------------------------------
        return X
