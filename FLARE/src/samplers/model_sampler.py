
import numpy as np
try:
    # SciPy ≥ 1.7 provides a Sobol low‑discrepancy generator (deterministic).
    from scipy.stats import qmc
    _HAS_SOBOL = True
except Exception:                     # SciPy not available → safe fallback
    _HAS_SOBOL = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KDTree


class modelSampler:
    """
    Deterministic batched active‑learning sampler.

    * Keeps the exact public API (`modelSampler`, `__init__`, `gen_samples`).
    * Uses a Sobol sequence (or a simple deterministic fallback) to build a
      candidate pool – **no random numbers are drawn at run‑time**.
    * Uncertainty = variance of the RandomForest’s tree predictions.
    * Diversity   = distance to the nearest already‑labelled point (KD‑Tree).
    * A single weighted score (α ≈ 0.7) selects the most informative batch.
    * The forest is trained with ``n_jobs=1`` – this avoids the
      ``BrokenProcessPool`` error that appears when a process‑pool‑based
      evaluation nests another parallel job (the default ``n_jobs=-1``).
    """

    # ------------------------------------------------------------------
    # 1️⃣  Construction – store data, validate bounds, fit a deterministic RF
    # ------------------------------------------------------------------
    def __init__(self, X, y, sample_size, lb, ub):
        """
        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
            Already labelled feature matrix.
        y : array‑like, shape (n_samples,) or (n_samples, n_targets)
            Corresponding targets.
        sample_size : int
            Number of points to propose per call to ``gen_samples``.
        lb, ub : array‑like, shape (n_features,)
            Lower / upper bounds for each feature dimension.
        """
        # ---- Convert everything to NumPy (fast, safe, shape‑checked) ----
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sample_size = int(sample_size)
        print ('test')

        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)

        # ---- Simple sanity checks -------------------------------------------------
        if self.lb.shape != self.ub.shape:
            raise ValueError("`lb` and `ub` must have the same shape")
        if self.lb.ndim != 1:
            raise ValueError("`lb`/`ub` must be 1‑dimensional vectors")
        # If X already contains data, its dimensionality must match the bounds.
        if self.X.size > 0 and self.X.shape[1] != self.lb.size:
            raise ValueError(
                "Feature dimension of X does not match length of lb/ub"
            )

        # ---- Fit a *deterministic* RandomForestRegressor -------------------------
        #   • n_estimators is large enough for a smooth variance estimate.
        #   • random_state=0 guarantees reproducibility.
        #   • n_jobs=1 removes nested parallelism → fixes BrokenProcessPool error.
        self.model = RandomForestRegressor(
            n_estimators=300,
            random_state=0,
            n_jobs=1,            # <‑‑ critical fix for multiprocessing environments
            bootstrap=True,
        )
        self.model.fit(self.X, self.y)

        # ---- Pre‑compute scaling for the (lb, ub) → [0,1] affine map -------------
        self._dim   = self.lb.size          # number of features (d)
        self._scale = self.ub - self.lb     # width per dimension

        # ---- Private hyper‑parameters (kept hidden – API unchanged) -------------
        self._alpha       = 0.7            # weight for uncertainty (vs. diversity)
        self._pool_factor = 25             # candidate pool = factor * sample_size

    # ------------------------------------------------------------------
    # 2️⃣  Candidate generation – deterministic (Sobol or simple fallback)
    # ------------------------------------------------------------------
    def _candidate_pool(self) -> np.ndarray:
        """
        Build a deterministic candidate pool.

        Returns
        -------
        candidates : np.ndarray, shape (n_candidates, d)
            Points spread uniformly over the bounded search space.
        """
        # At least a few hundred points → stable statistics.
        n_candidates = max(self.sample_size * self._pool_factor, 200)

        if _HAS_SOBOL:
            # Sobol generates 2**m points; we take the smallest power of two ≥ n_candidates.
            m = int(np.ceil(np.log2(n_candidates)))
            sobol = qmc.Sobol(d=self._dim, scramble=False)  # deterministic, no scrambling
            unit = sobol.random_base2(m=m)[:n_candidates]     # shape (n_candidates, d)
            candidates = self.lb + unit * self._scale          # affine map to user bounds
        else:
            # Deterministic fallback: repeat a 1‑D linspace across every dimension.
            # Not as space‑filling as Sobol but still repeatable and requires no RNG.
            lin = np.linspace(0.0, 1.0, n_candidates)
            unit = np.tile(lin[:, None], (1, self._dim))      # (n_candidates, d)
            candidates = self.lb + unit * self._scale

        return candidates

    # ------------------------------------------------------------------
    # 3️⃣  Uncertainty score – variance across forest trees
    # ------------------------------------------------------------------
    def _uncertainty_score(self, candidates: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty as the variance of predictions across all trees.

        Parameters
        ----------
        candidates : np.ndarray, shape (n_candidates, d)

        Returns
        -------
        unc : np.ndarray, shape (n_candidates,)
            Larger values → higher model uncertainty.
        """
        # Pull predictions from each individual estimator.
        # Shape: (n_estimators, n_candidates) for single‑output,
        #        (n_estimators, n_candidates, n_targets) for multi‑output.
        tree_preds = np.stack(
            [tree.predict(candidates) for tree in self.model.estimators_],
            axis=0,
        )

        var = np.var(tree_preds, axis=0)      # variance across trees

        # Collapse multi‑target variance to a scalar per candidate.
        if var.ndim > 1:                      # (n_candidates, n_targets)
            var = var.mean(axis=1)            # (n_candidates,)

        return var

    # ------------------------------------------------------------------
    # 4️⃣  Diversity score – distance to nearest labelled point
    # ------------------------------------------------------------------
    def _diversity_score(self, candidates: np.ndarray) -> np.ndarray:
        """
        Compute a diversity score: Euclidean distance to the nearest already‑labelled
        sample.  If no labelled data exist yet, all candidates are considered equally
        diverse (return ``inf`` so diversity does not influence the final ranking).

        Parameters
        ----------
        candidates : np.ndarray, shape (n_candidates, d)

        Returns
        -------
        dist : np.ndarray, shape (n_candidates,)
        """
        if self.X.shape[0] == 0:                 # first AL iteration – no data
            return np.full(candidates.shape[0], np.inf)

        # KD‑Tree gives fast exact nearest‑neighbour queries.
        tree = KDTree(self.X, leaf_size=40)
        dists, _ = tree.query(candidates, k=1)   # (n_candidates, 1)
        return dists.ravel()                     # flatten to 1‑D

    # ------------------------------------------------------------------
    # 5️⃣  Public method – pick the batch that maximises the hybrid score
    # ------------------------------------------------------------------
    def gen_samples(self):
        """
        Generate a batch of ``sample_size`` points that jointly maximise model
        uncertainty **and** diversity.

        Returns
        -------
        sample_set : np.ndarray, shape (sample_size, d)
            Selected points (ready for evaluation and addition to the training set).
        model : RandomForestRegressor
            The already‑trained model – retained for backward compatibility.
        """
        # ----- Build a rich candidate pool ------------------------------------
        candidates = self._candidate_pool()                # (n_cand, d)

        # ----- Compute the two acquisition components ------------------------
        unc = self._uncertainty_score(candidates)         # (n_cand,)
        div = self._diversity_score(candidates)          # (n_cand,)

        # ----- Normalise each component to [0, 1] ----------------------------
        eps = 1e-12                                        # avoid div‑zero
        unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + eps)
        div_norm = (div - div.min()) / (div.max() - div.min() + eps)

        # ----- Weighted sum (α ≈ 0.7 → more emphasis on uncertainty) ----------
        score = self._alpha * unc_norm + (1.0 - self._alpha) * div_norm

        # ----- Select the top ``sample_size`` candidates --------------------
        # ``argpartition`` runs in O(n) → cheap even for thousands of points.
        top_idx = np.argpartition(-score, self.sample_size - 1)[:self.sample_size]
        sample_set = candidates[top_idx]

        # Ensure a true 2‑D array even when ``sample_size`` == 1.
        sample_set = np.atleast_2d(sample_set)
        
        # ----- Return the batch and the current model (unchanged API) -------
        return sample_set, self.model
