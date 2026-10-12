import numpy as np
from scipy.stats import kendalltau

class VB_FTRL:
    def __init__(self, d, mu=6, lambda_val=7, S=10, threshold=0.0, corr_func=None, scoring_func=None):
        """
        d             — dimension of the portfolio
        mu            — temperature parameter
        lambda_val    — uniform‐portfolio bias
        S             — number of quasi‐Newton steps
        threshold     — correlation threshold for triggering a restart
        corr_func     — function to compute correlation between two ranking arrays; 
                         should return (corr, p‐value) or a single correlation value
        scoring_func  — function that takes a 2D array (assets × history) and returns scores for each asset
        """
        self.d = d
        self.mu = mu
        self.lambda_val = lambda_val
        self.S = S
        self.threshold = threshold

        # correlation function (default: kendalltau)
        self.corr_func = corr_func or kendalltau

        # scoring function (must be provided)
        if scoring_func is None:
            raise ValueError("A scoring_func must be provided")
        self.scoring_func = scoring_func

        # current portfolio weights on the simplex
        self.w = np.ones(d) / d

        # history of return‐vectors; each element is a 1D array of length d
        self.returns = []

        # indices in `returns` where we last restarted
        self.restart_index = [0]

        # build the static affine‐mapping matrices
        self._initialize_mappings()

    def _initialize_mappings(self):
        # A : maps R^(d−1) -> R^d embedding into the simplex
        self.A = np.vstack([np.eye(self.d - 1), -np.ones(self.d - 1)])
        # e_d so that A·v + e_d sums to 1
        self.e_d = np.zeros(self.d)
        self.e_d[-1] = 1
        # pseudoinverse of A
        self.A_pinv = np.linalg.pinv(self.A)

    def affine_mapping(self, v):
        """Map unconstrained v in R^(d−1) back to the simplex in R^d."""
        return self.A.dot(v) + self.e_d

    def inverse_affine_mapping(self, w):
        """Map a simplex vector w in R^d back to the unconstrained v in R^(d−1)."""
        return self.A_pinv.dot(w - self.e_d)

    def _compute_rank_positions(self, scores):
        """
        Given a score vector of length d, transform into ranks:
        0 = best score, d−1 = worst score.
        """
        desc = np.argsort(scores)[::-1]
        ranks = np.empty_like(desc)
        ranks[desc] = np.arange(self.d)
        return ranks

    def n_restart(self):
        """Return the number of restarts that have occurred."""
        return len(self.restart_index) - 1

    def restart(self, prev_return, prev_index):
        """
        Perform a manual “restart” at the given previous return index.
        This simulates VB-FTRL‐qN starting from uniform on return_{prev_index},
        computes one quasi‐Newton update, and resets `self.w` accordingly.
        Also records the restart_index.

        Parameters:
        -----------
        prev_return : np.ndarray
            The return vector at time index `prev_index`.
        prev_index : int
            The index in self.returns corresponding to prev_return.
        """
        # Build X_sim: stack identity (d×d) above the previous return vector (1×d)
        X_sim = np.vstack((np.eye(self.d), np.array([prev_return])))
        # λ vector for simulation: length = d + 1
        lambda_sim = np.ones(self.d + 1)
        lambda_sim[:self.d] *= self.lambda_val
        ones_d = np.ones(self.d)

        # Start from uniform-weighted portfolio on the simplex
        v_sim = self.inverse_affine_mapping(np.ones(self.d) / self.d)

        # Perform S quasi‐Newton steps to obtain w_t from return_{prev_index}
        for _ in range(self.S):
            w_aff = self.affine_mapping(v_sim)          # current weight on simplex
            prod = X_sim.dot(w_aff)                     # (d+1)-vector of X_sim · w_aff
            # Compute gradients: shape ((d+1) × (d−1))
            grads = -(1 / prod)[:, None] * (self.A.T.dot(X_sim.T)).T
            # Hessian H: shape ((d−1) × (d−1))
            H = np.einsum('i,ij,ik->jk', lambda_sim, grads, grads)
            H_inv = np.linalg.inv(H)
            # π vector: length = d+1
            pi = lambda_sim * np.einsum('ij,jk,ik->i', grads, H_inv, grads)
            # Compute gradient of the objective w.r.t. v
            G = 2 * self.A.T.dot(ones_d) + grads.T.dot(lambda_sim + self.mu * pi)
            # Compute modified Hessian M
            M = H + 3 * self.mu * np.einsum('i,ij,ik->jk', pi, grads, grads)
            # Quasi‐Newton update on v
            v_sim = v_sim - np.linalg.solve(M, G)

        # Record the restart index and set the new portfolio
        self.restart_index.append(prev_index)
        self.w = self.affine_mapping(v_sim)

    def update(self, returnt):
        """
        Given the return-vector at time t, compute the new portfolio for t+1.
        Implements a look-ahead restart:
          1. Append returnt to self.returns.
          2. If there are at least 2 returns, compute scores at t−2 and t−1,
             rank them, and compute correlation. If corr < threshold:
               a. Call self.restart(prev_return, prev_index) with t−1’s return.
          3. Run the standard VB-FTRL‐qN update from the last true restart to get w_{t+1}.

        Returns:
        --------
        self.w : np.ndarray
            Updated portfolio weights on the simplex for the next period.
        """
        # 1) Append the new return vector at time t
        self.returns.append(returnt)
        n = len(self.returns)  # current number of stored returns

        # 2) Check restart condition (only if threshold != -1 and we have ≥ 2 returns)
        if self.threshold != -1 and n > 2:
            # Compute scores using history up to t−2 and up to t−1
            hist_prev = np.array(self.returns[:-2]).T  # shape: (d × (n-2))
            hist_curr = np.array(self.returns[:-1]).T  # shape: (d × (n-1))
            score_prev = self.scoring_func(hist_prev)  # length d
            score_curr = self.scoring_func(hist_curr)  # length d

            rank_prev = self._compute_rank_positions(score_prev)
            rank_curr = self._compute_rank_positions(score_curr)

            corr_res = self.corr_func(rank_prev, rank_curr)
            try:
                tau = corr_res[0]
            except Exception:
                tau = float(corr_res)

            if tau < self.threshold:
                # Perform a manual restart at t−1
                prev = self.returns[-2]
                prev_index = n - 2
                self.restart(prev, prev_index)

        # 3) Standard VB-FTRL update from the last restart index
        slice_returns = np.array(self.returns[self.restart_index[-1]:])
        t_prime = slice_returns.shape[0]  # number of “post-restart” returns

        # Build X: stack identity (d×d) above slice_returns (t_prime×d)
        X = np.vstack((np.eye(self.d), slice_returns))
        lambda_tau = np.ones(self.d + t_prime)
        lambda_tau[:self.d] *= self.lambda_val
        ones_d = np.ones(self.d)

        # Map current simplex weight back to unconstrained v
        v = self.inverse_affine_mapping(self.w)

        # Perform S quasi‐Newton steps to compute w_{t+1}
        for _ in range(self.S):
            w_aff = self.affine_mapping(v)
            prod = X.dot(w_aff)
            grads = -(1 / prod)[:, None] * (self.A.T.dot(X.T)).T
            H = np.einsum('i,ij,ik->jk', lambda_tau, grads, grads)
            H_inv = np.linalg.inv(H)
            pi = lambda_tau * np.einsum('ij,jk,ik->i', grads, H_inv, grads)
            G = 2 * self.A.T.dot(ones_d) + grads.T.dot(lambda_tau + self.mu * pi)
            M = H + 3 * self.mu * np.einsum('i,ij,ik->jk', pi, grads, grads)
            v = v - np.linalg.solve(M, G)

        # Update the portfolio to w_{t+1} on the simplex
        self.w = self.affine_mapping(v)
        return self.w
