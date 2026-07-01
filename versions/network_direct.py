import numpy as np

from fuzzy import equ, impl, eps, mmp, mip
from common.metrics import r2_score
from common.network import BaseMaxMinNet


class DirectSolveNet(BaseMaxMinNet):
    """
    Gradient-free max-min network trained by target propagation (one forward/backward
    pass per epoch). Layer forward: z = equ(mmp(A, z_prev), b).

    Backward pass, per layer:
      1. switch b: move it toward the pole the majority of samples prefer (s vs 1-s).
      2. weights A: solve s(a) = level over the batch (level = y if b=1, 1-y if b=0),
         blend it into A (lam_a), then apply a V1-style gradient step (grad, beta).
      3. target for the previous layer: invert the layer, x = min_i impl(A_i, equ(y,b)_i),
         then smooth it toward the current activation (eta).

    Options (attributes):
      method : "min" (exact greatest solution) or "quantile" (robust, uses q).
      grad   : gradient step after the solve. Needed to make "min" usable and to avoid the
               constant-output collapse on binary data (see diagnose_direct.py).
      lam_a  : how much of the solve to keep (1 = overwrite; <1 = moving average so the
               gradient can accumulate). Works better for binary data than continous one apparently
      eta    : target smoothing, t <- (1-eta)*z + eta*g(t). eta=1 = raw inversion.
      dtp    : difference target propagation (fallback; exact with true weights, ~neutral here).
      dual   : if level = 1-y, solve with the smallest solution (eps) instead of the greatest.
    """

    version = "DIRECT"

    def _init_extra(self, rng):
        self.b = [rng.random(self.sizes[k + 1]) for k in range(len(self.sizes) - 1)]
        self.method = "quantile"   # "min" = exact, "quantile" = robust
        self.q = 0.85
        self.dual = False # smallest solution (eps) for the inverted pole
        self.dtp = False # difference target propagation (fallback)
        self.grad = True # V1-style gradient step after the solve
        self.lam_a = 0.1 # solve blend: 1 = overwrite, <1 = gradient can accumulate
        self.eta = 1.0 # target relaxation: 1 = raw inversion, <1 = damped

    def neuron(self, k, z):
        return equ(mmp(self.A[k], z), self.b[k])

    def _forward_batch(self, X):
        Z = np.asarray(X, dtype=float)
        acts = [Z]
        for k in range(self.n_layers):
            Z = equ(mmp(self.A[k][None, :, :], Z[:, None, :]), self.b[k][None, :]) # (N, m)
            acts.append(Z)
        return acts

    def _solve_weights(self, Z_prev, level, direct):
        """Solve s(a) = level for a whole layer (greatest solution, or smallest if dual)."""
        G = impl(Z_prev[:, None, :], level[:, :, None]) # (N, m, n_in)
        great = G.min(axis=0) if self.method == "min" else np.quantile(G, self.q, axis=0)
        if not self.dual:
            return great

        E = eps(Z_prev[:, None, :], level[:, :, None]) # (N, m, n_in)
        small = E.max(axis=0) if self.method == "min" else np.quantile(E, 1.0 - self.q, axis=0)

        return np.where(direct[:, None], great, small)

    def _backward_layer_batch(self, k, Z_prev, T, lam_b, beta):
        """Backward pass for one layer; returns the target for the previous layer."""
        Z_prev = np.asarray(Z_prev, dtype=float)
        T = np.asarray(T, dtype=float)

        # 1. switch: move b toward the pole the majority of samples prefer
        s = mmp(self.A[k][None, :, :], Z_prev[:, None, :]) # (N, m)
        prefer_direct = np.mean(np.abs(T - s) <= np.abs(T - (1.0 - s)), axis=0) >= 0.5
        self.b[k] = (1.0 - lam_b) * self.b[k] + lam_b * prefer_direct.astype(float)
        direct = self.b[k] >= 0.5 # (m,)

        # 2. weights: solve, blend into A, then a gradient step
        level = np.where(direct[None, :], T, 1.0 - T) # (N, m)
        A_solved = self._solve_weights(Z_prev, level, direct)
        self.A[k] = (1.0 - self.lam_a) * self.A[k] + self.lam_a * A_solved
        if self.grad and beta:
            self._gradient_batch(k, Z_prev, T, beta)

        # 3. target for the previous layer
        tau = equ(T, self.b[k][None, :]) # (N, m)
        g_target = mip(self.A[k].T[None, :, :], tau[:, None, :])

        if self.dtp:  # difference target propagation
            s_cur = mmp(self.A[k][None, :, :], Z_prev[:, None, :])
            g_current = mip(self.A[k].T[None, :, :], s_cur[:, None, :])
            return np.clip(Z_prev + (g_target - g_current), 0.0, 1.0)

        # smoothly goes to the target (same idea as the update of b)
        return (1.0 - self.eta) * Z_prev + self.eta * g_target

    def _gradient_batch(self, k, Z_prev, T, beta):
        """
        V1-style gradient step: for each neuron and sample, nudge the *winning* input
        coordinate (the one reaching the max-min) toward reducing the error. sigma fixes
        the direction through the switch.
        """
        A = self.A[k]
        s = mmp(A[None, :, :], Z_prev[:, None, :]) # (N, m)
        err = T - equ(s, self.b[k][None, :]) # (N, m)
        sigma = np.where(s + self.b[k][None, :] >= 1.0, 1.0, -1.0)
        j_star = np.minimum(A[None, :, :], Z_prev[:, None, :]).argmax(axis=2) # (N, m)

        N, m = err.shape
        upd = (beta * sigma * err).ravel()
        rows = np.broadcast_to(np.arange(m)[None, :], (N, m)).ravel()
        grad = np.zeros_like(A)
        np.add.at(grad, (rows, j_star.ravel()), upd)
        self.A[k] = np.clip(A + grad / N, 0.0, 1.0)

    def fit(self, X, Y, X_eval, Y_eval, lam=None, lam_b=0.3, beta=2.0,
            epochs=50, patience=10, delta=1e-3, verbose=False, X_test=None, Y_test=None):
        """
        Training loop
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        history = []
        self.r2_train_hist, self.r2_test_hist = [], []
        best_r2 = -np.inf
        best_snapshot = self._snapshot()
        best_patience_r2 = -np.inf
        wait = 0

        for epoch in range(epochs):
            acts = self._forward_batch(X)
            T = Y.reshape(-1, 1)
            for k in reversed(range(self.n_layers)):
                T = self._backward_layer_batch(k, acts[k], T, lam_b, beta)

            r2 = r2_score(Y_eval, self.predict(X_eval))
            history.append(r2)
            self.r2_train_hist.append(r2_score(Y, self.predict(X)))
            self.r2_test_hist.append(
                r2_score(Y_test, self.predict(X_test)) if X_test is not None else r2)
            if verbose:
                print(f"epoch {epoch:3d}  R2={r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_snapshot = self._snapshot()

            if r2 > best_patience_r2 + delta:
                best_patience_r2 = r2
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch}")
                    break

        self._restore(best_snapshot)
        return history
