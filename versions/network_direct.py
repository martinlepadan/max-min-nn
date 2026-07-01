import numpy as np

from fuzzy import equ, impl, eps, mmp, mip
from common.metrics import r2_score
from common.network import BaseMaxMinNet


class DirectSolveNet(BaseMaxMinNet):
    """
    Network with minmax_greatest neurons

    Forward of a layer : z = equ(mmp(A, z_prev), b)

    Backpropagation:
    1. switch b : majority pole (does the majority of samples prefer s or 1-s?), then we update b by going smoothly toward that pole;
    2. weights A: solve the system s(a) = level over the whole batch, with level = y if b=1 (direct) and level = 1-y if b=0 (inverted);
    3. target for the repvious layer (same as in network_v3.py): we solve the max-min system mmp(A, x) = equ(y, b) by its greatest solution x = min_i impl(A_i, equ(y, b)_i).
    (for this last step, we may use a quantile instead of the exact min to be more robust if the solution is not feeasible for all targets)

    dual=True: if level=1-y, we solve the system by its smallest solution instead of the gretest one
    """

    version = "DIRECT"

    def _init_extra(self, rng):
        self.b = [rng.random(self.sizes[k + 1]) for k in range(len(self.sizes) - 1)]
        self.method = "quantile"
        self.q = 0.85
        self.dual = False

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
        """
        Solve s(a) = level for a whole layer
        """
        G = impl(Z_prev[:, None, :], level[:, :, None]) # (N, m, n_in)
        great = G.min(axis=0) if self.method == "min" else np.quantile(G, self.q, axis=0)
        if not self.dual:
            return great

        E = eps(Z_prev[:, None, :], level[:, :, None]) # (N, m, n_in)
        small = E.max(axis=0) if self.method == "min" else np.quantile(E, 1.0 - self.q, axis=0)

        return np.where(direct[:, None], great, small)

    def _backward_layer_batch(self, k, Z_prev, T, lam_b):
        """
        Backpropagation of a single layer
        """
        Z_prev = np.asarray(Z_prev, dtype=float)
        T = np.asarray(T, dtype=float)

        # First step: switch update
        s = mmp(self.A[k][None, :, :], Z_prev[:, None, :]) # (N, m)
        prefer_direct = np.mean(np.abs(T - s) <= np.abs(T - (1.0 - s)), axis=0) >= 0.5
        self.b[k] = (1.0 - lam_b) * self.b[k] + lam_b * prefer_direct.astype(float)
        direct = self.b[k] >= 0.5 # (m,)

        # Second step: weights update
        level = np.where(direct[None, :], T, 1.0 - T) # (N, m)
        self.A[k] = self._solve_weights(Z_prev, level, direct)

        # Last step: target for the previous layer
        tau = equ(T, self.b[k][None, :]) # (N, m)
        x_hat = mip(self.A[k].T[None, :, :], tau[:, None, :])  # (N, n_in)
        return x_hat

    def fit(self, X, Y, X_eval, Y_eval, lam=None, lam_b=0.3, beta=None,
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
                T = self._backward_layer_batch(k, acts[k], T, lam_b)

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
