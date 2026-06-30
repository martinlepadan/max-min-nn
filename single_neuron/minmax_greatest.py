from .neuron import Neuron
from fuzzy import impl
import numpy as np


class MinMaxGreatestNeuron(Neuron):
    """
    Principle (per epoch):
    - the switch `b` is the bifurcation variable: we read off the pole preferred by
      the data: we take b=1 (level = y) if s is closer to y than 1-s is, else b=0
      (level = 1-y) (and we update b like in the first neuron)
    - given b, the target of the aggregation is level = y <-> b (from the Notes RN MinMax)
    - the weights are the greatest solution of the system s(a) = level

    """

    def fit(self, X, Y, epochs=8, lam_b=0.3, q=0.10, method="min"):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        a = np.asarray(self.a, dtype=float).copy()
        b = 0.5
        for _ in range(epochs):
            s = np.minimum(a[None, :], X).max(axis=1)
            prefer_direct = np.mean(np.abs(Y - s) <= np.abs(Y - (1.0 - s))) >= 0.5
            b = (1.0 - lam_b) * b + lam_b * (1.0 if prefer_direct else 0.0)

            level = Y if b >= 0.5 else 1.0 - Y
            a = self._solve_greatest(X, level, q, method)

        self.a, self.b = a, float(b)
        return self

    @staticmethod
    def _solve_greatest(X, level, q, method="min"):
        """
        Greatest solution of the system s(a) = level.
        """
        G = impl(X, level[:, None])
        if method == "quantile":
            return np.quantile(G, q, axis=0)
        return G.min(axis=0)
