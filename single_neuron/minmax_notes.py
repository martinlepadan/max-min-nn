from .neuron import Neuron
from fuzzy import impl
import numpy as np


class MinMaxNotesNeuron(Neuron):
    """
    Neuron according to the notes, we solve the system s(a, x) = {y, 1-y} that depends of the value of 
    X_max and beta/alpha
    """

    def fit(self, X, Y, seed=0):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        rng = np.random.default_rng(seed)

        levels = np.empty(len(Y))
        bs = np.empty(len(Y))
        for k, (x, y) in enumerate(zip(X, Y)):
            levels[k], bs[k] = self._target(x, y, rng)

        self.a = impl(X, levels[:, None]).min(axis=0)
        self.b = float(bs.mean())
        return self

    @staticmethod
    def _target(x, y, rng):
        """Get the b (unform distrib) and level (the solution we are looking for) from X_max comparison with beta/alpha"""
        xmax = x.max()
        alpha = min(y, 1.0 - y)
        beta = max(y, 1.0 - y)
        if xmax >= beta:
            level = beta
            b = rng.uniform(y, 1.0) if y >= 0.5 else rng.uniform(0.0, y)
        elif xmax <= alpha:
            level = alpha
            b = 1.0 - y
        else:  # alpha < xmax < beta
            level = alpha
            b = rng.uniform(0.0, alpha) if y >= 0.5 else rng.uniform(beta, 1.0)
        return level, b
