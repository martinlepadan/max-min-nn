import numpy as np
from fuzzy import equ, mmp, impl

class Neuron:

    def __init__(self, n, b_init="1", seed=0):

        rng = np.random.default_rng(seed)
        self.a = rng.random(n)

        if type(b_init) is float:
            self.b = b_init
        if b_init == "random":
            self.b = 1.0 if rng.random() >= 0.5 else 0.0

    def forward(self, z):
        return equ(mmp(self.a, z), self.b)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return [self.forward(x) for x in X]

    def step(self, x, y, lam_a, beta, lam_b, method_both="closest"):

        # Same as in the network to get the greatest solution
        x = np.asarray(x, dtype=float)
        xmax = x.max()
        is_t2 = y <= xmax
        is_t4 = (1.0 - y) <= xmax
        both = is_t2 and is_t4
        is_t3 = (not is_t2) and (not is_t4)

        if is_t3:
            level, b_target, a_new = 1.0 - y, 1.0 - y, np.zeros_like(x)

        else:
            if both:
                if method_both == "error":
                    # choose {0,1} whose output (s or 1-s) is closer to y
                    s = mmp(self.a, x)
                    go_direct = abs(y - s) <= abs(y - (1.0 - s))
                else: # go with the closest pole
                    go_direct = self.b >= 0.5
            else:
                go_direct = is_t2
            level = y if go_direct else 1.0 - y
            b_target = 1.0 if go_direct else 0.0
            a_new = impl(x, level)

        # Step 1: update of (a,b)
        self.a = (1.0 - lam_a) * self.a + lam_a * a_new
        self.b = (1.0 - lam_b) * self.b + lam_b * b_target

        # Step 2: gradient
        if beta:
            s = mmp(self.a, x)
            yhat = equ(s, self.b)
            err = y - yhat
            sigma = 1.0 if (s + self.b) >= 1.0 else -1.0
            mins = np.minimum(self.a, x)
            winners = mins == mins.max()

            self.a[winners] = np.clip(self.a[winners] + beta * sigma * err, 0.0, 1.0)

        # Step 3: only for multiple layers

    def fit(self, X, Y, lam_a, lam_b, beta, method_both, epochs=200):
        for _ in range(epochs):
            for x, y in zip(X, Y):
                self.step(x, y, lam_a, beta, lam_b, method_both=method_both)
