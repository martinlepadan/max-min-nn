import numpy as np

from fuzzy import equ, mmp, mip, impl
from common.network import BaseMaxMinNet


class V2Net(BaseMaxMinNet):
    """
    V2 (IPMU): add a "switch" that allows the neuron to either take the value (b=1) or the opposite (b=0)
    to modelize non-monotone function
    """

    version = "V2"
    def _init_extra(self, rng):
        self.b = [rng.random(self.sizes[k + 1]) for k in range(len(self.sizes) - 1)]
        self.feasibility_history = []
        self._epoch_counts = {}
        self._total_counts = {}

    def neuron(self, k, z):
        return equ(mmp(self.A[k], z), self.b[k])

    @staticmethod
    def _greatest_input_batch(A, b, t, eps=1e-9):
        """
        Calculate the greatest input for all the neurons of a layer
        Parameters
        ----------
        A: array (m, n)
            Weights matrix of the layer
        b: array (m,)
            Switches vector of the layer
        t: array (m,)
            Target for the previous layer
        eps: float
            Threshold for the family boundaries

        Returns
        -------
        xmax (m, n) greatest input per neuron (rows with no family are -inf),
        empty (m,) True where no family applies.
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        t = np.asarray(t, dtype=float)
        alpha = np.minimum(t, 1.0 - t)
        beta = np.maximum(t, 1.0 - t)
        amax = A.max(axis=1)

        # applicability mask per family, 
        m1 = (np.abs(b - t) <= eps) & (amax >= beta)
        m2 = (b >= beta) & (amax >= t)
        m3 = np.abs(b - (1.0 - t)) <= eps
        m4 = (b <= alpha) & (amax >= (1.0 - t))

        # Get the solutions of the equations for each families
        neg = np.full_like(A, -np.inf)
        c1 = np.where(m1[:, None], 1.0, neg)
        c2 = np.where(m2[:, None], impl(A, t[:, None]), neg)
        c3 = np.where(m3[:, None], impl(A, alpha[:, None]), neg)
        c4 = np.where(m4[:, None], impl(A, (1.0 - t)[:, None]), neg)

        xmax = np.maximum.reduce([c1, c2, c3, c4])   # greatest element per neuron
        empty = ~(m1 | m2 | m3 | m4)
        return xmax, empty

    @staticmethod
    def approx_target(A, b, t):
        """Fallback equations if the previous ones had no feasible solutions"""
        tau = equ(t, b)
        return mip(np.asarray(A, dtype=float).T, tau)

    def _step1_weights(self, k, x, t, lam_a, lam_b):
        """
        Step 1: updating (a) and (b)

        Solve the following neuron's equation: equ(mmp(a), b) = y with (x) and (y) fixed.

        The solution set (a, b) is the union of T(1)..T(4). But we select
        directly the solution that makes b boolean:
            b = 1  ->  yhat = s(a)
            b = 0  ->  yhat = 1 - s(a)

        Different possibilities:
        - y <= Xmax -> we fall into the T(2) family: we set b=1
        - (1-y) <= Xmax -> we fall into the T(4) family we set b=0
        - both possible: we set b to the closest (b>=0.5 -> 1 else 0)
        - none possible -> (Xmax < min(y,1-y)): we set a=0, b=1-y

        The new weight is the greatest solution of de s(a)=level

        Then we update using the iterative update rule like in V1
        """
        A, b = self.A[k], self.b[k]
        xmax = x.max()
        is_t2 = t <= xmax
        is_t4 = (1.0 - t) <= xmax
        both = is_t2 & is_t4
        is_t3 = (~is_t2) & (~is_t4)

        go_direct = np.where(both, b >= 0.5, is_t2)

        level = np.where(go_direct, t, 1.0 - t)
        b_target = np.where(go_direct, 1.0, 0.0)
        a_new = impl(x[None, :], level[:, None])

        a_new[is_t3] = 0.0
        b_target = np.where(is_t3, 1.0 - t, b_target)

        self.A[k] = (1.0 - lam_a) * A + lam_a * a_new
        self.b[k] = (1.0 - lam_b) * b + lam_b * b_target

    # Calculate the gradient (not indicated in the paper, so we make this step optional)
    def _step2_gradient(self, k, x, t, beta):
        A, b = self.A[k], self.b[k]
        s = mmp(A, x)
        yhat = equ(s, b)
        err = t - yhat

        # without sigma, b would always increase
        sigma = np.where(s + b >= 1.0, 1.0, -1.0)
        mins = np.minimum(A, x[None, :])
        j_star = mins.argmax(axis=1)
        rows = np.arange(A.shape[0])
        A[rows, j_star] = np.clip(A[rows, j_star] + beta * sigma * err, 0.0, 1.0)

    def _layer_target(self, k, t):
        """Return (t_prev, mode) for layer k"""
        A, b = self.A[k], self.b[k]
        xmax, empty = self._greatest_input_batch(A, b, t)

        if empty.any():
            return self.approx_target(A, b, t), "approx" # an empty neuron -> approx
        
        x_max = xmax.min(axis=0) # greatest common candidate
        yhat = equ(mmp(A, x_max), b) # re-evaluate to check that the system is still consistent

        if np.allclose(yhat, t, atol=1e-8):
            return x_max, "exact"
        
        return self.approx_target(A, b, t), "approx" 

    def backward_layer(self, k, z_prev, t, lam_a, lam_b, beta):
        x = np.asarray(z_prev, dtype=float)
        t = np.asarray(t, dtype=float)

        self._step1_weights(k, x, t, lam_a, lam_b)
        if self.use_gradient_step:
            self._step2_gradient(k, x, t, beta)
        t_prev, mode = self._layer_target(k, t)
        self._record(k, mode)
        return t_prev

    def _record(self, layer, mode):
        self._epoch_counts.setdefault(layer, {"exact": 0, "approx": 0})
        self._epoch_counts[layer][mode] += 1

    def _on_epoch_start(self):
        self._epoch_counts = {}

    def _on_epoch_end(self, epoch):
        rates = {}
        for layer, c in self._epoch_counts.items():
            tot = c["exact"] + c["approx"]
            rates[layer] = c["exact"] / tot if tot else 0.0

            acc = self._total_counts.setdefault(layer, {"exact": 0, "approx": 0})
            acc["exact"] += c["exact"]
            acc["approx"] += c["approx"]
        self.feasibility_history.append(rates)

    def feasibility_rate(self):
        """
        Feasibility rate of the exact inversion over the whole training for each layer and in global
        """
        per_layer = {}
        total_exact = total = 0
        for layer in sorted(self._total_counts):
            c = self._total_counts[layer]
            n = c["exact"] + c["approx"]
            per_layer[layer] = c["exact"] / n if n else 0.0
            total_exact += c["exact"]
            total += n
        return {
            "per_layer": per_layer,
            "global": total_exact / total if total else 0.0,
        }
