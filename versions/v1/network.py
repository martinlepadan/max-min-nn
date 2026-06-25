import numpy as np

from fuzzy import mmp, mip
from common.network import BaseMaxMinNet


class V1Net(BaseMaxMinNet):
    """
    V1: A net with a backpropagation method based on target propagation (not SGD)
      forward : z = mmp(A, x)
      backward: 3 steps, target propagated with t_prev = mip(A.T, t)
    """

    version = "V1"

    def neuron(self, k, z):
        return mmp(self.A[k], z)

    def backward_layer(self, k, z_prev, t, lam, beta):
        z_prev = np.asarray(z_prev, dtype=float)
        t = np.asarray(t, dtype=float)
        A = self.A[k]

        # Step 1: weight update via the greatest solution or an approimation
        if self.adaptative:
            a = self.adaptative_approx(z_prev, t)
        else:
            a = self.greatest_solution(z_prev, t)

        A = (1.0 - lam) * A + lam * a

        # Step 2: small gradient step
        if self.use_gradient_step:
            mins = np.minimum(A, z_prev[None, :])
            z_out = mins.max(axis=1)
            j_star = mins.argmax(axis=1)
            rows = np.arange(A.shape[0])
            A[rows, j_star] += beta * (t - z_out)
            np.clip(A, 0.0, 1.0, out=A)

        self.A[k] = A

        # Step 3: target for the previous layer
        return mip(A.T, t)
