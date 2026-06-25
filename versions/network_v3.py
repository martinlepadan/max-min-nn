from versions.network_v2 import V2Net

class V3Net(V2Net):
    """
    V3: Same as V2 but always uses the shortcut method to do the inversion
    """

    version = "V3"

    def _layer_target(self, k, t):
        """Return (t_prev, mode) for layer k always using the approximation method"""
        A, b = self.A[k], self.b[k]
        return self.approx_target(A, b, t), "approx"