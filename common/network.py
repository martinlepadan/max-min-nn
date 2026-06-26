import copy
import numpy as np

from fuzzy import impl
from common.metrics import r2_score


def default_layers(n: int, generate: bool = False) -> list:
    """
    Default architectures   
    """
    if generate:
        return [n, 2 * n, n, 1]
    return [n, 2 * n, 4 * n, 2 * n, n, 1]


class BaseMaxMinNet:
    """
    Base class of the Max-Min network, all versions inherits from this class by overriding the neuron and backward_layer methods.
    """

    version = "BASE"

    def __init__(self, sizes: list, seed: int = 0, use_gradient_step: bool = True, adaptative: bool = False):
        self.sizes = list(sizes)
        self.use_gradient_step = use_gradient_step
        self.adaptative = adaptative
        rng = np.random.default_rng(seed)
        self.A = [rng.random((sizes[k + 1], sizes[k])) for k in range(len(sizes) - 1)]
        self.b = None
        self._init_extra(rng)

    def _init_extra(self, rng):
        """Init extra parameters for the next version"""
        pass

    def _on_epoch_start(self):
        pass

    def _on_epoch_end(self, epoch):
        pass

    def neuron(self, k: int, z: np.ndarray) -> np.ndarray:
        """
        Logic of the neuron for layer k applied to input z_{k-1}.

        Parameters
        ----------
        k : int
            Layer index
        z : np.ndarray
            Input to the layer (activations from the previous layer, z_{k-1}).

        Returns
        -------
        np.ndarray
            Output of the layer (activations for the next layer, z_k).
        """
        raise NotImplementedError

    def backward_layer(self, k, z_prev, t, lam_a, beta, lam_b=None):
        """
        Backward pass for layer k.

        Parameters
        ----------
        k : int
            Layer index
        z_prev : np.ndarray
            Input to the layer (activations from the previous layer, z_{k-1}).
        t : np.ndarray
            Target for the layer.
        lam_a : float
            Regularization parameter for the weights
        beta : float
            Step size.
        lam_b : float
            Regularization parameter for the switchs

        Returns
        -------
        np.ndarray
            Updated target for the previous layer.
        """
        raise NotImplementedError


    @staticmethod
    def greatest_solution(z_prev, t):
        """
        Compute the greatest solution a of the equation mmp(a, z_prev) = t
        Parameters
        ----------
        z_prev : np.ndarray
            Input to the layer (activations from the previous layer, z_{k-1}).
        t : np.ndarray
            Target for the layer.
        Returns
        -------
        np.ndarray
            Solution of the equation
        """
        z_prev = np.asarray(z_prev, dtype=float)
        t = np.asarray(t, dtype=float)
        return impl(z_prev[None, :], t[:, None])


    @staticmethod
    def adaptative_approx(z_prev, t):
        """
        Adaptive approximate solution of mmp(a, z_prev) = t
        """
        z_prev = np.asarray(z_prev, dtype=float)
        t = np.asarray(t, dtype=float)

        x_bar = z_prev.max()
        b_bar = np.minimum(x_bar, t)
        is_max = z_prev == x_bar

        a = np.random.rand(t.shape[0], z_prev.shape[0])
        a[:, is_max] = b_bar[:, None]
        return a
    
    @property
    def n_layers(self) -> int:
        return len(self.A)

    @property
    def n_inputs(self) -> int:
        return self.sizes[0]

    def forward(self, x):
        """Forward pass through the network
        Parameters
        ----------
        x : np.ndarray
            Input to the network (activations for the first layer, z_0).
        Returns
        -------
        list of np.ndarray
            Activations for each layer
        """

        z = np.asarray(x, dtype=float)
        activations = [z]
        for k in range(self.n_layers):
            z = self.neuron(k, z)
            activations.append(z)
        return activations

    def predict(self, X):
        """List of predictions for each input in X"""
        X = np.asarray(X, dtype=float)
        return np.array([self.forward(x)[-1][0] for x in X])

    def _snapshot(self):
        """Copy of the weights and switches"""
        return (copy.deepcopy(self.A), copy.deepcopy(self.b))

    def _restore(self, snap):
        self.A, self.b = copy.deepcopy(snap[0]), copy.deepcopy(snap[1])

    def fit(self, X, Y, X_eval, Y_eval, lam, lam_b, beta,
            epochs=300, patience=20, delta=1e-3, verbose=False):
        """
        Train the network on the dataset (X, Y) -> forward pass then back propagation.
        Early stopping on R2 is implemented to stop the training if not useful anymore.
        Bests weights are restored at the end.

        Returns
        -------
        list of float
            R2 on test set at each epoch
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        history = []
        best_r2 = -np.inf
        best_snapshot = self._snapshot()
        best_patience_r2 = -np.inf
        wait = 0

        for epoch in range(epochs):
            self._on_epoch_start()
            for x, y in zip(X, Y):
                acts = self.forward(x)
                t = np.array([y], dtype=float)
                for k in reversed(range(self.n_layers)):
                    t = self.backward_layer(k, acts[k], t, lam, beta)

            r2 = r2_score(Y_eval, self.predict(X_eval))
            history.append(r2)
            self._on_epoch_end(epoch)
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
                        print(f"early stop at epoch {epoch} (no +{delta} R2 over {patience} epochs)")
                    break

        self._restore(best_snapshot)
        return history
