import numpy as np

from versions.network_v1 import V1Net
from versions.network_v2 import V2Net
from common.network import default_layers


def _split(X, Y, frac_train=0.5):
    """Split the dataset in half for training and testing."""
    n_train = int(len(X) * frac_train)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def train_val_split(X, Y, frac_val=0.2):
    """
    Create a validation test in the training set
    """
    n_val = int(len(X) * frac_val)
    return X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]


def make_maxmin_dataset(version="V1", n_samples=1000, n_inputs=12, noise=0.01, seed=0):
    """
    Dataset 1: targets produced by a pure max-min network (monotone function) with added noise.
    
    Returns
    -------
    tuple of np.ndarray
        (X_train, Y_train, X_test, Y_test)
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_inputs))

    sizes = default_layers(n_inputs, generate=True)

    if version == "V1":
        net = V1Net(sizes, seed=seed + 100)

    if version == "V2":
        net = V2Net(sizes, seed=seed + 100)
        net.b = [rng.integers(0, 2, size=bk.shape[0]).astype(float) for bk in net.b] 

    Y = net.predict(X)

    Y = Y + rng.normal(0.0, noise, size=Y.shape)
    Y = np.clip(Y, 0.0, 1.0)

    return _split(X, Y)

# TODO:
# def load_dataset -> Dataset avec données réelles 