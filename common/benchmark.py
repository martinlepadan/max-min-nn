import time
import numpy as np

from common.metrics import r2_score
from common.network import default_layers
from common.data import make_maxmin_dataset, train_val_split


def run(net_cls, dataset, lam, beta, seed=0,
        epochs=300, use_gradient_step=True, adaptative=False, verbose=False):
    """
    Train one network and return the metrics
    """
    X_tr, Y_tr, X_te, Y_te = dataset
    X_fit, Y_fit, X_val, Y_val = train_val_split(X_tr, Y_tr)
    n = X_tr.shape[1]
    net = net_cls(default_layers(n), seed=seed,
                  use_gradient_step=use_gradient_step, adaptative=adaptative)

    t0 = time.perf_counter()
    history = net.fit(X_fit, Y_fit, X_val, Y_val, lam=lam, beta=beta,
                      epochs=epochs, verbose=verbose)
    seconds = time.perf_counter() - t0

    return {
        "version": net.version,
        "lam": lam, "beta": beta,
        "adaptative": adaptative,
        "gradient": use_gradient_step,
        "train_r2": r2_score(Y_te, net.predict(X_te)),
        "test_r2": r2_score(Y_tr, net.predict(X_tr)),
        "seconds": seconds,
        "epochs": len(history),
        "history": history,
        "feasibility": getattr(net, "feasibility_history", None),
    }


def run_seeds(net_cls, lam, beta, seeds=range(5), epochs=300,
              use_gradient_step=True, adaptative=False, dataset_fn=make_maxmin_dataset):
    """
    Run multiple networks with differents seeds to get aggregated results
    """
    results = []
    for seed in seeds:
        dataset = dataset_fn(seed=seed)
        results.append(run(net_cls, dataset, lam, beta, seed=seed,
                           epochs=epochs, use_gradient_step=use_gradient_step, adaptative=adaptative))

    r2s_test = np.array([r["test_r2"] for r in results])
    r2s_train = np.array([r["train_r2"] for r in results])
    secs = np.array([r["seconds"] for r in results])
    eps = np.array([r["epochs"] for r in results])
    return {
        "version": results[0]["version"],
        "lam": lam, "beta": beta,
        "r2_mean_train": r2s_train.mean(), "r2_std_train": r2s_train.std(),
        "r2_mean_test": r2s_test.mean(), "r2_std_test": r2s_test.std(),
        "sec_mean": secs.mean(), "sec_std": secs.std(),
        "ep_mean": eps.mean(), "ep_std": eps.std(),
        "results": results,
    }
