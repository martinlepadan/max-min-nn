import time
import numpy as np

from common.metrics import r2_score
from common.network import layer_sizes
from common.data import make_maxmin_dataset, train_val_split


def run(net_cls, dataset, lam, beta, lam_b=0.05, seed=0, hidden=None,
        epochs=300, use_gradient_step=True, adaptative=False, verbose=False,
        plot=False, fig_dir="figures/network"):
    """
    Train one network and return the metrics
    """
    X_tr, Y_tr, X_te, Y_te = dataset
    X_fit, Y_fit, X_val, Y_val = train_val_split(X_tr, Y_tr)
    n = X_tr.shape[1]

    net = net_cls(layer_sizes(n, hidden), seed=seed,
                  use_gradient_step=use_gradient_step, adaptative=adaptative)

    t0 = time.perf_counter()
    history = net.fit(X_fit, Y_fit, X_val, Y_val, lam=lam, lam_b=lam_b, beta=beta,
                      epochs=epochs, verbose=verbose, X_test=X_te, Y_test=Y_te)
    seconds = time.perf_counter() - t0
    feasibility = net.feasibility_rate() if hasattr(net, "feasibility_rate") else None
    feasibility_history = getattr(net, "feasibility_history", None)

    if plot:
        from common.figures import save_run_figures
        save_run_figures(net, Y_tr, net.predict(X_tr), Y_te, net.predict(X_te),
                         net.r2_train_hist, net.r2_test_hist,
                         out_dir=fig_dir, tag=f"seed{seed}")

    return {
        "version": net.version,
        "lam": lam, "beta": beta,
        "adaptative": adaptative,
        "gradient": use_gradient_step,
        "train_r2": r2_score(Y_tr, net.predict(X_tr)),
        "test_r2": r2_score(Y_te, net.predict(X_te)),
        "seconds": seconds,
        "epochs": len(history),
        "history": history,
        "feasibility": feasibility,
        "feasibility_history": feasibility_history,
    }


def run_seeds(net_cls, lam, beta, lam_b=0.05, seeds=range(5), epochs=300,
              use_gradient_step=True, adaptative=False, dataset_fn=make_maxmin_dataset,
              hidden=None, plot=False, fig_dir="figures/network"):
    """
    Run multiple networks with differents seeds to get aggregated results
    """
    results = []
    for i, seed in enumerate(seeds):
        dataset = dataset_fn(seed=seed)
        results.append(run(net_cls, dataset, lam, beta, lam_b=lam_b, seed=seed, hidden=hidden,
                           epochs=epochs, use_gradient_step=use_gradient_step, adaptative=adaptative,
                           plot=(plot and i == 0), fig_dir=fig_dir))

    r2s_test = np.array([r["test_r2"] for r in results])
    r2s_train = np.array([r["train_r2"] for r in results])
    secs = np.array([r["seconds"] for r in results])
    eps = np.array([r["epochs"] for r in results])

    feas = [r["feasibility"] for r in results]
    feasibility = None
    if all(f is not None for f in feas):
        globals_ = np.array([f["global"] for f in feas])
        layers = sorted(feas[0]["per_layer"])
        feasibility = {
            "global_mean": globals_.mean(), "global_std": globals_.std(),
            "per_layer_mean": {lay: float(np.mean([f["per_layer"][lay] for f in feas])) for lay in layers},
        }

    return {
        "version": results[0]["version"],
        "lam": lam, "beta": beta,
        "adaptative": adaptative,
        "gradient": use_gradient_step,
        "r2_mean_train": r2s_train.mean(), "r2_std_train": r2s_train.std(),
        "r2_mean_test": r2s_test.mean(), "r2_std_test": r2s_test.std(),
        "sec_mean": secs.mean(), "sec_std": secs.std(),
        "ep_mean": eps.mean(), "ep_std": eps.std(),
        "feasibility": feasibility,
        "results": results,
    }
