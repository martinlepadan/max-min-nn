from common.benchmark import run_seeds, run  # noqa: F401
from common.data import make_maxmin_dataset
from versions.network_v1 import V1Net
from versions.network_v2 import V2Net
from versions.network_v3 import V3Net

def report(agg):
    print(f"[{agg['version']}] | adaptative: {agg['adaptative']} | gradient: {agg['gradient']} | lambda={agg['lam']} | beta={agg['beta']}")
    print(f"train R2 = {agg['r2_mean_train']:.3f} +/- {agg['r2_std_train']:.3f}")
    print(f"test R2 = {agg['r2_mean_test']:.3f} +/- {agg['r2_std_test']:.3f}")
    print(f"time = {agg['sec_mean']:.1f}s +/- {agg['sec_std']:.1f}s")
    print(f"epochs = {agg['ep_mean']:.0f} +/- {agg['ep_std']:.0f}")
    feas = agg["feasibility"]
    if feas is None:
        print("feasibility  = N/A")
    else:
        per_layer = "  ".join(f"L{k}={r:.2f}" for k, r in feas["per_layer_mean"].items())
        print(f"feasibility  = global {feas['global_mean']:.3f} +/- {feas['global_std']:.3f}  |  per-layer: {per_layer}")


def unique_report(iter):
    print(f"[{iter['version']}] | adaptative: {iter['adaptative']} | gradient: {iter['gradient']} | lambda={iter['lam']} | beta={iter['beta']}")
    print(f"train R2 = {iter['train_r2']:.3f}")
    print(f"test R2 = {iter['test_r2']:.3f}")
    print(f"time = {iter['seconds']:.1f}s")
    print(f"epochs = {iter['epochs']:.0f}")
    feas = iter["feasibility"]
    if feas is None:
        print("feasibility  = N/A")
    else:
        per_layer = "  ".join(f"L{k}={r:.2f}" for k, r in feas["per_layer"].items())
        print(f"feasibility  = global {feas['global']:.3f}  |  per-layer: {per_layer}")


def main():

    for net_cls in (V1Net, V2Net, V3Net):
        # report(run_seeds(net_cls, lam=1e-7, beta=1e-2, seeds=range(5),
        #                 epochs=300, use_gradient_step=True, adaptative=True))
        unique_report(run(net_cls, make_maxmin_dataset(), lam=1e-7, beta=1e-2, adaptative=True))

    # unique_report(run(V1Net, make_maxmin_dataset(), lam=1e-7, beta=1e-2, adaptative=True))


if __name__ == "__main__":
    main()
