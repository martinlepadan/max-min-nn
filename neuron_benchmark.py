import numpy as np
from single_neuron.neuron import Neuron
from single_neuron.minmax_notes import MinMaxNotesNeuron
from single_neuron.minmax_greatest import MinMaxGreatestNeuron
from fuzzy import equ, mmp
from common.metrics import r2_score
from common.data import _split

SEEDS = (1, 2, 3, 4, 5)
NOISE = 0.0 

def a_star_of(n):
    """Fixed weights of the generator"""
    return np.linspace(0.2, 0.9, n)


def make_data(a_star, b_star, n_samples=300, noise=NOISE, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, len(a_star)))
    gen = Neuron(len(a_star), seed=0, b_init=float(b_star))
    gen.a = np.asarray(a_star, dtype=float)
    Y = gen.predict(X) + rng.normal(0, noise, n_samples)
    return _split(X, np.clip(Y, 0, 1))


def ceiling_r2(b_star, n=5):
    """R2 reachable with the true weights"""
    a_star = a_star_of(n)
    vals = []
    for s in SEEDS:
        _, _, X_te, Y_te = make_data(a_star, b_star, seed=s)
        yhat = np.array([equ(mmp(a_star, x), b_star) for x in X_te])
        vals.append(r2_score(Y_te, yhat))
    return float(np.mean(vals))


def _report(title, b_star, r2tr, r2te, aerr, bs):
    print(f"{title:<20} b*={b_star} -> b={np.mean(bs):4.2f} "
          f"| R2_train={np.mean(r2tr):6.3f} "
          f"| R2_test={np.mean(r2te):6.3f}+/-{np.std(r2te):.2f} "
          f"| mean|a-a*|={np.mean(aerr):.3f}")


def run_old(title, b_star, n=5,
            lam_a=1e-7, beta=0.1, lam_b=0.05, b_init="random",
            method_both="error", epochs=120):
    """First neuron (IPMU)"""
    a_star = a_star_of(n)
    r2tr, r2te, aerr, bs = [], [], [], []
    for s in SEEDS:
        X_tr, Y_tr, X_te, Y_te = make_data(a_star, b_star, seed=s)
        net = Neuron(n, seed=42 + s, b_init=b_init)
        net.fit(X_tr, Y_tr, lam_a=lam_a, lam_b=lam_b, beta=beta,
                method_both=method_both, epochs=epochs)
        r2tr.append(r2_score(Y_tr, net.predict(X_tr)))
        r2te.append(r2_score(Y_te, net.predict(X_te)))
        aerr.append(np.abs(net.a - a_star).mean())
        bs.append(net.b)
    _report(title, b_star, r2tr, r2te, aerr, bs)


def run_notes(title, b_star, n=5):
    a_star = a_star_of(n)
    r2tr, r2te, aerr, bs = [], [], [], []
    for s in SEEDS:
        X_tr, Y_tr, X_te, Y_te = make_data(a_star, b_star, seed=s)
        net = MinMaxNotesNeuron(n, seed=42 + s)
        net.fit(X_tr, Y_tr)
        r2tr.append(r2_score(Y_tr, net.predict(X_tr)))
        r2te.append(r2_score(Y_te, net.predict(X_te)))
        aerr.append(np.abs(net.a - a_star).mean())
        bs.append(net.b)
    _report(title, b_star, r2tr, r2te, aerr, bs)


def run_greatest(title, b_star, n=5, method="min"):
    """Greatest solution of the batch system"""
    a_star = a_star_of(n)
    r2tr, r2te, aerr, bs = [], [], [], []
    for s in SEEDS:
        X_tr, Y_tr, X_te, Y_te = make_data(a_star, b_star, seed=s)
        net = MinMaxGreatestNeuron(n, seed=42 + s)
        net.fit(X_tr, Y_tr, method=method)
        r2tr.append(r2_score(Y_tr, net.predict(X_tr)))
        r2te.append(r2_score(Y_te, net.predict(X_te)))
        aerr.append(np.abs(net.a - a_star).mean())
        bs.append(net.b)
    _report(title, b_star, r2tr, r2te, aerr, bs)


def plot_predictions(neuron_cls, path="predictions.png", n=5, seed=1):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, b_star in zip(axes, (1.0, 0.0)):
        a_star = a_star_of(n)
        X_tr, Y_tr, X_te, Y_te = make_data(a_star, b_star, seed=seed)
        if neuron_cls == Neuron:
            net = neuron_cls(n, seed=42 + seed, b_init="random")
            net.fit(X_tr, Y_tr, lam_a=1e-7, lam_b=0.05, beta=0.1,
                method_both="error", epochs=120)
        else:
            net = neuron_cls(n, seed=42 + seed)
            net.fit(X_tr, Y_tr)
        Y_hat = np.asarray(net.predict(X_te))
        r2 = r2_score(Y_te, Y_hat)
        ax.scatter(Y_te, Y_hat, s=12, alpha=0.5, edgecolor="none")
        ax.plot([0, 1], [0, 1], "r--", lw=1) 
        ax.set_xlabel("target  y")
        ax.set_ylabel("prediction  y_hat")
        ax.set_title(f"b*={b_star:g}  (b={net.b:.3f}, R2={r2:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    print("             Simple compraison")
    print("\nData generated with b=1 ")
    for b_star in (1.0, 0.0):
        run_old("   IPMU:", b_star)
        run_notes("   From the notes:", b_star)
        run_greatest("   Mix", b_star)
        print()
        if b_star == 1.0:
            print("\nData generated with b=0 ")

    plot_predictions(Neuron, path="figures/neuron/predictions.png")
    plot_predictions(MinMaxNotesNeuron, path="figures/neuron/predictions_note.png")
    plot_predictions(MinMaxGreatestNeuron, path="figures/neuron/predictions_greatest.png")


if __name__ == "__main__":
    main()
