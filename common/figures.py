import os

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from common.metrics import r2_score

TRAIN_C = "red"
TEST_C = "C0"


def _learning_curve(ax, r2_train, r2_test):
    ax.plot(r2_train, color=TRAIN_C, lw=1.3, label="train")
    ax.plot(r2_test, color=TEST_C, lw=1.3, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R²")
    lo = min([*r2_train, *r2_test], default=0.0)
    ax.set_ylim(bottom=max(-1.0, lo - 0.05), top=1.02)
    ax.set_ylim(bottom=lo - 0.05, top=1.02)
    ax.legend(frameon=False, fontsize=9)


def _pred_vs_true(ax, y_tr, yhat_tr, y_te, yhat_te):
    ax.scatter(y_tr, yhat_tr, s=12, alpha=0.5, edgecolor="none", color=TRAIN_C, label="train")
    ax.scatter(y_te, yhat_te, s=12, alpha=0.5, edgecolor="none", color=TEST_C, label="test")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("target  y")
    ax.set_ylabel("prediction  y_hat")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def save_run_figures(net, y_tr, yhat_tr, y_te, yhat_te, r2_train, r2_test,
                     out_dir="figures/network", tag=""):

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{net.version}{'_' + tag if tag else ''}"

    fig, ax = plt.subplots(figsize=(5, 3.2))
    _learning_curve(ax, r2_train, r2_test)
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"{stem}_learning.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    _pred_vs_true(ax, y_tr, yhat_tr, y_te, yhat_te)
    ax.set_title(f"R2 train={r2_score(y_tr, yhat_tr):.3f}  test={r2_score(y_te, yhat_te):.3f}")
    fig.tight_layout()
    p2 = os.path.join(out_dir, f"{stem}_predictions.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    return p1, p2
