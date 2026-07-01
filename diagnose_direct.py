"""
Diagnostic of DirectSolveNet : why/where does backpropagation fails

The goal is to test two steps of the backpropagation that may hold the problem:
    - the update of the weights
    - the update of the targets

To test the weights
    We only train the weights of a network (student) by giving him the targets of the network that generated the data (teacher).
    We then compare the weights of the student and the teacher.

To test the targets:
    We freeze the weights of the student to the true weights of the teacher.
    We only propagate the targets and we compare them to the true activations of the teacher.
    Our idea is that, we think that the neuron is always overistemating the targets.
"""
import numpy as np

from common.network import layer_sizes
from common.metrics import r2_score
from fuzzy import equ, mip
from versions.network_direct import DirectSolveNet

N_INPUTS = 12
N_TRAIN = 500
N_TEST = 500


def make_teacher(hidden=None, poles="ones", seed=100):
    sizes = layer_sizes(N_INPUTS, hidden)
    teacher = DirectSolveNet(sizes, seed=seed)
    rng = np.random.default_rng(seed)
    if poles == "ones": # To create a monotone dataset
        teacher.b = [np.ones(s) for s in sizes[1:]]
    else: # To create a non-monotone dataset
        teacher.b = [(rng.random(s) > 0.5).astype(float) for s in sizes[1:]]
    return teacher, sizes


def gen(teacher, n_samples, seed, noise=0.0):
    """Get data and real activations from teacher network"""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, N_INPUTS))
    acts = teacher._forward_batch(X)
    Y = np.clip(teacher.predict(X) + rng.normal(0, noise, n_samples), 0.0, 1.0)
    return X, Y, acts


def test_weights(method="min", poles="ones", hidden=None, noise=0.0, seed=0):
    teacher, sizes = make_teacher(hidden, poles, seed=seed + 100)
    _, _, acts_tr = gen(teacher, N_TRAIN, seed=1, noise=noise)
    X_te, Y_te, _ = gen(teacher, N_TEST, seed=2, noise=noise)

    student = DirectSolveNet(sizes, seed=seed)
    student.method = method # we try it we the min method (not supposed to work well) and with quantil (more robust with inconsistent system)
    student.b = [bk.copy() for bk in teacher.b] # we also fix the switch to the trues ones

    for k in range(student.n_layers):
        z_prev = acts_tr[k] # real input of the layer
        z_k = acts_tr[k + 1] # real output of the layer
        
        # Updating the weights with the real targets
        direct = student.b[k] >= 0.5
        level = np.where(direct[None, :], z_k, 1.0 - z_k)
        student.A[k] = student._solve_weights(z_prev, level, direct)

    return r2_score(Y_te, student.predict(X_te))


def test_targets(poles="ones", hidden=None, seed=0):
    teacher, sizes = make_teacher(hidden, poles, seed=seed + 100)
    X, Y, acts = gen(teacher, N_TRAIN, seed=1)

    student = DirectSolveNet(sizes, seed=seed)
    student.A = [Ak.copy() for Ak in teacher.A]
    student.b = [bk.copy() for bk in teacher.b]

    rows = []
    T = acts[-1]
    for k in reversed(range(student.n_layers)):
        tau = equ(T, student.b[k][None, :])
        t_prev = mip(student.A[k].T[None, :, :], tau[:, None, :])
        z_star = acts[k]
        gap = t_prev - z_star # difference between the calculated target and the real activation
        rows.append({
            "k": k,
            "mean_gap": float(gap.mean()),
            "frac_pos": float((gap > 1e-9).mean()),
            "frac_neg": float((gap < -1e-9).mean()),
            "frac_zero": float((np.abs(gap) <= 1e-9).mean()),
        })
        T = t_prev
    return list(reversed(rows))


def main():
    print("Testing the weights")
    for poles in ("ones", "random"):
        print(f"\n  b* = {poles}")
        for hidden in (None, [24, 12], [5, 5], []):
            arch = layer_sizes(N_INPUTS, hidden)
            r_min = np.mean([test_weights("min", poles, hidden, seed=s) for s in range(3)])
            r_q = np.mean([test_weights("quantile", poles, hidden, seed=s) for s in range(3)])
            print(f"    {str(arch):22s}  R²(min)={r_min:6.3f}   R²(quantile)={r_q:6.3f}")

    print("\n\n")
    print("Testing the targets")
    print("gap = t_prev - z*_{k-1}")
    for poles in ("ones", "random"):
        for hidden in (None, [5, 5]):
            arch = layer_sizes(N_INPUTS, hidden)
            print(f"\n  b*={poles}  archi={arch}")
            rows = test_targets(poles, hidden, seed=0)
            for r in rows:
                print(f"    layer {r['k']}: mean gap={r['mean_gap']:+.3f}  "
                      f"pos={r['frac_pos']:.0%} neg={r['frac_neg']:.0%} zero={r['frac_zero']:.0%}")


if __name__ == "__main__":
    main()
