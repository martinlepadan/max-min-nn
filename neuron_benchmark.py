import numpy as np
from single_neuron.neuron import Neuron
from common.metrics import r2_score
from common.data import _split


def make_data(a_star, b_star, n_samples=400, noise=0.01, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, len(a_star)))
    gen = Neuron(len(a_star), seed=0, b_init=b_star)
    gen.a = np.asarray(a_star, dtype=float)
    Y = gen.predict(X) + rng.normal(0, noise, n_samples)

    return _split(X, np.clip(Y, 0, 1))


def run_case(title, b_star, lam_a, beta, lam_b, b_init, n=5, epochs=500, seed=1, method_both="error"):
    a_star = np.linspace(0.2, 0.9, n) # we fix the weights of the generator to see if the neuron can approach them
    X_tr, Y_tr, X_te, Y_te = make_data(a_star, b_star, seed=seed)
    net = Neuron(n, seed=42, b_init=b_init)
    net.fit(X_tr, Y_tr, lam_a=lam_a, beta=beta, lam_b=lam_b, epochs=epochs, method_both=method_both)
    a_err = np.abs(net.a - a_star).mean()
    print(f"{title:<30} b*={b_star} -> b={net.b:5.3f} | R2_train={r2_score(Y_tr, net.predict(X_tr)):6.3f} "
          f"| R2_test={r2_score(Y_te, net.predict(X_te)):6.3f} | mean|a-a*|={a_err:.3f}")



def main(tests=[1, 2, 3, 4]):
    print("              Benchmark on a single neuron\n")

    if 1 in tests:
        print("First test | Test if b can go from a polarity to another")

        print("   With b moving to the closest pole:")
        run_case("      b*=0 from b_init=1", b_star=0.0, lam_a=1e-7, beta=1e-2, lam_b=0.05, b_init=1.0, method_both="closest")
        run_case("      b*=1 from b_init=0", b_star=1.0, lam_a=1e-7, beta=1e-2, lam_b=0.05, b_init=0.0, method_both="closest")

        print("   With b moving according to the error:")
        run_case("       b*=0 from b_init=1", b_star=0.0, lam_a=1e-7, beta=1e-2, lam_b=0.05, b_init=1.0, method_both="error")
        run_case("       b*=1 from b_init=0", b_star=1.0, lam_a=1e-7, beta=1e-2, lam_b=0.05, b_init=0.0, method_both="error")

    if 2 in tests:
        print("\nSecond test | Test multiple betas (b is initiliazed at random)")
        print("   With b*=0:")
        for beta in [1e-2, 1e-1, 0.5]:
            run_case(f"       beta={beta:g}, 500 ep", b_star=1.0, lam_a=1e-7, beta=beta, lam_b=0.05, b_init="random", method_both="error")

        print("   With b*=1:")
        for beta in [1e-2, 1e-1, 0.5]:
            run_case(f"       beta={beta:g}, 500 ep", b_star=0.0, lam_a=1e-7, beta=beta, lam_b=0.05, b_init="random", method_both="error")

    if 3 in tests:
        print("\nThird test | Is the gradient step useful ?")
        run_case("  Step1 only (lam_a=0.05)", b_star=1.0, lam_a=0.05, beta=0.0, lam_b=0.05, b_init="random")
        run_case("  Step2 only (beta=0.1)", b_star=1.0, lam_a=0.0, beta=0.1, lam_b=0.05, b_init="random")
        run_case("  Both steps (beta=0.1, lam_a=0.05)", b_star=1.0, lam_a=0.05, beta=0.2, lam_b=0.05, b_init="random")

    if 4 in tests:
        print("\nFourth test | Do we need a separate lamba for b ?")
        run_case("  With lambda_b = lambda_a", b_star=1.0, lam_a=1e-7, beta=0.0, lam_b=1e-7, b_init="random")
        run_case("   With lambda_b =|= lambda_a", b_star=1.0, lam_a=1e-7, beta=0.1, lam_b=0.01, b_init="random")


if __name__ == "__main__":
    main([4])