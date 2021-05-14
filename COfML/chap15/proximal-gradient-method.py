import numpy as np

def prox_L1(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def iterative_shrinkage_thresholdiing_algorithm():
    np.random.seed(42)

    N, M = 50, 20
    gamma = 1e-2
    regularizer = 1e+1
    
    x, Phi, noise = _create_data(N, M, nonzero=5)
    y = Phi @ x + 1e-1 * noise

    x_hat_0 = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    x_hat_prev = x_hat_0
    
    while True:
        x_hat = prox_L1(x_hat_prev - gamma * (Phi.T @ (Phi @ x_hat_prev - y)), gamma=gamma*regularizer)
        if np.linalg.norm(x_hat - x_hat_prev) < 1e-5:
            break
        x_hat_prev = x_hat

    # Figures
    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(x)), x, color='black')
    plt.ylim(y.min() - 5, y.max() + 5)
    plt.savefig("figures/x.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(y)), y, color='black')
    plt.ylim(y.min() - 5, y.max() + 5)
    plt.savefig("figures/y.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(x_hat_0)), x_hat_0, color='black')
    plt.ylim(x_hat_0.min() - 5, x_hat_0.max() + 5)
    plt.savefig("figures/x_hat_0.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(x_hat)), x_hat, color='black')
    plt.ylim(y.min() - 5, y.max() + 5)
    plt.savefig("figures/x_hat.png", bbox_inches='tight')
    plt.close()

def _create_data(N=20, M=10, nonzero=2):
    Phi = np.random.randn(M, N)
    x = np.zeros(N)
    noise = np.random.randn(M)

    indices = np.random.choice(N, nonzero, replace=False)
    values = np.random.randint(1, 10, nonzero)
    signs = np.random.randint(0, 2, nonzero)
    signs = np.where(signs > 0, 1, -1)
    x[indices] = signs * values

    return x, Phi, noise

def main():
    plt.rcParams['font.size'] = 18

    iterative_shrinkage_thresholdiing_algorithm()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()