import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    c1_list = [0, 0.1, 0.2, 0.5, 1]
    colors = sns.color_palette("Set1", len(c1_list))

    xmin, xmax = -0.15, 4.25
    ymin, ymax = -6, 8

    phi = Function(scale=-1, offset=-2)
    _phi = Function(scale=-1, offset=-2, no_grad=True)
    alpha = np.linspace(xmin, xmax, 1000)
    phi_alpha = phi(alpha)
    
    for c1, color in zip(c1_list, colors):
        plt.figure(figsize=(12,8))

        plt.plot(alpha, phi_alpha, color='black')
        plt.scatter([0], [_phi(0)], color='black')

        bound = _phi(0) + c1 * alpha * phi.derivative(0)
        mask = (bound >= phi_alpha)
        mask = mask.astype(np.int)
        is_boundary = mask[1:] - mask[:-1]
        is_boundary = is_boundary.astype(np.bool)

        start = None

        for _alpha, _is_boundary in zip(alpha, is_boundary):
            if _is_boundary:
                if start is None:
                    start = _alpha
                else:
                    plt.axvspan(start, _alpha, color=color, alpha=0.2)
                    start = None
        
        if start is not None:
            plt.axvspan(start, _alpha, color=color, alpha=0.2)
            start = None

        plt.plot(alpha, bound, color=color)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\phi(\alpha)$')

        plt.savefig("figures/fig-linear-search_Armijo_c1={}.png".format(c1), bbox_inches='tight')
        plt.close()

    # Total
    plt.figure(figsize=(12,8))

    plt.plot(alpha, phi_alpha, color='black')
    plt.scatter([0], [_phi(0)], color='black')

    for c1, color in zip(c1_list, colors):
        bound = _phi(0) + c1 * alpha * phi.derivative(0)

        plt.plot(alpha, bound, label=r"$c_{1}=$" + r"${}$".format(c1), color=color)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\phi(\alpha)$')
    plt.legend()

    plt.savefig("figures/fig-linear-search_Armijo.png", bbox_inches='tight')
    plt.close()


class Function:
    def __init__(self, scale=1, offset=0, no_grad=False):
        self.x = None
        self.scale = scale
        self.offset = offset
        self.no_grad = no_grad

    def __call__(self, x):
        if self.x is not None and not self.no_grad:
            raise ValueError("Function has already called.")
        scale = self.scale
        offset = self.offset
        self.x = x

        scaled = scale * x - offset

        return scaled**4 - 4 * scaled**2 + scaled

    def derivative(self, x=None, derivative_cum=1):
        if self.no_grad:
            raise ValueError("You are given `no_grad=True`.")
        if self.x is None:
            raise ValueError("Function has not called yet.")
        scale = self.scale
        offset = self.offset
        if x is None:
            x = self.x

        scaled = scale * x - offset

        return derivative_cum * scale * (4 * scaled**3 - 8 * scaled + 1)


if __name__ == '__main__':
    plt.rcParams['font.size'] = 20
    main()