import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    c1 = 0.1
    c2_list = [0.2, 0.5, 1]
    colors = sns.color_palette("Set1", len(c2_list))

    xmin, xmax = -0.15, 4.25
    ymin, ymax = -6, 8

    phi = Function(scale=-1, offset=-2)
    _phi = Function(scale=-1, offset=-2, no_grad=True)
    alpha = np.linspace(xmin, xmax, 1000)
    phi_alpha = phi(alpha)
    
    for c2 in c2_list:
        plt.figure(figsize=(12,8))

        plt.plot(alpha, phi_alpha, color='black')
        plt.scatter([0], [_phi(0)], color='black')

        upper_bound = _phi(0) + c1 * alpha * phi.derivative(0)
        mask = (upper_bound >= phi_alpha)
        mask = mask.astype(np.int)
        is_boundary_c1 = mask[1:] - mask[:-1]
        is_boundary_c1 = is_boundary_c1.astype(np.bool)

        lower_bound = c2 * phi.derivative(0)
        mask = (phi.derivative() >= lower_bound)
        mask = mask.astype(np.int)
        is_boundary_c2 = mask[1:] - mask[:-1]
        is_boundary_c2 = is_boundary_c2.astype(np.bool)

        start_c1, start_c2 = None, None

        for _alpha, _is_boundary_c1, _is_boundary_c2 in zip(alpha, is_boundary_c1, is_boundary_c2):
            if _is_boundary_c1:
                if start_c1 is None:
                    start_c1 = _alpha
                else:
                    plt.axvspan(start_c1, _alpha, color=colors[0], alpha=0.2)
                    start_c1 = None
            if _is_boundary_c2:
                if start_c2 is None:
                    start_c2 = _alpha
                else:
                    plt.axvspan(start_c2, _alpha, color=colors[1], alpha=0.2)
                    start_c2 = None
        
        if start_c1 is not None:
            plt.axvspan(start_c1, _alpha, color=colors[0], alpha=0.2)
            start_c1 = None
        
        if start_c2 is not None:
            plt.axvspan(start_c2, _alpha, color=colors[1], alpha=0.2)
            start_c2 = None

        plt.plot(alpha, upper_bound, color=colors[0])

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\phi(\alpha)$')

        plt.savefig("figures/fig-linear-search_Wolfe_c1={}_c2={}.png".format(c1, c2), bbox_inches='tight')
        plt.close()
    
      
    for c2 in c2_list:
        plt.figure(figsize=(12,8))

        plt.plot(alpha, phi_alpha, color='black')
        plt.scatter([0], [_phi(0)], color='black')

        upper_bound = _phi(0) + c1 * alpha * phi.derivative(0)
        mask = (upper_bound >= phi_alpha)
        mask = mask.astype(np.int)
        is_boundary_c1 = mask[1:] - mask[:-1]
        is_boundary_c1 = is_boundary_c1.astype(np.bool)

        lower_bound = np.abs(c2 * phi.derivative(0))
        mask = (np.abs(phi.derivative()) >= lower_bound)
        mask = mask.astype(np.int)
        is_boundary_c2 = mask[1:] - mask[:-1]
        is_boundary_c2 = is_boundary_c2.astype(np.bool)

        start_c1, start_c2 = None, None

        for _alpha, _is_boundary_c1, _is_boundary_c2 in zip(alpha, is_boundary_c1, is_boundary_c2):
            if _is_boundary_c1:
                if start_c1 is None:
                    start_c1 = _alpha
                else:
                    plt.axvspan(start_c1, _alpha, color=colors[0], alpha=0.2)
                    start_c1 = None
            if _is_boundary_c2:
                if start_c2 is None:
                    start_c2 = _alpha
                else:
                    plt.axvspan(start_c2, _alpha, color=colors[1], alpha=0.2)
                    start_c2 = None
        
        if start_c1 is not None:
            plt.axvspan(start_c1, _alpha, color=colors[0], alpha=0.2)
            start_c1 = None
        
        if start_c2 is not None:
            plt.axvspan(start_c2, _alpha, color=colors[1], alpha=0.2)
            start_c2 = None

        plt.plot(alpha, upper_bound, color=colors[0])

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\phi(\alpha)$')

        plt.savefig("figures/fig-linear-search_strong-Wolfe_c1={}_c2={}.png".format(c1, c2), bbox_inches='tight')
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