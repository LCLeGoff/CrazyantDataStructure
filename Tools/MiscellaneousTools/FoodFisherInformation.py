import numpy as np
import scipy.stats as scs
import scipy.optimize as scopt
import scipy.integrate as scint


def uniform_von_mises(x, q, kappa):
    return q*scs.vonmises.pdf(x, kappa=kappa)+(1-q)/(2*np.pi)


def uniform_von_mises_diff(x, q, kappa):
    return q*kappa*np.sin(x)*scs.vonmises.pdf(x, kappa=kappa)


def fit_uniform_von_mises(tab):
    dtheta = .1
    thetas = np.arange(-np.pi, np.pi+dtheta, dtheta)
    thetas2 = (thetas[1:]+thetas[:-1])/2.
    tab2 = list(tab)+list(-tab)
    y, x = np.histogram(tab2, thetas, density=True)

    fit = scopt.curve_fit(uniform_von_mises, thetas2, y, bounds=((0, 0), (1, np.inf)))
    q, kappa = fit[0]
    cov = fit[1]

    return q, kappa, cov


def compute_fisher_information_uniform_von_mises(tab, get_fit_quality=False):
    q, kappa, cov = fit_uniform_von_mises(tab)

    def fisher_fct(x):
        y = uniform_von_mises_diff(x, q, kappa) ** 2 / uniform_von_mises(x, q, kappa)
        return y

    res = scint.quad(fisher_fct, -np.pi, np.pi)

    if get_fit_quality is True:
        return res[0], q, kappa, cov
    else:
        return res[0]


def compute_fisher_information_uniform_von_mises_fix(q, kappa):

    if q == 0 or kappa == 0:
        return 0
    else:
        def fisher_fct(x):
            y = uniform_von_mises_diff(x, q, kappa) ** 2 / uniform_von_mises(x, q, kappa)
            return y
        res = scint.quad(fisher_fct, -np.pi, np.pi)
        return res[0]

