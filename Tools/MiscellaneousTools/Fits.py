import scipy as sc
import scipy.optimize as scopt
import scipy.stats as scs
import numpy as np


def cst_fit(x, y):
    af = sc.polyfit(x, y, 0)
    y_fit = np.ones(len(x))*af
    x_fit = x
    return af, x_fit, y_fit


def linear_fit(x, y):
    [af, bf] = sc.polyfit(x, y, 1)
    y_fit = sc.polyval([af, bf], x)
    x_fit = x
    return af, bf, x_fit, y_fit


def exp_fit(x, y, cst=False):
    if cst is False or cst is None:

        fct = lambda t, a, b: b*np.exp(a*t)
        res = scopt.curve_fit(fct,  x,  y, p0=(-1, 0))
        af, bf = res[0]
        x_fit = x
        y_fit = fct(x_fit, af, bf)
        return af, bf, x_fit, y_fit
    else:
        fct = lambda t, a, b, c: b*np.exp(a*t)+c
        res = scopt.curve_fit(fct,  x,  y, p0=cst)
        af, bf, cf = res[0]
        x_fit = x
        y_fit = fct(x_fit, af, bf, cf)
        return af, bf, cf, x_fit, y_fit


def exp_fct(x, a, b, c=0):
    return b*np.exp(a*x)+c


def power_fit(x, y):
    log_x = np.log10(x)
    log_y = np.log10(y)
    [af, bf] = sc.polyfit(log_x, log_y, 1)
    log_y_fit = sc.polyval([af, bf], log_x)
    x_fit = x
    y_fit = 10 ** log_y_fit
    return af, bf, x_fit, y_fit


def log_fit(x, y):
    log_x = np.log(x)
    [af, bf] = sc.polyfit(log_x, y, 1)
    y_fit = sc.polyval([af, bf], log_x)
    x_fit = x
    return af, bf, x_fit, y_fit


def inverse_fit(x, y):
    inverse_y = 1/y
    [af, bf] = sc.polyfit(x, inverse_y, 1)
    inverse_y_fit = sc.polyval([af, bf], x)
    x_fit = x
    y_fit = 1/inverse_y_fit
    return af, bf, x_fit, y_fit


def fct_inverse_cst(x, a, b):
    return a/x+b


def inverse_cst_fit(x, y, p0):
    popt, _ = scopt.curve_fit(fct_inverse_cst, x, y, p0=p0)
    a, b = popt

    x_fit = x
    y_fit = fct_inverse_cst(x, a, b)
    return a, b, x_fit, y_fit


def fct_gauss(x, c, x0, s):
    return c*np.exp(-(x-x0)**2/(2*s**2))


def fct_gauss_cst(x, c, x0, s, d):
    return c*np.exp(-(x-x0)**2/(2*s**2))+d


def centered_fct_gauss(x, c, s):
    return c*np.exp(-x**2/(2*s**2))


def centered_fct_gauss_cst(x, c, s, d):
    return c*np.exp(-x**2/(2*s**2))+d


def fct_von_mises(x, kappa):
    return scs.vonmises.pdf(x, kappa=kappa)


def fct_von_mises_cst(x, kappa, b, k):
    return b*np.exp(kappa*np.cos(x))+k


def gauss_fit(x, y):
    popt, _ = scopt.curve_fit(fct_gauss, x, y)
    c, x0, s = popt

    x_fit = x
    y_fit = fct_gauss(x, c, x0, s)
    return c, x0, s, x_fit, y_fit


def gauss_cst_fit(x, y):
    popt, _ = scopt.curve_fit(fct_gauss_cst, x, y)
    c, x0, s, d = popt

    x_fit = x
    y_fit = fct_gauss_cst(x, c, x0, s, d)
    return c, x0, s, d, x_fit, y_fit


def centered_gauss_fit(x, y, p0=None):
    popt, _ = scopt.curve_fit(centered_fct_gauss, x, y, p0=p0)
    c, s = popt

    x_fit = x
    y_fit = centered_fct_gauss(x, c, s)
    return c, s, x_fit, y_fit


def vonmises_fit(x, y, p0=None):
    popt, _ = scopt.curve_fit(fct_von_mises, x, y, p0=p0)
    kappa = popt

    x_fit = x
    y_fit = fct_von_mises(x, kappa)
    return kappa, x_fit, y_fit


def vonmises_cst_fit(x, y, p0=None):
    popt, _ = scopt.curve_fit(fct_von_mises_cst, x, y, p0=p0)
    kappa, b, k = popt

    x_fit = x
    y_fit = fct_von_mises_cst(x, kappa, b, k)
    return kappa, b, k, x_fit, y_fit


def centered_gauss_cst_fit(x, y, p0=None):
    popt, _ = scopt.curve_fit(centered_fct_gauss_cst, x, y, p0=p0)
    c, s, d = popt

    x_fit = x
    y_fit = centered_fct_gauss_cst(x, c, s, d)
    return c, s, d, x_fit, y_fit


def laplace_fct(x, a, b):
    return b*np.exp(-a*np.abs(x))


def laplace_fit(x, y, p0=None):
    popt, _ = scopt.curve_fit(laplace_fct, x, y, p0=p0)
    a, b = popt

    x_fit = x
    y_fit = laplace_fct(x, a, b)
    return a, b, x_fit, y_fit


def uniform_vonmises_fct(x, q, kappa):
    return q*scs.vonmises.pdf(x, kappa)+(1-q)/2/np.pi


def uniform_vonmises_fit(x, y, p0=None):
    popt, _ = scopt.curve_fit(uniform_vonmises_fct, x, y, p0=p0)
    q, kappa = popt

    x_fit = x
    y_fit = uniform_vonmises_fct(x, q, kappa)
    return q, kappa, x_fit, y_fit
