import scipy as sc
import scipy.optimize as scopt
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
    if cst is False:

        log_y = np.log(y)
        [af, bf] = sc.polyfit(x, log_y, 1)
        log_y_fit = sc.polyval([af, bf], x)
        x_fit = x
        y_fit = np.exp(log_y_fit)
        return af, bf, x_fit, y_fit
    else:
        fct = lambda t, a, b, c: b*np.exp(a*t)+c
        res = scopt.curve_fit(fct,  x,  y, p0=cst)
        af, bf, cf = res[0]
        x_fit = x
        y_fit = fct(x_fit, af, bf, cf)
        return af, bf, cf, x_fit, y_fit


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


def gauss_fit(x, y):
    x0 = x[np.nanargmax(y)]
    x_squared = x-x0
    mask = np.where(x_squared >= 0)[0]
    x_squared = (x_squared[mask]-x0)**2

    log_y = np.log(y)
    log_y = log_y[mask]

    [af, bf] = sc.polyfit(x_squared, log_y, 1)
    log_y_fit = sc.polyval([af, bf], x_squared)

    x_fit = np.sqrt(x_squared)+x0
    y_fit = np.exp(log_y_fit)
    return af, bf, x_fit, y_fit
