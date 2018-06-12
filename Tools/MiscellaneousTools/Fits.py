import scipy as sc
import numpy as np


def linear_fit(x, y):
    [af, bf] = sc.polyfit(x, y, 1)
    y_fit = sc.polyval([af, bf], x)
    x_fit = x
    return af, bf, x_fit, y_fit


def exp_fit(x, y):
    log_y = np.log(y)
    [af, bf] = sc.polyfit(x, log_y, 1)
    log_y_fit = sc.polyval([af, bf], x)
    x_fit = x
    y_fit = np.exp(log_y_fit)
    return af, bf, x_fit, y_fit


def power_fit(x, y):
    log_x = np.log10(x)
    log_y = np.log10(y)
    [af, bf] = sc.polyfit(log_x, log_y, 1)
    log_y_git = sc.polyval([af, bf], log_x)
    x_fit = x
    y_fit = 10 ** log_y_git
    return af, bf, x_fit, y_fit
