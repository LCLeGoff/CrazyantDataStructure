import numpy as np


def running_mean(x, n):
    x2 = np.insert(x, 0, np.zeros(n + 1))
    s = np.cumsum(x2)
    s = np.array(list(s) + list(np.full(n, s[-1])))
    res = s[n:] - s[:-n]
    n2 = int(np.floor(n / 2) + 1)
    res = res[n2:-n2]
    return res


def turn_to_list(names):
    if isinstance(names, str):
        names = [names]
    return names