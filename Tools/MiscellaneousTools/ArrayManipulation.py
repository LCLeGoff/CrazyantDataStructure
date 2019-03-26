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
    return list(names)


def round_float(x, n):
    f = np.vectorize(lambda y: np.around(y, -np.array(np.floor(np.sign(y) * np.log10(abs(y))), dtype=int) + n - 1))
    return f(x)


def get_interval_containing(value, interval_beginnings):
    i = 0
    while i < len(interval_beginnings) and value >= interval_beginnings[i]:
        i += 1

    i -= 1
    if i == -1:
        return None
    else:
        return interval_beginnings[i]
