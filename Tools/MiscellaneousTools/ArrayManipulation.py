import numpy as np


def rolling_sum(x, n):
    x2 = np.insert(x, 0, np.zeros(n + 1))
    s = np.nancumsum(x2)
    s = np.array(list(s) + list(np.full(n, s[-1])))
    res = s[n:] - s[:-n]
    n2 = int(np.floor(n / 2) + 1)
    res = res[n2:-n2]
    return res


def rolling_mean(x, n):
    rs = rolling_sum(x, n)

    ones = np.ones(len(x))
    mask = np.where(np.isnan(x))[0]
    ones[mask] = np.nan
    rn = rolling_sum(ones, n)

    return rs/rn


def rolling_sum_angle(x, n):
    x2 = np.insert(x, 0, np.zeros(n + 1))
    x2 = np.exp(1j*x2)
    s = np.nancumsum(x2)
    s = np.array(list(s) + list(np.full(n, s[-1])))
    res = s[n:] - s[:-n]
    n2 = int(np.floor(n / 2) + 1)
    res = np.angle(res[n2:-n2])
    return res


def rolling_mean_angle(x, n):
    x2 = np.insert(x, 0, np.zeros(n + 1))
    x2 = np.exp(1j*x2)
    s = np.nancumsum(x2)
    s = np.array(list(s) + list(np.full(n, s[-1])))
    res = s[n:] - s[:-n]
    n2 = int(np.floor(n / 2) + 1)

    ones = np.ones(len(x))
    mask = np.where(np.isnan(x))[0]
    ones[mask] = np.nan
    rn = rolling_sum(ones, n)

    res = np.angle(res[n2:-n2]/rn)

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


def get_index_interval_containing(value, interval_beginnings):
    i = 0
    while i < len(interval_beginnings) and value >= interval_beginnings[i]:
        i += 1

    i -= 1
    if i == -1:
        return None
    else:
        return i


def auto_corr(tab):
    x = np.array(tab).ravel()
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-len(tab):]
    return r / (variance * (np.arange(len(tab), 0, -1)))


def get_entropy(tab, get_variance=False):
    tab2 = np.array(tab)
    mask = np.where(tab2 != 0)[0]
    tab2 = tab2[mask]

    h = -np.sum(tab2 * np.log2(tab2))
    if get_variance is True:
        v = np.sum(tab2 * np.log2(tab2)**2)
        n = len(tab2)
        v = (v-h**2)/float(n)
        return h, v
    else:
        return h


def get_max_entropy(tab):
    n = len(tab)
    tab2 = np.ones(n)/float(n)
    return get_entropy(tab2)


def smooth(tab, window):
    tab2 = np.zeros(len(tab))
    nbr = np.zeros(len(tab))
    for i in range(len(tab)):
        i0 = max(0, i-window)
        i1 = min(len(tab), i+window)
        tab2[i] += np.sum(tab[i0:i1])
        nbr[i] = len(tab[i0:i1])

    return tab2/nbr


def log_range(x_min, x_max, nbr):
    return list(
        np.logspace(np.log(x_min) / np.log(10), np.log(x_max) / np.log(10), nbr, endpoint=True))
