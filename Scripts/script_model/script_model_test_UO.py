from matplotlib import colors

import numpy as np
import scipy.stats as scs
import pylab as pb
import random as rd


# def fct(q, n):
#     res = np.zeros((len(q)))
#     mask = q <= 0.5
#     res[mask] = np.pi * (1 - 2 * q[mask]) ** (1 / (n[mask] + 1))
#     mask = q > 0.5
#     res[mask] = -np.pi * (2 * q[mask] + 1) ** (1 / (n[mask] + 1))
#     return res


n_exp = 1000
lg_exp = 300
# res = np.zeros((n_exp, lg_exp))
# for i in range(n_exp):
#     res[i, 0] = np.pi*(1-2*scs.uniform.rvs())
#     res[i, 1:] = scs.uniform.rvs(size=lg_exp-1)
#     res[i, 1:] = fct(res[i, 1:], np.arange(1, lg_exp))
#
#     # y = scs.expon.rvs(scale=1/0.8, size=lg_exp-1)
#     # z = 1-2*scs.bernoulli.rvs(p=0.5, size=lg_exp-1)
#     # res[i, 1:] = y*z
#
#     # res[i, :] = np.cumsum(res[i, :])
#
# for j in list(range(lg_exp)):
#     y, x = pb.histogram(res[:, j], normed=True, bins=np.arange(-np.pi, np.pi, .1))
#     pb.plot(x[:-1], y, marker='o')
# # pb.yscale('log')
# pb.show()
#
# res = np.zeros((n_exp, lg_exp))
# dloc = 0.1
# for i in range(n_exp):
#     print(i, n_exp)
#     phero = dict()
#     phero[0.] = 0
#
#     p_attach = 0.5
#     # loc = round(np.pi*(1-2*scs.uniform.rvs()), 1)
#     loc = 0
#
#     res[i, 0] = loc
#     tab_u = scs.uniform.rvs(size=lg_exp-1)
#     tab_v = scs.uniform.rvs(size=lg_exp-1)
#     for j in range(1, lg_exp):
#         u = tab_u[j-1]
#         v = tab_v[j-1]
#
#         if loc not in phero:
#             phero[loc] = 1
#         else:
#             phero[loc] += 1
#
#         loc1 = round(loc + dloc, 1)
#         loc2 = round(loc - dloc, 1)
#
#         if u < p_attach:
#             if loc < 0:
#                 loc = loc1
#             elif loc > 0:
#                 loc = loc2
#             else:
#                 if v < 0.5:
#                     loc = loc1
#                 else:
#                     loc = loc2
#         else:
#             if loc1 not in phero:
#                 phero[loc1] = 0
#             if loc2 not in phero:
#                 phero[loc2] = 0
#
#             p = (1 + phero[loc1]) / float(2 + phero[loc1] + phero[loc2])
#             if v < p:
#                 loc = loc1
#             else:
#                 loc = loc2
#
#         res[i, j] = loc
#
# dx = 0.1
# bins = np.arange(-10+dx/2., 10, dx)
# x_bins = (bins[1:]+bins[:-1])/2.
#
# times = list(range(0, lg_exp, 1))
# norm = colors.Normalize(0, len(times) - 1)
# cmap = pb.get_cmap('jet')
#
# for i, j in enumerate(times):
#     y, x = pb.histogram(res[:, j], normed=True, bins=bins)
#     mask = np.where(y != 0)[0]
#     pb.plot(x_bins[mask], y[mask], marker='o', c=cmap(norm(i)))
# pb.yscale('log')
# pb.axvline(0)
# pb.show()


res = np.zeros((n_exp, lg_exp))
dloc = 0.1
c = 0
p_attach = 0.1
p_persistence = 0.


def evaporation():
    if loc not in phero:
        phero[loc] = 0
    else:
        phero[loc] -= 1
        phero[loc] = max(0, phero[loc])


def marking():
    if loc not in phero:
        phero[loc] = c + 1
    else:
        phero[loc] += 1


for i in range(n_exp):
    print(i, n_exp)
    phero = dict()
    phero[0.] = c

    loc = round(np.pi*(1-2*scs.uniform.rvs()), 1)
    # loc = 0

    res[i, 0] = loc
    # tab_u = scs.uniform.rvs(size=lg_exp-1)
    # tab_v = scs.uniform.rvs(size=lg_exp-1)
    # tab_w = scs.uniform.rvs(size=lg_exp-1)
    for j in range(1, lg_exp):
        # u = tab_u[j-1]
        # v = tab_v[j-1]
        # w = tab_w[j-1]
        u = rd.random()
        v = rd.random()
        w = rd.random()

        loc1 = round(loc + dloc, 1)
        loc2 = round(loc - dloc, 1)

        if u < p_attach:

            # marking()

            if loc < 0:
                loc = loc1
            elif loc > 0:
                loc = loc2
            # else:
            #     if v < 0.5:
            #         loc = loc1
            #     else:
            #         loc = loc2
        else:

            # evaporation()
            if w < p_persistence:
                if res[i, j-2] < loc:
                    loc = loc1
                else:
                    loc = loc2
            else:

                if loc1 not in phero:
                    phero[loc1] = c
                if loc2 not in phero:
                    phero[loc2] = c

                p = (1 + phero[loc1]) / float(2 + phero[loc1] + phero[loc2])
                if p != 0.5:
                    print('!')
                if v < p:
                    loc = loc1
                else:
                    loc = loc2

        res[i, j] = loc

dx = .5
lim = 10
bins = np.arange(-lim+dx/2., lim, dx)
x_bins = (bins[1:]+bins[:-1])/2.

times = list(range(0, lg_exp, 10))
norm = colors.Normalize(0, len(times) - 1)
cmap = pb.get_cmap('jet')

for i, j in enumerate(times):
    y, x = pb.histogram(res[:, j], normed=True, bins=bins)
    mask = np.where(y != 0)[0]
    pb.plot(x_bins[mask], y[mask], marker='o', c=cmap(norm(i)))
pb.yscale('log')
pb.axvline(0)
pb.axvline(-np.pi)
pb.axvline(np.pi)
x = np.arange(0, lim, dx)
pb.plot(x, 0.78*np.exp(-0.78*x))
pb.show()
