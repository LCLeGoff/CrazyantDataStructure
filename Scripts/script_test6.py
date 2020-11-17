import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as scopt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
import statsmodels.sandbox.stats.runs as smr

mu_data = 0
kappa_data = 2
kappa = 15
p = 0.16

s = int(1e5)
n = int(1/p)
thetas = np.arange(-np.pi, np.pi, 0.1)
x = (thetas[1:]+thetas[:-1])/2.

tab_data = np.random.vonmises(mu=mu_data, kappa=kappa_data, size=s)
plt.plot(x, scs.vonmises.pdf(x, kappa=kappa_data), label='info')
cols = ['k', 'r', 'y', 'orange', 'grey', 'green', 'b', 'violet', 'navy', 'darkred']
for n in range(1, 10):
    tab_noise = np.random.vonmises(mu=0, kappa=kappa, size=(s, n))
    res = np.angle(np.exp(1j*(tab_data+np.sum(tab_noise, axis=1))))

    y, _ = np.histogram(res, thetas, density=True)
    plt.plot(x, y, 'o', label='data', c=cols[n-1])

    kappa_fit, _ = scopt.curve_fit(lambda z, k: scs.vonmises.pdf(z, k), x, y)
    kappa_fit = kappa_fit[0]
    plt.plot(x, scs.vonmises.pdf(x, kappa=kappa_fit), '--', c=cols[n-1],
             label='n=%i, fit2: %.2f' % (n, round(kappa_fit, 2)))

    kappa_test = 1/(1/kappa_data+1/kappa*n)
    plt.plot(x, scs.vonmises.pdf(x, kappa=kappa_test),  c=cols[n-1],
             label='n=%i, test: %.2f' % (n, round(kappa_test, 2)))


plt.legend()
plt.show()

# kappa_bar = 0.5
# kappa = 3
#
# # dx = 0.05
# s = int(1e6)
# thetas = np.arange(-np.pi, np.pi, 0.2)
# x = (thetas[1:]+thetas[:-1])/2.
#
# cols = ['k', 'r', 'y', 'orange', 'grey', 'green', 'b', 'violet', 'navy', 'darkred']
# # for i, p in enumerate(np.arange(kappa_bar/kappa+0.001, 1, dx)):
# dx = 5
# for i, kappa_data in enumerate(np.arange(kappa_bar+0.1, 100, dx)):
#     # kappa_data = 1/(1/kappa_bar-1/kappa/p)
#     p = 1/(kappa*(1/kappa_bar-1/kappa_data))
#     n = int(1/p)
#     print(p, n)
#
#     tab_data = np.random.vonmises(mu=0, kappa=kappa_data, size=s)
#     tab_noise = np.random.vonmises(mu=0, kappa=kappa, size=(s, n))
#     res = np.angle(np.exp(1j*(tab_data+np.sum(tab_noise, axis=1))))
#
#     y, _ = np.histogram(res, thetas, density=True)
#     plt.plot(x, y, label='p=%.2f' % p, c=cols[i % 10])
#
# plt.legend()
# plt.show()


# def mst_edges(V, k):
#     """
#     Construct the approximate minimum spanning tree from vectors V
#     :param: V: 2D array, sequence of vectors
#     :param: k: int the number of neighbor to consider for each vector
#     :return: V ndarray of edges forming the MST
#     """
#
#     # k = len(X)-1 gives the exact MST
#     k = min(len(V) - 1, k)
#
#     # generate a sparse graph using the k nearest neighbors of each point
#     G = kneighbors_graph(V, n_neighbors=k, mode='distance')
#
#     # Compute the minimum spanning tree of this graph
#     full_tree = minimum_spanning_tree(G, overwrite=True)
#
#     return np.array(full_tree.nonzero()).T
#
#
# def ww_test(X, Y, k=10):
#     """
#     Multi-dimensional Wald-Wolfowitz test
#     :param X: multivariate sample X as a numpy ndarray
#     :param Y: multivariate sample Y as a numpy ndarray
#     :param k: number of neighbors to consider for each vector
#     :return: W the WW test statistic, R the number of runs
#     """
#     m, n = len(X), len(Y)
#     N = m + n
#
#     XY = np.concatenate([X, Y]).astype(np.float)
#
#     # XY += np.random.normal(0, noise_scale, XY.shape)
#
#     edges = mst_edges(XY, k)
#
#     labels = np.array([0] * m + [1] * n)
#
#     c = labels[edges]
#     runs_edges = edges[c[:, 0] == c[:, 1]]
#
#     # number of runs is the total number of observations minus edges within each run
#     R = N - len(runs_edges)
#
#     # expected value of R
#     e_R = ((2.0 * m * n) / N) + 1
#
#     # variance of R is _numer/_denom
#     _numer = 2 * m * n * (2 * m * n - N)
#     _denom = N ** 2 * (N - 1)
#
#     # see Eq. 1 in Friedman 1979
#     # W approaches a standard normal distribution
#     W = (R - e_R) / np.sqrt(_numer/_denom)
#
#     return W, R
#
#
# def get_divergence(x, y):
#     nx = len(x)
#     ny = len(y)
#     # c = smr.runstest_2samp(x, y)[0]
#     c = ww_test(x, y)[0]
#     return 1-c*(nx+ny)/2/nx/ny
#
#
# s = 1000
# std = 1
# k = 1
# mu = [0]*k
# tab_x = np.random.normal(0, std, size=(s, k))
#
# tab_u = []
# tab_v = []
# var_u = 0.05
# for i in range(100):
#     u = np.random.normal(0, scale=np.sqrt(var_u), size=k)
#     temp = [u[i]**2 for i in range(k)]
#     temp += [u[i]*u[j] for i in range(k) for j in range(i+1, k)]
#     tab_u.append(temp)
#
#     tab_y = tab_x+u.transpose()
#     tab_v.append(2*get_divergence(tab_x, tab_y))
#
# # tab_u = np.c_[tab_u]
# # tab_v = np.c_[tab_v]
# # fisher = np.linalg.inv(tab_u.transpose().dot(tab_u)).dot(tab_u.transpose()).dot(tab_v)
# # print(fisher, 1/fisher)
#
# tab_u = np.array(tab_u)
# tab_v = np.c_[tab_v]
#
# print(tab_u[0]**2/tab_v[0])
#
# def f(x):
#     return np.linalg.norm(tab_u.dot(np.c_[x])-tab_v)
#
#
# res = scopt.minimize(f, x0=[1]*int(k*(k+1)/2))
# x = list(res.x)
# mat = np.zeros((k, k))
# for i in range(k):
#     mat[i, i] = x[i]
# for i in range(0, k)[::-1]:
#     for j in range(i+1, k)[::-1]:
#         temp = x.pop()
#         mat[j, i] = temp
#         mat[i, j] = temp
#
# print(mat)
# print(1/mat)

# def f2(x):
#     return np.linalg.norm(tab_u.transpose().dot(np.c_[x]).dot(tab_u)-tab_v)
#
#
# res = scopt.minimize(f2, x0=[1]*len(tab_u))
# print(res)
