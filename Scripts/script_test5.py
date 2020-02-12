import numpy as np
import pylab as pb
import pandas as pd
import random as rd
import scipy.stats as scs


add = '/data/Dropbox/POSTDOC/exp_triangle/Analyses_exp_janv_2013/Analyses_2017_01_15/' \
      'Analyses_2fps/Results/stays/stays_corr_ch_h2Tot.csv'
stay_tab_ch = list(pd.read_csv(add, index_col=['id_t', 'id_ant'], delimiter='\t').astype(int).values.ravel())
add = '/data/Dropbox/POSTDOC/exp_triangle/Analyses_exp_janv_2013/Analyses_2017_01_15/' \
      'Analyses_2fps/Results/stays/stays_corr_gl_h2Tot.csv'
stay_tab_gl = list(pd.read_csv(add, index_col=['id_t', 'id_ant'], delimiter='\t').astype(int).values.ravel())
stay_tab_ch = list(np.ones(len(stay_tab_ch)))
stay_tab_gl = list(np.ones(len(stay_tab_gl)))

# stay_tab_ch = list(np.array(scs.expon.rvs(scale=10, loc=1, size=int(1e4)), dtype=int))
# stay_tab_gl = list(np.array(scs.expon.rvs(scale=10, loc=1, size=int(1e4)), dtype=int))


def fct_choice(y0, z0):
    return (c + y0) ** alpha / ((c + z0) ** alpha + (c + y0) ** alpha)


n = 1000
lg = 60*60

for (alpha, c) in [(1, 1)]:  #, (.5, .1), (.35, .01), (2, 10), (10, 75)]:
    pb.subplots()
    for n_ant in [1]:  #, 100]:
        print((alpha, c), n_ant)

        lg2 = int(2*lg)
        # dxs = np.array(scs.expon.rvs(scale=1, size=lg2*n_ant), dtype=int)

        res = []
        for i in range(n):
            rd_tab = np.random.random(lg2*n_ant)
            time_rd_tab = np.random.randint(0, len(stay_tab_ch)-1, len(rd_tab))
            ii = 0
            print(i)
            act_pos = np.zeros(n_ant, dtype=int)
            z_line = np.zeros((6, 6))
            z_line = np.zeros((2*lg2, 2*lg2))

            jump_time = np.zeros((n_ant, 2))
            jump_time[:, 0] = range(n_ant)
            jump_time[:, 1] = rd.sample(stay_tab_ch, n_ant)
            argsort = jump_time.argsort(axis=0)[:, 1]
            jump_time = jump_time[argsort, :]

            t = 0
            while t < lg:
                # print(t)
                ant = int(jump_time[0, 0])
                dt = float(jump_time[0, 1])

                x = act_pos[ant]
                # dx = dxs[i]
                dx = 1

                if x % 2 == 0:
                    y = z_line[(x-1) % 6, x % 6]
                    z = z_line[x % 6, (x+1) % 6]
                    # y = z_line[x-1+lg2, x+lg2]
                    # z = z_line[x+lg2, x+1+lg2]

                    p = fct_choice(y, z)
                    # p = 0.5

                    if rd_tab[ii] < p:
                        act_pos[ant] -= dx
                        x1, x2 = x-1, x
                    else:
                        act_pos[ant] += dx
                        x1, x2 = x, x+1
                    new_t = stay_tab_ch[time_rd_tab[ii]]
                else:
                    p = 0.5
                    if rd_tab[ii] < p:
                        act_pos[ant] -= dx
                        x1, x2 = x-1, x
                    else:
                        act_pos[ant] += dx
                        x1, x2 = x, x+1
                    new_t = stay_tab_gl[time_rd_tab[ii]]

                z_line[x1 % 6, x2 % 6] += 1
                # z_line[x1+lg2, x2+lg2] += 1

                jump_time[:, 1] -= dt
                jump_time[0, 1] = new_t
                argsort = jump_time.argsort(axis=0)[:, 1]
                jump_time = jump_time[argsort, :]

                t += dt
                ii += 1

            res += list(act_pos)

        y, x = np.histogram(np.abs(res), range(0, 100, 5), normed=True)
        pb.semilogy(x[:-1], y, 'o-', label='n=%i, alpha=%.2f, c=%.2f' % (n_ant, alpha, c))

    pb.legend()
pb.show()
