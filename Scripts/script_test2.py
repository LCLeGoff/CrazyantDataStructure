import scipy.stats as scs
import pylab as pb
import numpy as np

import pylab as pb
import scipy as sc
import scipy.stats as scs
import numpy as np

bins = np.arange(1, 3.1, 0.1)
fig, ax = pb.subplots()
for i, n in enumerate(range(100, 1100, 100)):
    beta_simu = []
    for _ in range(500):
        times = scs.pareto.rvs(b=1, size=int(n))
        y, x = np.histogram(times, range(0, 7200, 10))
        x = (x[1:]+x[:-1])/2.
        mask = np.where(y != 0)[0]
        log_x = np.log(x[mask])
        log_y = np.log(y[mask])
        res = sc.polyfit(log_x, log_y, 1)
        af = res[0]
        if not (np.isnan(af)):
            beta_simu.append(-af)
    h2 = np.histogram(beta_simu, bins)
    ax.plot(h2[1][:-1], h2[0], label=n, c=(i/10., 0, 1-i/10.))
pb.axvline(2, ls='--', c='gray')
pb.legend()
pb.show()


# def simulation(walk_times, stop_times, open_times, close_times):
#     times = []
#     energies = []
#     for i in range(1000):
#         found = 0
#         energy = 60*60
#
#         act_gate_time = np.random.choice(close_times)
#         open_gate = 0
#
#         act_ant_time = np.random.choice(walk_times)
#         walking = 1
#
#         total_time = 0
#         while found == 0 and energy > 0:
#             if walking == 1:
#                 if open_gate == 1:
#                     found = 1
#                 else:
#                     if act_gate_time <= act_ant_time:
#                         found = 1
#                         total_time += act_gate_time
#                         energy -= act_gate_time
#                     else:
#                         total_time += act_ant_time
#                         act_gate_time -= act_ant_time
#                         energy -= act_ant_time
#                         walking = 0
#                         act_ant_time = np.random.choice(stop_times)
#
#             else:
#                 if act_gate_time <= act_ant_time:
#                     total_time += act_gate_time
#                     act_ant_time -= act_gate_time
#                     open_gate = 1-open_gate
#                     if open_gate == 0:
#                         act_gate_time = np.random.choice(close_times)
#                     else:
#                         act_gate_time = np.random.choice(open_times)
#                 else:
#                     total_time += act_ant_time
#                     act_gate_time -= act_ant_time
#                     walking = 1
#                     act_ant_time = np.random.choice(walk_times)
#
#         times.append(total_time)
#         energies.append(max(0, energy))
#     return times, energies
#
#
# lamb_w = 0.2
# lamb_s = 1
# beta = 1
#
# walk_times = scs.expon.rvs(size=1000, scale=1/lamb_w)
# stop_times = scs.pareto.rvs(size=1000, b=beta)
# open_times = [10]
# close_times = [30*60]
# times, energies = simulation(walk_times, stop_times, open_times, close_times)
# h, x = pb.histogram(times)
# pb.plot(x[1:]/60., h, '.-')
# pb.figure()
# h, x = pb.histogram(energies)
# pb.plot(x[1:], h, '.-')
