import scipy.stats as scs
import pylab as pb
import numpy as np


def simulation(walk_times, stop_times, open_times, close_times):
    times = []
    energies = []
    for i in range(1000):
        found = 0
        energy = 60*60

        act_gate_time = np.random.choice(close_times)
        open_gate = 0

        act_ant_time = np.random.choice(walk_times)
        walking = 1

        total_time = 0
        while found == 0 and energy > 0:
            if walking == 1:
                if open_gate == 1:
                    found = 1
                else:
                    if act_gate_time <= act_ant_time:
                        found = 1
                        total_time += act_gate_time
                        energy -= act_gate_time
                    else:
                        total_time += act_ant_time
                        act_gate_time -= act_ant_time
                        energy -= act_ant_time
                        walking = 0
                        act_ant_time = np.random.choice(stop_times)

            else:
                if act_gate_time <= act_ant_time:
                    total_time += act_gate_time
                    act_ant_time -= act_gate_time
                    open_gate = 1-open_gate
                    if open_gate == 0:
                        act_gate_time = np.random.choice(close_times)
                    else:
                        act_gate_time = np.random.choice(open_times)
                else:
                    total_time += act_ant_time
                    act_gate_time -= act_ant_time
                    walking = 1
                    act_ant_time = np.random.choice(walk_times)

        times.append(total_time)
        energies.append(max(0, energy))
    return times, energies


lamb_w = 0.2
lamb_s = 1
beta = 1

walk_times = scs.expon.rvs(size=1000, scale=1/lamb_w)
stop_times = scs.pareto.rvs(size=1000, b=beta)
open_times = [10]
close_times = [30*60]
times, energies = simulation(walk_times, stop_times, open_times, close_times)
h, x = pb.histogram(times)
pb.plot(x[1:]/60., h, '.-')
pb.figure()
h, x = pb.histogram(energies)
pb.plot(x[1:], h, '.-')
