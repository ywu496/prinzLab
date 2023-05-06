# 05/15/17
# Author: Kun Tian (io.kuntian@gmail.com)
# Python 2.7.10

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import ode_solver
import tspn

# constants
cc_amplitude = -120            # pA; current step amplitude; could be any of [-90, 50, 80, 100]
start_cc = 5000                # ms
end_cc = 8000
step_size = 0.25               # ms

# initialize
# primary_fr = 0                              # firing rate (vector) of the primary synapse
# num_secondary = 0                           # number of secondary synapses
# num_syn = num_secondary + 1                 # total # of synapses
# tau_rise = 1                                # rise time constant for the nicotinic synaptic conductance
# tau_fall = 5                                # fall time constant for the nicotinic synaptic conductance
# gsyn_scaling = 0.534985                     # nicotinic syanptic conductance pulse scaling constant
# gsyn_threshold = 10.68                      # threshold nicotinic synaptic conductance
# gsyn_max = [0]                              # nS; 0 is the primary synapse

t_total = end_cc + 1200                       # ms; total simulation time
t_len = int(t_total / step_size)
t_template = np.arange(0, t_total, step_size)

y0 = []
y0.append(-65)                  # initial membrane potential; mV
y0.append(0.001)                # intracellular [Ca2+]; mM
y0.append(0.0000422117)         # m
y0.append(0.9917)               # h
y0.append(0.00264776)           # n
y0.append(0.5873)               # mA
y0.append(0.1269)               # hA
y0.append(0.0517)               # mh
y0.append(0.000025)             # mM
y0.append(7.6e-5)               # mCaL
y0.append(0.94)                 # hCaL
y0.append(0.4)                  # s
y0.append(0.000025)             # mKCa
y0.append(0)                    # INa
y0.append(0)                    # IK
y0.append(0)                    # ICaL
y0.append(0)                    # IM
y0.append(0)                    # IKCa
y0.append(0)                    # IA
y0.append(0)                    # Ih
y0.append(0)                    # Ileak
y0.append(0)                    # mh_inf

# G = [0, 400, 300, 1, 14, 10, 5, 2, 1, 100,]  # idx, Na, K, CaL, M, KCa, A, h, leak, C
G = [0, 400, 300, 5, 10, 10, 1, 0.4, 0.5, 100,]  # idx, Na, K, CaL, M, KCa, A, h, leak, C

# compute current clamp (cc) template
cc_template = np.zeros((t_len, 1))

for i in range(0, t_len):
    if int(start_cc / step_size) <= i <= int(end_cc / step_size):
        cc_template[i] = cc_amplitude
    else:
        cc_template[i] = -20

# update dynamic variables
y = ode_solver.update(tspn.step, y0, t_len, cc_template, G, step_size)
sio.savemat('y.mat', {'output':y})
np.set_printoptions(threshold=np.nan)

# calculate ISI
spkCount = 0
spkIdx = np.zeros(100, dtype=np.int)

slope = np.sign(np.diff(y[:,0]))
for i in range(20001, 32000):
    if slope[i-1] == 1 and slope[i] == -1 and y[i,0] > -20:
        spkCount += 1
        spkIdx[spkCount] = i

spkIdx = spkIdx[spkIdx > 0]
isi = np.diff(t_template[spkIdx]) / 1e3
isi_inv = [1 / ind for ind in isi]

print("# of ISI is %d" % len(isi))

## plot
# plot setting
start_idx = 4950
end_idx = 8950
start = int(start_idx / step_size)
end = int(end_idx / step_size)

print("RMP is %s" % y[19200,0])
print("Vmin during cc is %s" % np.min(y[20000:28000,0]))
print("V at the end of cc is %s" % y[27999,0])

# plot: cc clamp
plt.figure(figsize=(9,6))
plt.subplot(311)
plt.plot(t_template[start:end] / 1000, y[start:end,0], 'b')
plt.ylabel('V (mV)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.ylim(-150, 50)
plt.subplot(312)
plt.plot(t_template[start:end] / 1000, y[start:end,1], 'g')
plt.ylabel('CaS (uM)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
# plt.ylim(plt.ylim()[0], 5)
plt.subplot(313)
plt.plot(t_template[start:end] / 1000, cc_template[start:end], 'r')
plt.xlabel('time (s)', fontsize=18)
plt.ylabel('I (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.ylim(-110, 110)
plt.savefig('IV.jpg', dpi=500)
plt.close()

# plot: kinetics
plt.figure(figsize=(16,9))
plt.subplot(431)
plt.plot(t_template[start:end] / 1000, y[start:end,2])
plt.ylabel('m', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(432)
plt.plot(t_template[start:end] / 1000, y[start:end,3])
plt.ylabel('h', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(433)
plt.plot(t_template[start:end] / 1000, y[start:end,4])
plt.ylabel('n', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(434)
plt.plot(t_template[start:end] / 1000, y[start:end,5])
plt.ylabel('mA', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(435)
plt.plot(t_template[start:end] / 1000, y[start:end,6])
plt.ylabel('hA', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(436)
plt.plot(t_template[start:end] / 1000, y[start:end,7])
plt.ylabel('mh', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(437)
plt.plot(t_template[start:end] / 1000, y[start:end,8])
plt.ylabel('mM', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(438)
plt.plot(t_template[start:end] / 1000, y[start:end,9])
plt.ylabel('mCaL', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(439)
plt.plot(t_template[start:end] / 1000, y[start:end,10])
plt.ylabel('hCaL', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(4,3,10)
plt.plot(t_template[start:end] / 1000, y[start:end,11])
plt.ylabel('s', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(4,3,11)
plt.plot(t_template[start:end] / 1000, y[start:end,11])
plt.ylabel('mKCa', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.savefig('Kinetics.jpg', dpi=500)
plt.close()

# plot: I
plt.figure(figsize=(16,9))
plt.subplot(421)
plt.plot(t_template[start:end] / 1000, y[start:end,13])
plt.ylabel('INa (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(422)
plt.plot(t_template[start:end] / 1000, y[start:end,14])
plt.ylabel('IK (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(423)
plt.plot(t_template[start:end] / 1000, y[start:end,15])
plt.ylabel('ICaL (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(424)
plt.plot(t_template[start:end] / 1000, y[start:end,16])
plt.ylabel('IM (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(425)
plt.plot(t_template[start:end] / 1000, y[start:end,17])
plt.ylabel('IKCa (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(426)
plt.plot(t_template[start:end] / 1000, y[start:end,18])
plt.ylabel('IA (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(427)
plt.plot(t_template[start:end] / 1000, y[start:end,19])
plt.ylabel('Ih (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(428)
plt.plot(t_template[start:end] / 1000, y[start:end,0], 'r')
plt.ylabel('V (mV)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.ylim(-100, 100)
plt.savefig('I.jpg', dpi=500)
plt.close()

# plot: Instant FR
plt.figure(figsize=(16,9))
plt.subplot(211)
plt.plot(isi_inv)
plt.ylabel('instant FR', fontsize=18)
plt.ylim(0, 25)
plt.subplot(212)
plt.plot(t_template[start:end] / 1000, y[start:end,0], 'r')
plt.ylabel('V (mV)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.ylim(-80, 80)
plt.savefig('Instant_FR.jpg', dpi=500)
plt.close()

# plot: delay
plt.figure(figsize=(6,9))
plt.subplot(311)
plt.plot(t_template[start:end] / 1000, y[start:end,0], 'k')
plt.ylabel('V (mV)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.ylim(-100, 50)
plt.subplot(312)
gA_dynamics = (y[start:end,5] **3) * y[start:end,6]
plt.plot(t_template[start:end] / 1000, gA_dynamics, 'k')
plt.ylabel('IA kinetics', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.subplot(313)
plt.plot(t_template[start:end] / 1000, y[start:end,18], 'k')
plt.ylabel('IA (pA)', fontsize=18)
plt.xlim(start_idx / 1000, end_idx / 1000)
plt.savefig('Delay.jpg', dpi=500)
plt.close()