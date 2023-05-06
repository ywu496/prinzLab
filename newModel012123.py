#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:32:48 2023

@author: wuyuxuan
"""

import pylab as plt
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import signal
import csv
import math
from segTest import segmentSpikeTest
from axonBranch import axBranch

#environmental variables, temperature, salt concentration, 
        # time parameters
dt = 0.01 #numerical integration time step
t_total = 20.0 #total simulation time in ms
t_now = 0.0 #time right now in ms
t = [t_now] #time array in ms

# stimulus parameters (stimulus is current pulse injected in first compartment)
t_stimstart = 5.0 #stimulation start time
t_stimend = 5.5 #stimulation end time
I_stim_amp = 1.0 #stimulation current amplitude in nA, 0.15 is good
I_stim_now = 0.0 #stimulus current in nA
I_stim = [I_stim_now] #stimulus current time course

# temperature
default_temp_C = 22 #default temperature in Celcius. Hochman lab exp mostly at room temperature, 22C. Assumption is that default model parameters are tuned to this temperature.
default_temp_K = default_temp_C + 273.15 #conversion to Kelvin


# ADJUST THIS TO CHANGE TEMPERATURE THROUGHOUT:
temp_C = 39.6 #temperature used in simulation in Celsius


temp_K = temp_C + 273.15 #conversion to Kelvin
Q10 = 3.0 #Q10 for adjusting activation and inactivation rates, typical range for ion channels is 2.4 - 4, see https://link.springer.com/referenceworkentry/10.1007%2F978-1-4614-7320-6_236-1 
taufac = np.power(Q10, (default_temp_K-temp_K)/10) #factor to multiply activation and inactivation time constants to adjust for temperature dependence of gating dynamics

# 'uniform' parameters set here will apply throughout the model, so they don't have to be changed for each branch individually
# 'default' means not adjusted for temperature relative to the McKinnon parameters
E_Na_uniform_default = 60.0 #Na reversal potential in mV, according to McKinnon, default is 60mV
E_Na_uniform = E_Na_uniform_default * temp_K / default_temp_K #adjust according to Nernst equation
E_leak_uniform_default = -55.0 #leak reversal potential in mV, according to McKinnon, default is -55mV
E_leak_uniform = E_leak_uniform_default * temp_K / default_temp_K #adjust according to Nernst equation
E_K_uniform_default = -90.0 #K reversal potential in mV, according to McKinnon, default is -90mV
E_K_uniform = E_K_uniform_default * temp_K / default_temp_K #adjust according to Nernst equation
E_GABA_uniform_default = -60.0 #GABA (i.e., Cl-) reversal potential in mV, see Prescott paper fig. 6, vary from -65mV (control) to -50mV (SCI)
E_GABA_uniform = E_GABA_uniform_default * temp_K / default_temp_K
#####
E_A_uniform_default = -90.0 #default -90
E_A_uniform = E_A_uniform_default* temp_K  /default_temp_K

E_H_uniform_default = -32.0 #default -32
E_H_uniform = E_H_uniform_default * temp_K /default_temp_K

E_M_uniform_default = -90.0 #default -90
E_M_uniform = E_M_uniform_default * temp_K /default_temp_K

E_CaL_uniform_default = 120.0 #default 120
E_CaL_uniform = E_CaL_uniform_default * temp_K /default_temp_K

E_KCa_uniform_default = -90.0 #default -90
E_KCa_uniform = E_KCa_uniform_default * temp_K /default_temp_K



#####
"""
ADD E_OtherChannel_Uniform
"""
G_Na_abs_uniform = 7.0 #Na conductance in nS, based on McKinnon Table 1, default is 300nS
G_Kd_abs_uniform = 100.0 #Kd conductance in nS, based on McKinnon Table 1, default is 2000nS
G_leak_abs_uniform = 1.0 #leak conductance in nS, based on McKinnon Table 1, default is 1nS
G_GABA_uniform = 0.0 #GABA conductance in nS

G_A_abs_uniform  = 0 #default 50 nS, used 1, high impact, hyperpolarize
G_H_abs_uniform  = 0 #default 1nS, used 1, low impact, depolarize

G_M_abs_uniform  = 0 #default 50 nS,used 1, decrease the number spikes, 
G_CaL_abs_uniform  = 1.2 #default 1.2 nS, used 1.2
G_KCa_abs_uniform  = 0 #default 50 nS, used 1

###########
f = 0.01                   # percent of free to bound Ca2+
alpha_CaS = 0.2              # uM/pA; convertion factor from current to concentration 0.002 original
kCaS = 0.024               # /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025

SCa = 1                    # uM; half-saturation of [Ca2+]; 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
tauKCa_0 = 50              # ms
tau_hA_scale = 100          # scaling factor for tau_hA







#line 538 539, related to specific connections
        g_main_b1 = 2.0*g_ax_comp*b1_g_ax_comp/(g_ax_comp+b1_g_ax_comp)
        g_main_b2 = 2.0*g_ax_comp*b2_g_ax_comp/(g_ax_comp+b2_g_ax_comp)
        
        gGABA[1] = g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)
        
        #1078 onwards is just simulation. Needs to automate integration process, instantiate a connnection instance??? what factors into this? 
        #place of connection, index of compartments connected?, compartment axial conductance?, look into this?
        #gaba channels in daughter branches
        self.gGABA[self.n_comp-2] = self.g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)
        
        #branches join together, which involve intersections depicted as axial conductances being made
        #channels specific to intersections need to be manually added with respect to where intersections are and by how much are these channel residing
        #integration process needs autonomous judgment on where to start and when to end, and process of climbing up from daughter branches to mother and then back
        #middle segment, end segment, stimulation segment, intersection segment
        #each of these segments will have a different scheme of integration
        #need to automate





        