#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:09:40 2023

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
from axonBranchTaper import axBranch

index2 = 0
#gold
#b1d_ini = 0.4
#b2d_ini = 0.13

#new
b1d_ini = 0.4
b2d_ini = 0.01
twoDarray = []

while index2 < 1:
    oneDarray = []
    index1 = 0
    index2+=1
    while index1 < 1:

        index1 +=1
        
        
    
        # time parameters
        dt = 0.01 #numerical integration time step
        t_total = 50.0 #total simulation time in ms
        t_now = 0.0 #time right now in ms
        t = [t_now] #time array in ms
        
        # stimulus parameters (stimulus is current pulse injected in first compartment)
        t_stimstart = 5.0 #stimulation start time
        t_stimend = 5.5 #stimulation end time
        I_stim_amp = 0.15 #stimulation current amplitude in nA, 0.15 is good
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
        
        G_gap_junction = 0.02#nS 
        #sparse firing
        #0,06 too large
        #0.02 middle
        #0.002 weak
        #0, no connection
        
        #0.04 too large
        #0.02 middle
        #0.002 weak
        #0, no connection
        
        
        ###########
        f = 0.01                   # percent of free to bound Ca2+
        alpha_CaS = 0.2              # uM/pA; convertion factor from current to concentration 0.002 original
        kCaS = 0.024               # /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025
        
        SCa = 1                    # uM; half-saturation of [Ca2+]; 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
        tauKCa_0 = 50              # ms
        tau_hA_scale = 100          # scaling factor for tau_hA
        
        """
        ADD G_OtherChannel_Uniform
        """
        # model geometry settings
        # main axon
        L = 200.0 #length of axon in um, distance between sympathetic ganglia is about 1mm
        d = 0.5 #diameter of axon in um
        l_comp = 10.0 #length of each compartment in um
        n_comp = int(L/l_comp) #number of compartments
        
        

        
        
        comp_vol = np.pi * ((d/2) ** 2) * l_comp #um^3


        

        
        # geometry-related
        mid_ind = int(n_comp/2) #index of compartment roughly in middle of cable
        A_mem_comp = np.pi*d*l_comp*1e-8 #membrane surface area of compartment in square cm
        A_cross_comp = np.pi*d*d*1e-8/4 #axon cross-sectional in square cm

        
        # capacitance related
        C_mem_comp = A_mem_comp*1e3 #membrane capacitace of individual compartment in nF, assuming 1uF/cm2
        conductance_scaling_factor = 1e6*C_mem_comp/100 #factor used to scale conductances because McKinnon et al model has 100pF capacitance

        # membrane conductances and reversals
        # main branch
        G_leak_abs = G_leak_abs_uniform #leak conductance in nS, based on McKinnon Table 1, default is 1nS
        g_mem_leak_comp = conductance_scaling_factor*G_leak_abs/1e3 #membrane leak conductance per compartment in uS
        E_leak = E_leak_uniform #leak reversal potential in mV, according to McKinnon, default is -55mV
        # Na:
        G_Na_abs = G_Na_abs_uniform #Na conductance in nS, based on McKinnon Table 1, default is 300nS
        g_mem_Na_comp = conductance_scaling_factor*G_Na_abs/1e3 #membrane Na conductance per compartment in uS
        E_Na = E_Na_uniform #Na reversal potential in mV, according to McKinnon, default is 60mV
        # Kd:
        G_Kd_abs = G_Kd_abs_uniform #Kd conductance in nS, based on McKinnon Table 1, default is 2000nS
        g_mem_Kd_comp = conductance_scaling_factor*G_Kd_abs/1e3 #membrane Kd conductance per compartment in uS
        E_K = E_K_uniform #K reversal potential in mV, according to McKinnon, default is -90mV
        # GABA conductance for compartments proximal to (but not at) branch point
        G_GABA_abs = G_GABA_uniform #GABA conductance in nS
        g_mem_GABA = conductance_scaling_factor*G_GABA_abs/1e3 #GABA conductance per compartment in uS
        E_GABA = E_GABA_uniform #GABA (i.e., Cl-) reversal potential in mV, see Prescott paper fig. 6, vary from -65mV (control) to -50mV (SCI)
        
        
        
        G_A_abs = G_A_abs_uniform #default 50 nS
        g_mem_A_comp = conductance_scaling_factor*G_A_abs/1e3
        E_A = E_A_uniform#  default of -90.0 in mV
        
        
        G_H_abs = G_H_abs_uniform #default 1nS
        g_mem_H_comp = conductance_scaling_factor*G_H_abs/1e3
        E_H = E_H_uniform   #  default of -32.0 in mV
        
        
        G_M_abs = G_M_abs_uniform #default 50 nS
        g_mem_M_comp = conductance_scaling_factor*G_M_abs/1e3
        E_M = E_M_uniform#  default of -90.0 in mV
        
        G_CaL_abs = G_CaL_abs_uniform #default 1.2 nS
        g_mem_CaL_comp = conductance_scaling_factor*G_CaL_abs/1e3
        E_CaL = E_CaL_uniform# default of 120.0 in mV
        ###########
        
        G_KCa_abs = G_KCa_abs_uniform #default 50 nS
        g_mem_KCa_comp = conductance_scaling_factor*G_KCa_abs/1e3
        E_KCa = E_KCa_uniform# at default already in mV
        
        
        
        
        
        

        
        
        """
        ADD OtherChannel for both main and other two branches
        """
        
        # axial conductance related
        # main axon
        R_ax = 100.0 #axial resistivity in Ohm cm, from https://www.frontiersin.org/articles/10.3389/fncel.2019.00413/full
        g_ax_comp = A_cross_comp*1e6/(R_ax*l_comp*1e-4) #axial conductance between compartments in uS
        #g_ax_comp = 0.0 #uncouple compartments, for testing



        
        # compartmental voltage changes
        deltaV = [0.0]
        for i in range(1, n_comp, 1):
            deltaV.append(0.0)
        deltaV = np.asarray(deltaV)
        
        # initial values for compartmental voltages, gating variables, conductances, and currents
        V_init = -65.0 #initialize all voltages
        mNa_init = 0.0 #initialize all Na channels to deactivated
        hNa_init = 1.0 #initialize all Na channels to deinactivated
        nKd_init = 0.0 #initialize all Kd channels to deactivated
        
        mA_init = 0.0
        hA_init = 1.0
        mH_init = 0.0
        mM_init = 0.0
        mCaL_init = 0.0
        hCaL_init = 1.0
        mKCa_init = 0.0
        CaS_init = 0.001
        
        
        
        gNa_init = g_mem_Na_comp * np.power(mNa_init, 2) * hNa_init
        gKd_init = g_mem_Kd_comp * np.power(nKd_init, 4)
        gleak_init = g_mem_leak_comp
        
        
        gA_init = g_mem_A_comp * np.power(mA_init, 3) * hA_init 
        gH_init = g_mem_H_comp * mH_init
        gM_init = g_mem_M_comp * np.power(mM_init, 2)
        gCaL_init = g_mem_CaL_comp * mCaL_init * hCaL_init
        gKCa_init = g_mem_KCa_comp * mKCa_init 
        
        
        INa_init = gNa_init * (V_init - E_Na) #initialize Na currents
        IKd_init = gKd_init * (V_init - E_K) #initialize Kd currents
        Ileak_init = gleak_init * (V_init - E_leak) #initialize leak currents
        
        IA_init = gA_init * (V_init - E_A)
        IH_init = gH_init * (V_init - E_H)
        IM_init = gM_init * (V_init - E_M)
        ICaL_init = gCaL_init * (V_init - E_CaL)
        IKCa_init = gKCa_init * (V_init - E_KCa)
        
        b1d_start = b1d_ini + index1 * -0.03
        b2d_start = b2d_ini + index2 * 0.03
        b1_L = 200.0
        b1_l_comp = 10.0
        b2_L = 200.0
        b2_l_comp = 10.0
        b1_endStartratio = 1
        b2_endStartratio = 1
        b1_n_comp = int(b1_L / b1_l_comp)
        b2_n_comp = int(b2_L / b2_l_comp)
        
        b1d_all = np.linspace(b1d_start, b1d_start * b1_endStartratio, b1_n_comp)
        b2d_all = np.linspace(b2d_start, b1d_start * b2_endStartratio, b2_n_comp)

        
        b1 = axBranch(L=b1_L, d = b1d_all, l_comp =b1_l_comp , R_ax = 100.0, V_init = -65.0, \
                     mNa_init = mNa_init, hNa_init = hNa_init, nKd_init = nKd_init, mA_init = mA_init, hA_init = hA_init, mH_init = mH_init, \
                     mM_init = mM_init, mCaL_init = mCaL_init, hCaL_init = hCaL_init, mKCa_init = mKCa_init, CaS_init = CaS_init, \
                     G_leak_abs_uniform = G_leak_abs_uniform, E_leak_uniform = E_leak_uniform, G_Na_abs_uniform = G_Na_abs_uniform, \
                     E_Na_uniform = E_Na_uniform, G_Kd_abs_uniform = G_Kd_abs_uniform, E_K_uniform = E_K_uniform, \
                     G_GABA_uniform = G_GABA_uniform, E_GABA_uniform = E_GABA_uniform, \
                     G_A_abs_uniform = G_A_abs_uniform, E_A_uniform = E_A_uniform, G_H_abs_uniform = G_H_abs_uniform,\
                     E_H_uniform = E_H_uniform, G_M_abs_uniform = G_M_abs_uniform, E_M_uniform = E_M_uniform, \
                     G_CaL_abs_uniform = G_CaL_abs_uniform, E_CaL_uniform = E_CaL_uniform, G_KCa_abs_uniform = G_KCa_abs_uniform, E_KCa_uniform = E_KCa_uniform)
        #Branch b2
            
            
            
            
            
            
            
            
            