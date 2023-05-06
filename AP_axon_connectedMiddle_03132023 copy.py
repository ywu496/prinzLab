#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 03:42:28 2023

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
from integration import Integration





#This part uses segment SpikeTest from segTest.py to track spike conditions in each segment 

def mNaUpdate(Vold, mNaOld):
        alpha = 0.36 * (Vold + 33) / (1 - np.exp(-(Vold + 33) / 3))
        beta = - 0.4 * (Vold + 42) / (1 - np.exp((Vold + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        new_mNa = vinf + (mNaOld- vinf) * np.exp(-dt / tau) if dt < tau else vinf
        return new_mNa
    
def hNaUpdate(Vold, hNaOld):
        alpha = - 0.1 * (Vold + 55) / (1 - np.exp((Vold + 55) / 6))
        beta = 4.5 / (1 + np.exp(-Vold / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        new_hNa = vinf + (hNaOld - vinf) * np.exp(-dt / tau) if dt < tau else vinf
        return new_hNa

def nKdUpdate(Vold, nKdOld):
        alpha = 0.0047 * (Vold - 8) / (1 - np.exp(-(Vold - 8) / 12))
        beta = np.exp(-(Vold + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (Vold + 12) / (1 - np.exp(-(Vold + 12) / 12))
        beta = np.exp(-(Vold + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        new_nKd = vinf + (nKdOld - vinf) * np.exp(-dt / tau) if dt < tau else vinf
        return new_nKd
def mCaLUpdate(Vold, mCaLOld):
        alpha_mCaL = 7.5 / (1 + np.exp((13 - Vold) / 7))
        beta_mCaL = 1.65 / (1 + np.exp((Vold - 14) / 4))
        mCaL_inf = alpha_mCaL / (alpha_mCaL + beta_mCaL)
        tau_mCaL = 1 / (alpha_mCaL + beta_mCaL)
        new_mCaL = mCaL_inf + (mCaLOld - mCaL_inf) * np.exp(-dt / tau_mCaL) if dt < tau_mCaL else mCaL_inf
        return new_mCaL 
def hCaLUpdate(Vold, hCaLOld):
        alpha_hCaL = 0.0068 / (1 + np.exp((Vold + 30) / 12))
        beta_hCaL = 0.06 / (1 + np.exp(-Vold / 11))
        hCaL_inf = alpha_hCaL / (alpha_hCaL + beta_hCaL)
        tau_hCaL = 1 / (alpha_hCaL + beta_hCaL)
        new_hCaL = hCaL_inf + (hCaLOld - hCaL_inf) * np.exp(-dt / tau_hCaL) if dt < tau_hCaL else hCaL_inf
        return new_hCaL
def mMUpdate(Vold, mMOld):
        mM_inf = 1 / (1 + np.exp(-(Vold + 35) / 10))
        tau_mM = 2000 / (3.3 * (np.exp((Vold + 35) / 40) + np.exp(-(Vold + 35) / 20)))
        new_mM = mM_inf + (mMOld - mM_inf) * np.exp(-dt / tau_mM) if dt < tau_mM else mM_inf
        return new_mM

"""
CaS needs input and so does SCa, also no need of Vold
"""
def mKCaUpdate(mKCaOld, CaS_old):
        mKCa_inf = CaS_old ** 2 / (CaS_old ** 2 + SCa ** 2)
        tau_mKCa = tauKCa_0 / (1 + (CaS_old / SCa) ** 2)
        new_mKCa = mKCa_inf + (mKCaOld - mKCa_inf) * np.exp(-dt / tau_mKCa) if dt < tau_mKCa else mKCa_inf
        return new_mKCa

def mAUpdate(Vold, mAOld):
        mA_inf = (0.0761 * np.exp((Vold + 94.22) / 31.84) / (1 + np.exp((Vold + 1.17) / 28.93))) ** (1/3)
        tau_mA = 0.3632 + 1.158 / (1 + np.exp((Vold + 55.96) / 20.12))
        new_mA = mA_inf + (mAOld - mA_inf) * np.exp(-dt / tau_mA) if dt < tau_mA else mA_inf
        return new_mA
def hAUpdate(Vold, hAOld):
        hA_inf = (1 / (1 + np.exp(0.069 * (Vold + 53.3)))) ** 4
        tau_hA = (0.124 + 2.678 / (1 + np.exp((Vold + 50) / 16.027))) * tau_hA_scale
        new_hA = hA_inf + (hAOld - hA_inf) * np.exp(-dt / tau_hA) if dt < tau_hA else hA_inf
        return new_hA

def mHUpdate(Vold, mHOld):
        mh_inf = 1 / (1 + np.exp((Vold + 87.6) / 11.7))
        tau_mh_activ = 53.5 + 67.7 * np.exp(-(Vold + 120) / 22.4)
        tau_mh_deactiv = 40.9 - 0.45 * Vold
        tau_mh = tau_mh_activ if mh_inf > mHOld else tau_mh_deactiv
        new_mH = mh_inf + (mHOld - mh_inf) * np.exp(-dt / tau_mh)
        return new_mH
    
def CaSUpdate(CaSold, ICaL_old, comp_vol):
    #print (ICaL_old)
    #print(comp_vol)
    alpha_CaS = dt * 1e6 / (comp_vol * 2 * 96485) #uM        1mM/1000uM
    #alpha_CaS = 0.2
    #print(alpha_CaS)

    return  CaSold * np.exp(-f * kCaS * dt) - alpha_CaS / kCaS * ICaL_old * (1 - np.exp(-f * kCaS * dt))

#fire function
def fireSlots(start, end, freq, duration):
        stimTime = end - start
        numStim = math.floor(stimTime * freq)
        wholeDura = 1 / freq
        if(wholeDura <= duration):
            print ("Too large of a duration!")
            return
        
        
        cur = start
        result = []
        for i in range(numStim):
            pair = (cur, cur + duration)
            cur = cur + wholeDura
            result.append(pair)
        return result
    
def integCopy(axo):
    axo.V_old = axo.V.copy() #array of previous time step's compartment voltages in mV
    axo.mNa_old = axo.mNa.copy() #array of previous time step's compartment Na activations
    axo.hNa_old = axo.hNa.copy() #array of previous time step's compartment Na inactivations
    axo.nKd_old = axo.nKd.copy() #array of previous time step's compartment Kd activations
    
    axo.mA_old = axo.mA.copy()
    axo.hA_old = axo.hA.copy()
    axo.mH_old = axo.mH.copy()
    axo.mCaL_old = axo.mCaL.copy()
    axo.hCaL_old = axo.hCaL.copy()
    axo.mM_old = axo.mM.copy()
    axo.mKCa_old = axo.mKCa.copy()
    axo.CaS_old = axo.CaS.copy()

def gatingUpdate(axo, i):
    axo.mNa[i] = mNaUpdate(axo.V_old[i], axo.mNa_old[i])
    axo.hNa[i] = hNaUpdate(axo.V_old[i], axo.hNa_old[i])
    axo.gNa[i] = axo.g_mem_Na_comp * np.power(axo.mNa[i], 2) * axo.hNa[i]

    # Kd
    axo.nKd[i] = nKdUpdate(axo.V_old[i], axo.nKd_old[i])
    axo.gKd[i] = axo.g_mem_Kd_comp * np.power(axo.nKd[i], 4)
    # leak
    axo.gleak[i] = axo.g_mem_leak_comp 
    
    axo.mA[i] = mAUpdate(axo.V_old[i], axo.mA_old[i])
    axo.hA[i] = hAUpdate(axo.V_old[i], axo.hA_old[i])
    axo.gA[i] = axo.g_mem_A_comp * np.power(axo.mA[i], 3) * axo.hA[i] 
    
    axo.mH[i] = mHUpdate(axo.V_old[i], axo.mH_old[i])
    axo.gH[i] = axo.g_mem_H_comp * axo.mH[i]
    
    axo.mM[i] = mMUpdate(axo.V_old[i], axo.mM_old[i])
    axo.gM[i] = axo.g_mem_M_comp * np.power(axo.mM[i], 2)
    
    axo.mCaL[i] = mCaLUpdate(axo.V_old[i], axo.mCaL_old[i])
    axo.hCaL[i] = hCaLUpdate(axo.V_old[i], axo.hCaL_old[i])
    axo.gCaL[i] = axo.g_mem_CaL_comp * axo.mCaL[i] * axo.hCaL[i]
    
    axo.mKCa[i] = mKCaUpdate(axo.mKCa_old[i], axo.CaS_old[i])
    axo.gKCa[i] = axo.g_mem_KCa_comp * axo.mKCa[i]
    
    axo.CaS[i] = CaSUpdate(axo.CaS_old[i], axo.gCaL[i] * (axo.V_old[i] - axo.E_CaL), axo.comp_vol) #

#stimulus {index:current, }
def IntegParamUpdate(axo, ACsettingArray,dt, stimDict=None):
    for i in range(len(ACsettingArray)):
        gatingUpdate(axo, i)
        temp_B = (-(axo.gNa[i]+axo.gKd[i]+axo.gleak[i]+axo.gGABA[i]+axo.gA[i] + axo.gH[i] + axo.gM[i] + axo.gCaL[i] + axo.gKCa[i])/axo.C_mem_comp)#in units of uS/nF

        A_condVpairArr, C_condVpairArr = ACsettingArray[i]
        temp_A = 0.0
        temp_d = 0.0

        for aPair in A_condVpairArr:
            conductance, comp_voltage = aPair
            temp_A += conductance / axo.C_mem_comp
            temp_d += conductance / axo.C_mem_comp * comp_voltage
            temp_B += (-conductance/axo.C_mem_comp)
            
            
        temp_C = 0.0
        for cPair in C_condVpairArr:
            conductance, comp_voltage = cPair
            temp_C += conductance / axo.C_mem_comp
            temp_d += conductance / axo.C_mem_comp * comp_voltage
            temp_B += (-conductance/axo.C_mem_comp)
            
        axo.A[i] = (temp_A)
        axo.B[i] = (temp_B)
        axo.C[i] = (temp_C)
        
        if stimDict == None:
            temp_stim = 0.0
        else:
            temp_stim = stimDict.get(i, 0.0)
            
        axo.D[i] = ((axo.gNa[i]*axo.E_Na+axo.gKd[i]*axo.E_K+axo.gleak[i]*axo.E_leak+axo.gGABA[i]*axo.E_GABA + temp_stim
                 +axo.gA[i]*axo.E_A + axo.gH[i]*axo.E_H + axo.gM[i]*axo.E_M + axo.gCaL[i]*axo.E_CaL + axo.gKCa[i]*axo.E_KCa)/axo.C_mem_comp) #in units of nA/nF
        axo.a[i] = (temp_A * dt)
        axo.b[i] = (temp_B * dt)
        axo.c[i] = (temp_C * dt)
    
        axo.d[i] = ((temp_d + axo.D[i] + axo.B[i] * axo.V[i]) * dt)
        if i == 0:
            axo.b_p[0] = axo.b[0] # _p stands for prime as in Dayan and Abbott appendix 6
            axo.d_p[0] = axo.d[0]
        else:
            axo.b_p[i] = axo.b[i] + axo.a[i]*axo.c[i-1]/(1-axo.b_p[i-1]) #equation 6.54 in D&A
            axo.d_p[i] = axo.d[i] + axo.a[i]*axo.d_p[i-1]/(1-axo.b_p[i-1]) #equation 6.55 in D&A 

# Plot I_stim (stimulus current in first compartment) vs time
def plotI_stim(time, stimulus, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, stimulus)
    plt.title(name + ': I_stim vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('I (nA)')
    plt.xlim(0, total_time)
    plt.ylim(-1, 5)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()

# Plot V_rec_first (recorded membrane voltage in first compartment) vs time
def plotVfirst(time, firstVoltage, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, firstVoltage)
    plt.title(name + ': V_first vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('V (mV)')
    plt.xlim(0, total_time)
    #plt.xlim(0.9, 1.2)
    plt.ylim(-100, 100)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()

# Plot V_rec_second (recorded membrane voltage in second compartment) vs time
def plotVsecond(time, secondVoltage, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, secondVoltage)
    plt.title(name + ': V_second vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('V (mV)')
    plt.xlim(0, total_time)
    #plt.xlim(0.9, 1.2)
    plt.ylim(-100, 100)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()

# Plot V_rec_middle (recorded membrane voltage in middle compartment) vs time
def plotVmiddle(time, middleVoltage, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, middleVoltage)
    plt.title(name + ': V_middle vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('V (mV)')
    plt.xlim(0, total_time)
    #plt.xlim(0.9, 1.2)
    plt.ylim(-100, 100)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()

# Plot V_rec_last (recorded membrane voltage in last compartment) vs time
def plotVlast(time, lastVoltage, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, lastVoltage)
    plt.title(name + ': V_last vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('V (mV)')
    plt.xlim(0, total_time)
    #plt.xlim(0.9, 1.2)
    plt.ylim(-100, 100)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()
    
# Plot V_rec_nexttoGABA (recorded membrane voltage in compartment next to GABA) vs time
def plotVnexttoGABA(time, nexttoGABAVoltage, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, nexttoGABAVoltage)
    plt.title(name + ': V_nexttoGABA vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('V (mV)')
    plt.xlim(0, total_time)
    #plt.xlim(0.9, 1.2)
    plt.ylim(-100, 100)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()
### ALL ARRAYS ACROSS COMPARTMENTS WILL USE INDICES 0 THROUGH n_comp-1 
### THIS IS DIFFERENT FROM DAYAN & ABBOTT, which uses 1 THROUGH N

def plotCaS(time, CaS, name, total_time):
    plt.figure(figsize=(12,9))
    plt.plot(time, CaS)
    plt.title(name + ': Ca2+ Concentration vs time')
    plt.xlabel('t (ms)')
    plt.ylabel('CaS (uM)')
    plt.xlim(0, total_time)
    #plt.xlim(0.9, 1.2)
    plt.ylim(0.001, 0.002)
    #axes = plt.gca()
    #axes.yaxis.grid()
    plt.show()
    
        
index2 = 0
b1d_ini = 0.35
b2d_ini = 0.12
twoDarray = []

while index2 < 1:
    oneDarray = []
    index1 = 0
    index2+=1
    while index1 < 1:

        index1 +=1
        
        
    
        # time parameters
        dt = 0.01 #numerical integration time step
        t_total = 60.0 #total simulation time in ms
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
        
        b1 = axBranch(L=200.0, d = b1d_ini + index1 * -0.03, l_comp =10.0 , R_ax = 100.0, V_init = -65.0, \
                     mNa_init = mNa_init, hNa_init = hNa_init, nKd_init = nKd_init, mA_init = mA_init, hA_init = hA_init, mH_init = mH_init, \
                     mM_init = mM_init, mCaL_init = mCaL_init, hCaL_init = hCaL_init, mKCa_init = mKCa_init, CaS_init = CaS_init, \
                     G_leak_abs_uniform = G_leak_abs_uniform, E_leak_uniform = E_leak_uniform, G_Na_abs_uniform = G_Na_abs_uniform, \
                     E_Na_uniform = E_Na_uniform, G_Kd_abs_uniform = G_Kd_abs_uniform, E_K_uniform = E_K_uniform, \
                     G_GABA_uniform = G_GABA_uniform, E_GABA_uniform = E_GABA_uniform, \
                     G_A_abs_uniform = G_A_abs_uniform, E_A_uniform = E_A_uniform, G_H_abs_uniform = G_H_abs_uniform,\
                     E_H_uniform = E_H_uniform, G_M_abs_uniform = G_M_abs_uniform, E_M_uniform = E_M_uniform, \
                     G_CaL_abs_uniform = G_CaL_abs_uniform, E_CaL_uniform = E_CaL_uniform, G_KCa_abs_uniform = G_KCa_abs_uniform, E_KCa_uniform = E_KCa_uniform)
        #Branch b2
        b2 = axBranch(L=200.0, d = b2d_ini + index2 * 0.03, l_comp =10.0 , R_ax = 100.0, V_init = -65.0, \
                     mNa_init = mNa_init, hNa_init = hNa_init, nKd_init =  nKd_init, mA_init = mA_init, hA_init = hA_init, mH_init = mH_init, \
                     mM_init = mM_init, mCaL_init = mCaL_init, hCaL_init = hCaL_init, mKCa_init = mKCa_init, CaS_init = CaS_init, \
                     G_leak_abs_uniform = G_leak_abs_uniform, E_leak_uniform = E_leak_uniform, G_Na_abs_uniform = G_Na_abs_uniform, \
                     E_Na_uniform = E_Na_uniform, G_Kd_abs_uniform = G_Kd_abs_uniform, E_K_uniform = E_K_uniform, \
                     G_GABA_uniform = G_GABA_uniform, E_GABA_uniform = E_GABA_uniform, \
                     G_A_abs_uniform = G_A_abs_uniform, E_A_uniform = E_A_uniform, G_H_abs_uniform = G_H_abs_uniform,\
                     E_H_uniform = E_H_uniform, G_M_abs_uniform = G_M_abs_uniform, E_M_uniform = E_M_uniform, \
                     G_CaL_abs_uniform = G_CaL_abs_uniform, E_CaL_uniform = E_CaL_uniform, G_KCa_abs_uniform = G_KCa_abs_uniform, E_KCa_uniform = E_KCa_uniform)
        
#A(0) -> C(n_comp -1), crosspoint = b1.n_comp // 2 connected to b2.n_comp // 2, stimulate at compartment 0.
#---------
         #-
         #-
#--------
        b1.Cas_monitor_main = [CaS_init] #the same compartment as v_rec_middle
        b2.Cas_monitor_main = [CaS_init]
        b1.connIndex = b1.n_comp // 2
        b2.connIndex = b2.n_comp // 2
        
        b1.First = segmentSpikeTest( -30 )
        b1.Middle = segmentSpikeTest( -30 )
        b1.Last = segmentSpikeTest( -30 )
        b2.First  = segmentSpikeTest( -30 )
        b2.Middle  = segmentSpikeTest( -30 )
        b2.Last = segmentSpikeTest( -30 )

        # coupling between main axon and branches, see Dayan & Abbott page 219, fig 6.16
        g_middle_junction = 2.0*b1.g_ax_comp*b2.g_ax_comp/(b1.g_ax_comp+b2.g_ax_comp)
        
        
        b1.ACArr = [([(0.0,0.0)],[(b1.g_ax_comp, b1.V[1])])]
        

        for i in range(1, b1.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
            if i == b1.connIndex:
                b1.ACArr.append(([(b1.g_ax_comp, b1.V[i-1])], [(b1.g_ax_comp,b1.V[i+1]), (g_middle_junction, b2.V[b2.connIndex])]))
                
            else:
                b1.ACArr.append(([(b1.g_ax_comp, b1.V[i-1])], [(b1.g_ax_comp,b1.V[i+1])]))


        b1.ACArr.append(([(b1.g_ax_comp, b1.V[b1.n_comp-2])], [(0.0,0.0)]))
        
        stimDict1 = {}
        stimDict1[0] = I_stim_now
        Integration.IntegParamInit(b1, b1.ACArr, dt, stimDict1)
        
        b1.V_rec_first = [b1.V_init] #recorded voltage time course in first compartment in mV
        b1.V_rec_second = [b1.V_init] #recorded voltage time course in second compartment in mV
        b1.V_rec_middle = [b1.V_init] #recorded voltage time course in middle compartment in mV
        b1.V_rec_last = [b1.V_init] #recorded voltage time course in last compartment in mV
        b1.V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV
        
        
        
        
        
        
        b2.ACArr = [([(0.0,0.0)],[(b2.g_ax_comp, b2.V[1])])]
        
        
        for i in range(1, b2.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
            if i == b2.connIndex:
                b2.ACArr.append(([(b2.g_ax_comp, b2.V[i-1])], [(b2.g_ax_comp,b2.V[i+1]), (g_middle_junction, b1.V[b1.connIndex])]))
            else:
                b2.ACArr.append(([(b2.g_ax_comp, b2.V[i-1])], [(b2.g_ax_comp,b2.V[i+1])]))
        
        
        
        b2.ACArr.append(([(b2.g_ax_comp, b2.V[b2.n_comp-2])], [(0.0,0.0)]))
        
        Integration.IntegParamInit(b2, b2.ACArr, dt, stimDict = None)
        
        # recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
        b2.V_rec_first = [b2.V_init] #recorded voltage time course in first compartment in mV
        b2.V_rec_second = [b2.V_init] #recorded voltage time course in second compartment in mV
        b2.V_rec_middle = [b2.V_init] #recorded voltage time course in middle compartment in mV
        b2.V_rec_last = [b2.V_init] #recorded voltage time course in last compartment in mV
        b2.V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV

        # BEGIN SIMULATION
        slotList = fireSlots(5, 20, 1/4, 0.5)         #original slotList = fireSlots(5, 28, 1/5, 1)
        while t_now+dt < t_total: 
                t_now += dt
                t.append(t_now)
                
                stimulatedOrNot = False
                for pair in slotList:
                    s, e = pair
                    if(t_now >= s and t_now < e):
                        stimulatedOrNot = True
                        break
                if stimulatedOrNot:
                
               # if t_now>=t_stimstart and t_now<t_stimend:
        
                    I_stim_now=I_stim_amp #apply stimulus current
                else:
                    I_stim_now=0.0;
                
                # store previous time step values in _old arrays
                integCopy(b1)
                integCopy(b2)
                
                b1.ACArr = [([(0.0,0.0)],[(b1.g_ax_comp, b1.V[1])])]
                
                for i in range(1, b1.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
                    if i == b1.connIndex:
                        b1.ACArr.append(([(b1.g_ax_comp, b1.V[i-1])], [(b1.g_ax_comp,b1.V[i+1]), (g_middle_junction, b2.V[b2.connIndex])]))
                    else:
                        b1.ACArr.append(([(b1.g_ax_comp, b1.V[i-1])], [(b1.g_ax_comp,b1.V[i+1])]))
                    
                
                b1.ACArr.append(([(b1.g_ax_comp, b1.V[b1.n_comp-2])], [(0.0,0.0)]))
                stimDict2 = {}
                stimDict2[0] = I_stim_now
                
                IntegParamUpdate(b1, b1.ACArr,dt, stimDict2)
                
                
                
                
                b2.ACArr = [([(0.0,0.0)],[(b2.g_ax_comp, b2.V[1])])]
                for i in range(1, b2.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
                    if i == b2.connIndex:
                        b2.ACArr.append(([(b2.g_ax_comp, b2.V[i-1])], [(b2.g_ax_comp,b2.V[i+1]), (g_middle_junction, b1.V[b1.connIndex])]))
                    else:
                        b2.ACArr.append(([(b2.g_ax_comp, b2.V[i-1])], [(b2.g_ax_comp,b2.V[i+1])]))
                b2.ACArr.append(([(b2.g_ax_comp, b2.V[b2.n_comp-2])], [(0.0,0.0)]))
                IntegParamUpdate(b2, b2.ACArr, dt, stimDict = None)
                
                
                ################

         
                 # branch 1     
                b1.deltaV[b1.n_comp-1] = b1.d_p[b1.n_comp-1]/(1-b1.b_p[b1.n_comp-1]) #equation 6.56 in D&A, update voltage for last compartment
                b1.V[b1.n_comp-1] = b1.V_old[b1.n_comp-1] + b1.deltaV[b1.n_comp-1]
                         
                for i in range(b1.n_comp-1, 0, -1): # step through the middle compartments backward to update voltages
                    b1.deltaV[i-1] = (b1.c[i-1]*b1.deltaV[i]+b1.d_p[i-1])/(1-b1.b_p[i-1])
                    b1.V[i-1] = b1.V_old[i-1] + b1.deltaV[i-1]       
         
                 # branch 2            
                b2.deltaV[b2.n_comp-1] = b2.d_p[b2.n_comp-1]/(1-b2.b_p[b2.n_comp-1]) #equation 6.56 in D&A, update voltage for last compartment
                b2.V[b2.n_comp-1] = b2.V_old[b2.n_comp-1] + b2.deltaV[b2.n_comp-1]
                         
                for i in range(b2.n_comp-1, 0, -1): # step through the middle compartments backward to update voltages
                    b2.deltaV[i-1] = (b2.c[i-1]*b2.deltaV[i]+b2.d_p[i-1])/(1-b2.b_p[i-1])
                    b2.V[i-1] = b2.V_old[i-1] + b2.deltaV[i-1]
             ######################
                
                I_stim.append(I_stim_now) #record stimulus current
                b1.Cas_monitor_main.append(b1.CaS[b1.mid_ind])
                b2.Cas_monitor_main.append(b2.CaS[b2.mid_ind])
                
                
                
                
                b1.V_rec_first.append(b1.V[0]) #record voltage in first compartment
                b1.V_rec_middle.append(b1.V[b1.connIndex]) #record voltage in middle compartment
                b1.V_rec_last.append(b1.V[b1.n_comp-1]) #record voltage in last compartment    
        
                b2.V_rec_first.append(b2.V[0]) #record voltage in first compartment
                b2.V_rec_middle.append(b2.V[b2.connIndex]) #record voltage in middle compartment
                b2.V_rec_last.append(b2.V[b2.n_comp-1]) #record voltage in last compartment

                
        plotI_stim(t, I_stim, 'Main', t_total) #Comment out if you don't want to output the graph

        
        plotVfirst(t, b1.V_rec_first, 'b1', t_total) #Comment out if you don't want to output the graph
        #plotVsecond(t, b1.V_rec_second, 'b1', t_total) #Comment out if you don't want to output the graph
        #plotVnexttoGABA(t, b1.V_rec_nexttoGABA, 'Main', t_total) #Comment out if you don't want to output the graph
        plotVmiddle(t, b1.V_rec_middle, 'b1', t_total) #Comment out if you don't want to output the graph
        plotVlast(t, b1.V_rec_last, 'b1', t_total) #Comment out if you don't want to output the graph
        
        plotVfirst(t, b2.V_rec_first, 'b2', t_total) #Comment out if you don't want to output the graph
          
        #plotVsecond(t, b2.V_rec_second, 'b2', t_total) #Comment out if you don't want to output the graph
        #plotVnexttoGABA(t, b2.V_rec_nexttoGABA, 'Main', t_total) #Comment out if you don't want to output the graph
        plotVmiddle(t, b2.V_rec_middle, 'b2', t_total) #Comment out if you don't want to output the graph
        plotVlast(t, b2.V_rec_last, 'b2', t_total) #Comment out if you don't want to output the graph
        plotCaS(t, b1.Cas_monitor_main, 'b1_Middle', t_total)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

