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

def IntegParamUpdate(axo, ACsettingArray, dt):
    axo.A = []
    axo.B = []
    axo.C = []
    axo.D = []
    axo.a = []
    axo.b = []
    axo.c = []
    axo.d = []
    
    for i in range(len(ACsettingArray)):
        ###attention
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
            
        axo.A.append(temp_A)
        axo.B.append(temp_B)
        axo.C.append(temp_C)
        axo.D.append((axo.gNa[i]*axo.E_Na+axo.gKd[i]*axo.E_K+axo.gleak[i]*axo.E_leak+axo.gGABA[i]*axo.E_GABA
                 +axo.gA[i]*axo.E_A + axo.gH[i]*axo.E_H + axo.gM[i]*axo.E_M + axo.gCaL[i]*axo.E_CaL + axo.gKCa[i]*axo.E_KCa)/axo.C_mem_comp) #in units of nA/nF

        axo.a.append(temp_A * dt)
        axo.b.append(temp_B * dt)
        axo.c.append(temp_C * dt)
    
        axo.d.append((temp_d + axo.D[i] + axo.B[i] * axo.V[i]) * dt)
        if i == 0:
            axo.b_p[0] = axo.b[0] # _p stands for prime as in Dayan and Abbott appendix 6
            axo.d_p[0] = axo.d[0]
        else:
            axo.b_p[i] = axo.b[i] + axo.a[i]*axo.c[i-1]/(1-axo.b_p[i-1]) #equation 6.54 in D&A
            axo.d_p[i] = axo.d[i] + axo.a[i]*axo.d_p[i-1]/(1-axo.b_p[i-1]) #equation 6.55 in D&A 

def IntegParamUpdate(axo, ACsettingArray, dt):

    
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
        axo.D[i] = ((axo.gNa[i]*axo.E_Na+axo.gKd[i]*axo.E_K+axo.gleak[i]*axo.E_leak+axo.gGABA[i]*axo.E_GABA
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
        
        
        #generate and fill in arrays of compartmental voltages, gating variables, and currents
        V = [V_init] #array of compartment voltages in mV
        mNa = [mNa_init] #array of Na activation variables
        hNa = [hNa_init] #array of Na inactivation variables
        nKd = [nKd_init] #array of Kd activation variables
        
        mA = [mA_init]
        hA = [hA_init]
        mH = [mH_init]
        mM = [mM_init]
        mCaL = [mCaL_init]
        hCaL = [hCaL_init]
        mKCa= [mKCa_init]
        CaS = [CaS_init]
        
        gNa = [gNa_init] #array of Na conductances
        gKd = [gKd_init] #array of Kd conductances
        gleak = [gleak_init] #array of leak conductances
        gGABA = [0.0] #array of GABA conductances, will be zeros except for compartments proximal to branch point
        
        gA = [gA_init]
        gH = [gH_init]
        gM = [gM_init]
        gCaL = [gCaL_init]
        gKCa = [gKCa_init]
        
        
        
        for i in range(1, n_comp):
            V.append(V_init) #initialize compartment voltage array
            mNa.append(mNa_init) #initialize compartment Na activation array
            hNa.append(hNa_init) #initialize compartment Na inactivation array
            nKd.append(nKd_init) #initialize compartment Kd activation array
            
            mA.append(mA_init) #initialize compartment Kd activation array
            hA.append(hA_init) #initialize compartment Kd activation array
            mH.append(mH_init)
            mCaL.append(mCaL_init)
            hCaL.append(hCaL_init)
            mM.append(mM_init)
            mKCa.append(mKCa_init)
            CaS.append(CaS_init)
        
            
            gNa.append(gNa_init)
            gKd.append(gKd_init)
            gleak.append(gleak_init)
            gGABA.append(0.0)
        
        
            gA.append(gA_init)
            gH.append(gH_init)
            gM.append(gM_init)
            gCaL.append(gCaL_init)
            gKCa.append(gKCa_init)
            

            
        gGABA[1] = g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)

        #Branch b1
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
        b3 = axBranch(L=200.0, d = b1d_ini + index1 * -0.03, l_comp =10.0 , R_ax = 100.0, V_init = -65.0, \
                     mNa_init = mNa_init, hNa_init = hNa_init, nKd_init = nKd_init, mA_init = mA_init, hA_init = hA_init, mH_init = mH_init, \
                     mM_init = mM_init, mCaL_init = mCaL_init, hCaL_init = hCaL_init, mKCa_init = mKCa_init, CaS_init = CaS_init, \
                     G_leak_abs_uniform = G_leak_abs_uniform, E_leak_uniform = E_leak_uniform, G_Na_abs_uniform = G_Na_abs_uniform, \
                     E_Na_uniform = E_Na_uniform, G_Kd_abs_uniform = G_Kd_abs_uniform, E_K_uniform = E_K_uniform, \
                     G_GABA_uniform = G_GABA_uniform, E_GABA_uniform = E_GABA_uniform, \
                     G_A_abs_uniform = G_A_abs_uniform, E_A_uniform = E_A_uniform, G_H_abs_uniform = G_H_abs_uniform,\
                     E_H_uniform = E_H_uniform, G_M_abs_uniform = G_M_abs_uniform, E_M_uniform = E_M_uniform, \
                     G_CaL_abs_uniform = G_CaL_abs_uniform, E_CaL_uniform = E_CaL_uniform, G_KCa_abs_uniform = G_KCa_abs_uniform, E_KCa_uniform = E_KCa_uniform)
        

            
        main_First = segmentSpikeTest( -30 )
        main_Middle = segmentSpikeTest( -30 ) 
        main_Last = segmentSpikeTest( -30 )
        b1.First = segmentSpikeTest( -30 )
        b1.Middle = segmentSpikeTest( -30 )
        b1.Last = segmentSpikeTest( -30 )
        b2.First  = segmentSpikeTest( -30 )
        b2.Middle  = segmentSpikeTest( -30 )
        b2.Last = segmentSpikeTest( -30 )

        # coupling between main axon and branches, see Dayan & Abbott page 219, fig 6.16
        g_main_b1 = 2.0*g_ax_comp*b1.g_ax_comp/(g_ax_comp+b1.g_ax_comp)
        g_main_b2 = 2.0*g_ax_comp*b2.g_ax_comp/(g_ax_comp+b2.g_ax_comp)
        
        # initialize compartmental integration parameters
        A = [(g_main_b1+g_main_b2)/C_mem_comp] #initialize branch compartment, in units of uS/nF
        #####
        B = [-(gNa[0]+gKd[0]+gleak[0]+gGABA[0] + gA[0] + gH[0] + gM[0] + gCaL[0] + gKCa[0]
               +g_ax_comp+g_main_b1+g_main_b2)/C_mem_comp] #in units of uS/nF
        #####
        C = [g_ax_comp/C_mem_comp] #in units of uS/nF
        #####
        D = [(gNa[0]*E_Na+gKd[0]*E_K+gleak[0]*E_leak+gGABA[0]*E_GABA
              +gA[0]*E_Na + gH[0]*E_H + gM[0]*E_Na + gCaL[0]*E_CaL + gKCa[0]*E_KCa)/C_mem_comp] #in units of nA/nF
        ######
        a = [A[0]*dt] #in units of uS*ms/nF
        b = [B[0]*dt] #in units of uS*ms/nF
        c = [C[0]*dt] #in units of uS*ms/nF
        d = [(D[0]+(g_main_b1*b1.V[b1.n_comp-1]+g_main_b2*b2.V[b2.n_comp-1])/C_mem_comp+B[0]*V[0]+C[0]*V[1])*dt] #in units of nA*ms/nF
        
        for i in range(1, n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
            A.append(g_ax_comp/C_mem_comp) #in units of uS/nF
            B.append(-(gNa[i]+gKd[i]+gleak[i]+gGABA[i]+gA[i] + gH[i] + gM[i] + gCaL[i] + gKCa[i]+2.0*g_ax_comp)/C_mem_comp) #in units of uS/nF
            C.append(g_ax_comp/C_mem_comp) #in units of uS/nF
            D.append((gNa[i]*E_Na+gKd[i]*E_K+gleak[i]*E_leak+gGABA[i]*E_GABA + 
                      gA[i]*E_A + gH[i]*E_H + gM[i]*E_M + gCaL[i]*E_CaL + gKCa[i]*E_KCa)/C_mem_comp) #in units of nA/nF
            a.append(A[i]*dt)    
            b.append(B[i]*dt) 
            c.append(C[i]*dt)
            d.append((D[i]+A[i]*V[i-1]+B[i]*V[i]+C[i]*V[i+1])*dt)
        
        A.append(g_ax_comp/C_mem_comp) #in units of uS/nF, for last compartment
        B.append(-(gNa[n_comp-1]+gKd[n_comp-1]+gleak[n_comp-1]+gGABA[n_comp-1] +
                   gA[n_comp-1] + gH[n_comp-1] + gM[n_comp-1] + gCaL[n_comp-1] + gKCa[n_comp-1]+g_ax_comp)/C_mem_comp) #in units of uS/nF
        C.append(0.0) #in units of uS/nF
        D.append((gNa[n_comp-1]*E_Na+gKd[n_comp-1]*E_K+gleak[n_comp-1]*E_leak+gGABA[n_comp-1]*E_GABA+I_stim_now
                  +gA[n_comp-1]*E_A + gH[n_comp-1]*E_H + gM[n_comp-1]*E_M + gCaL[n_comp-1]*E_CaL + gKCa[n_comp-1]*E_KCa)/C_mem_comp) #in units of nA/nF
        a.append(A[n_comp-1]*dt)    
        b.append(B[n_comp-1]*dt) 
        c.append(C[n_comp-1]*dt)
        d.append((D[n_comp-1]+A[n_comp-1]*V[n_comp-2]+B[n_comp-1]*V[n_comp-1])*dt)
        
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)
        
        b_p = b.copy() 
        d_p = d.copy()
        Cas_monitor_main = [CaS_init] #the same compartment as v_rec_middle
        # recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
        V_rec_first = [V_init] #recorded voltage time course in first compartment in mV
        V_rec_second = [V_init] #recorded voltage time course in second compartment in mV
        V_rec_middle = [V_init] #recorded voltage time course in middle compartment in mV
        V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV
        V_rec_last = [V_init] #recorded voltage time course in last compartment in mV

        
        b1.ACArr = [([(0.0,0.0)],[(b1.g_ax_comp, b1.V[1])])]
        
        


        for i in range(1, b1.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
            b1.ACArr.append(([(b1.g_ax_comp, b1.V[i-1])], [(b1.g_ax_comp,b1.V[i+1])]))

            

        
        b1.ACArr.append(([(b1.g_ax_comp, b1.V[b1.n_comp-2])], [(g_main_b1,V[0])]))
        

        
        Integration.IntegParamInit(b1, b1.ACArr, dt)

        
        # recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
        b1.V_rec_first = [b1.V_init] #recorded voltage time course in first compartment in mV
        b1.V_rec_second = [b1.V_init] #recorded voltage time course in second compartment in mV
        b1.V_rec_middle = [b1.V_init] #recorded voltage time course in middle compartment in mV
        b1.V_rec_last = [b1.V_init] #recorded voltage time course in last compartment in mV
        b1.V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV
        
        b2.ACArr = [([(0.0,0.0)],[(b2.g_ax_comp, b2.V[1])])]


        
        for i in range(1, b2.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
            b2.ACArr.append(([(b2.g_ax_comp, b2.V[i-1])], [(b2.g_ax_comp,b2.V[i+1])]))

        
        b2.ACArr.append(([(b2.g_ax_comp, b2.V[b2.n_comp-2])], [(g_main_b2,V[0])]))


        Integration.IntegParamInit(b2, b2.ACArr, dt)

        # recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
        b2.V_rec_first = [b2.V_init] #recorded voltage time course in first compartment in mV
        b2.V_rec_second = [b2.V_init] #recorded voltage time course in second compartment in mV
        b2.V_rec_middle = [b2.V_init] #recorded voltage time course in middle compartment in mV
        b2.V_rec_last = [b2.V_init] #recorded voltage time course in last compartment in mV
        b2.V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV
        

        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        
        
            
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
                V_old = V.copy() #array of previous time step's compartment voltages in mV
                mNa_old = mNa.copy() #array of previous time step's compartment Na activations
                hNa_old = hNa.copy() #array of previous time step's compartment Na inactivations
                nKd_old = nKd.copy() #array of previous time step's compartment Kd activations
                mA_old = mA.copy()
                hA_old = hA.copy()
                mH_old = mH.copy()
                mCaL_old = mCaL.copy()
                hCaL_old = hCaL.copy()
                mM_old = mM.copy()
                mKCa_old = mKCa.copy()
                CaS_old = CaS.copy()

                
        
        
                
                b1.V_old = b1.V.copy() #array of previous time step's compartment voltages in mV
                b1.mNa_old = b1.mNa.copy() #array of previous time step's compartment Na activations
                b1.hNa_old = b1.hNa.copy() #array of previous time step's compartment Na inactivations
                b1.nKd_old = b1.nKd.copy() #array of previous time step's compartment Kd activations
                
                b1.mA_old = b1.mA.copy()
                b1.hA_old = b1.hA.copy()
                b1.mH_old = b1.mH.copy()
                b1.mCaL_old = b1.mCaL.copy()
                b1.hCaL_old = b1.hCaL.copy()
                b1.mM_old = b1.mM.copy()
                b1.mKCa_old = b1.mKCa.copy()
                b1.CaS_old = b1.CaS.copy()
                
        
                b2.V_old = b2.V.copy() #array of previous time step's compartment voltages in mV
                b2.mNa_old = b2.mNa.copy() #array of previous time step's compartment Na activations
                b2.hNa_old = b2.hNa.copy() #array of previous time step's compartment Na inactivations
                b2.nKd_old = b2.nKd.copy() #array of previous time step's compartment Kd activations
                
                b2.mA_old = b2.mA.copy()
                b2.hA_old = b2.hA.copy()
                b2.mH_old = b2.mH.copy()
                b2.mCaL_old = b2.mCaL.copy()
                b2.hCaL_old = b2.hCaL.copy()
                b2.mM_old = b2.mM.copy()
                b2.mKCa_old = b2.mKCa.copy()
                b2.CaS_old = b2.CaS.copy()
                
                
        
                # integratinng branch 1 dynamic variables
                # first compartment
                # Na
                b1.ACArr = [([(0.0,0.0)],[(b1.g_ax_comp, b1.V[1])])]

                for i in range(1, b1.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
                    b1.ACArr.append(([(b1.g_ax_comp, b1.V[i-1])], [(b1.g_ax_comp,b1.V[i+1])]))
                
                b1.ACArr.append(([(b1.g_ax_comp, b1.V[b1.n_comp-2])], [(g_main_b1,V[0])]))
                
                IntegParamUpdate(b1, b1.ACArr, dt)
                
                
                b2.ACArr = [([(0.0,0.0)],[(b2.g_ax_comp, b2.V[1])])]


                
                for i in range(1, b2.n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
                    b2.ACArr.append(([(b2.g_ax_comp, b2.V[i-1])], [(b2.g_ax_comp,b2.V[i+1])]))

                
                b2.ACArr.append(([(b2.g_ax_comp, b2.V[b2.n_comp-2])], [(g_main_b2,V[0])]))
                IntegParamUpdate(b2, b2.ACArr, dt)

        
                ###last branch compartments get axial current from main axon compartment 0
        
                # main axon
                # compartment 0 gets axial current from branch compartments n-1
                # Na
               
                 
        
                gNa[0] = g_mem_Na_comp * np.power(mNa[0], 2) * hNa[0]
                mNa[0] = mNaUpdate(V_old[0], mNa_old[0])
                hNa[0] = hNaUpdate(V_old[0], hNa_old[0])
                gNa[0] = g_mem_Na_comp * np.power(mNa[0], 2) * hNa[0]
        
                # Kd
                nKd[0] = nKdUpdate(V_old[0], nKd_old[0])
                gKd[0] = g_mem_Kd_comp * np.power(nKd[0], 4)
                # leak
                gleak[0] = g_mem_leak_comp
                
                
                
                mA[0] = mAUpdate(V_old[0], mA_old[0])
                hA[0] = hAUpdate(V_old[0], hA_old[0])
                gA[0] = g_mem_A_comp * np.power(mA[0], 3) * hA[0] 
                
                mH[0] = mHUpdate(V_old[0], mH_old[0])
                gH[0] = g_mem_H_comp * mH[0]
                
                mM[0] = mMUpdate(V_old[0], mM_old[0])
                gM[0] = g_mem_M_comp * np.power(mM[0], 2)
                
                mCaL[0] = mCaLUpdate(V_old[0], mCaL_old[0])
                hCaL[0] = hCaLUpdate(V_old[0], hCaL_old[0])
                gCaL[0] = g_mem_CaL_comp * mCaL[0] * hCaL[0]
                
                mKCa[0] = mKCaUpdate(mKCa_old[0], CaS_old[0])
                gKCa[0] = g_mem_KCa_comp * mKCa[0]
                CaS[0] = CaSUpdate(CaS_old[0], gCaL[0] * (V_old[0] - E_CaL), comp_vol)
                
        
                # integration parameters
                A[0] = (g_main_b1+g_main_b2)/C_mem_comp #in units of uS/nF
                ####
                B[0] = -(gNa[0]+gKd[0]+gleak[0]+gGABA[0]+gA[0]
                            +gH[0]+gM[0]+gCaL[0]+gKCa[0]+g_ax_comp+g_main_b1+g_main_b2)/C_mem_comp #in units of uS/nF
                ####
                C[0] = g_ax_comp/C_mem_comp #in units of uS/nF
                ####
                D[0] = (gNa[0]*E_Na+gKd[0]*E_K+gleak[0]*E_leak+gGABA[0]*E_GABA
                        + gA[0]*E_A + gH[0]*E_H + gM[0]*E_M + gCaL[0]*E_CaL + gKCa[0]*E_KCa)/C_mem_comp #in units of nA/nF
                ####
                a[0] = A[0]*dt #in units of uS*ms/nF
                b[0] = B[0]*dt #in units of uS*ms/nF
                c[0] = C[0]*dt #in units of uS*ms/nF              
                d[0] = (D[0]+(g_main_b1*b1.V[b1.n_comp-1]+g_main_b2*b2.V[b2.n_comp-1])/C_mem_comp+B[0]*V[0]+C[0]*V[1])*dt #in units of nA*ms/nF
        
                b_p[0] = b[0] # _p stands for prime as in Dayan and Abbott appendix 6
                d_p[0] = d[0]
                
                for i in range(1, n_comp-1, 1): #middle compartments
                    # Na
                    mNa[i] = mNaUpdate(V_old[i], mNa_old[i])
                    hNa[i] = hNaUpdate(V_old[i], hNa_old[i])
                    gNa[i] = g_mem_Na_comp * np.power(mNa[i], 2) * hNa[i]
        
                    # Kd
                    nKd[i] = nKdUpdate(V_old[i], nKd_old[i])
        
                    gKd[i] = g_mem_Kd_comp * np.power(nKd[i], 4)
        
                    # leak
                    gleak[i] = g_mem_leak_comp 
                    
                    mA[i] = mAUpdate(V_old[i], mA_old[i])
                    hA[i] = hAUpdate(V_old[i], hA_old[i])
                    gA[i] = g_mem_A_comp * np.power(mA[i], 3) * hA[i] 
                    
                    mH[i] = mHUpdate(V_old[i], mH_old[i])
                    gH[i] = g_mem_H_comp * mH[i]
                    
                    mM[i] = mMUpdate(V_old[i], mM_old[i])
                    gM[i] = g_mem_M_comp * np.power(mM[i], 2)
                    
                    mCaL[i] = mCaLUpdate(V_old[i], mCaL_old[i])
                    hCaL[i] = hCaLUpdate(V_old[i], hCaL_old[i])
                    gCaL[i] = g_mem_CaL_comp * mCaL[i] * hCaL[i]
        
                    mKCa[i] = mKCaUpdate(mKCa_old[i], CaS_old[i])
                    gKCa[i] = g_mem_KCa_comp * mKCa[i]
                    CaS[i] = CaSUpdate(CaS_old[i], gCaL[i] * (V_old[i] - E_CaL),comp_vol)#gCaL[i] * (V_old[i] - E_CaL)
                    #if i == mid_ind:
                        #print(f'I_CaL: {gCaL[i] * (V_old[i] - E_CaL)}')
                        #print(f'[Ca]: {CaS[i]}')
        
                    
                    # integration parameters
                    A[i] = g_ax_comp/C_mem_comp #in units of uS/nF
                    #####
                    B[i] = -(gNa[i]+gKd[i]+gleak[i]+gGABA[i]+gA[i]
                                +gH[i]+gM[i]+gCaL[i]+gKCa[i]+2.0*g_ax_comp)/C_mem_comp
                    #####
                    C[i] = g_ax_comp/C_mem_comp #in units of uS/nF
                    #####
                    D[i] = (gNa[i]*E_Na+gKd[i]*E_K+gleak[i]*E_leak+gGABA[i]*E_GABA
                            + gA[i]*E_A + gH[i]*E_H + gM[i]*E_M + gCaL[i]*E_CaL + gKCa[i]*E_KCa)/C_mem_comp
                    #####
                    a[i] = A[i]*dt    
                    b[i] = B[i]*dt
                    c[i] = C[i]*dt
                    d[i] = (D[i]+A[i]*V[i-1]+B[i]*V[i]+C[i]*V[i+1])*dt
                    
                    b_p[i] = b[i] + a[i]*c[i-1]/(1-b_p[i-1]) #equation 6.54 in D&A
                    d_p[i] = d[i] + a[i]*d_p[i-1]/(1-b_p[i-1]) #equation 6.55 in D&A
                
                #compartment n_comp-1
                # Na
                mNa[n_comp-1] = mNaUpdate(V_old[n_comp-1], mNa_old[n_comp-1])
                hNa[n_comp-1] = hNaUpdate(V_old[n_comp-1], hNa_old[n_comp-1])
                gNa[n_comp-1] = g_mem_Na_comp * np.power(mNa[n_comp-1], 2) * hNa[n_comp-1]
        
                # Kd
                nKd[n_comp-1] = nKdUpdate(V_old[n_comp-1], nKd_old[n_comp-1])
                gKd[n_comp-1] = g_mem_Kd_comp * np.power(nKd[n_comp-1], 4)
                        # leak
                gleak[n_comp-1] = g_mem_leak_comp 
                mA[n_comp-1] = mAUpdate(V_old[n_comp-1], mA_old[n_comp-1])
                hA[n_comp-1] = hAUpdate(V_old[n_comp-1], hA_old[n_comp-1])
                gA[n_comp-1] = g_mem_A_comp * np.power(mA[n_comp-1], 3) * hA[n_comp-1] 
                
                mH[n_comp-1] = mHUpdate(V_old[n_comp-1], mH_old[n_comp-1])
                gH[n_comp-1] = g_mem_H_comp * mH[n_comp-1]
                
                mM[n_comp-1] = mMUpdate(V_old[n_comp-1], mM_old[n_comp-1])
                gM[n_comp-1] = g_mem_M_comp * np.power(mM[n_comp-1], 2)
                
                mCaL[n_comp-1] = mCaLUpdate(V_old[n_comp-1], mCaL_old[n_comp-1])
                hCaL[n_comp-1] = hCaLUpdate(V_old[n_comp-1], hCaL_old[n_comp-1])
                gCaL[n_comp-1] = g_mem_CaL_comp * mCaL[n_comp-1] * hCaL[n_comp-1]
                
                mKCa[n_comp-1] = mKCaUpdate(mKCa_old[n_comp-1], CaS_old[n_comp-1])
                gKCa[n_comp-1] = g_mem_KCa_comp * mKCa[n_comp-1]
                CaS[n_comp-1] = CaSUpdate(CaS_old[n_comp-1], gCaL[n_comp-1] * (V_old[n_comp-1] - E_CaL),comp_vol)
        
        
                # integration parameters
                A[n_comp-1] = g_ax_comp/C_mem_comp #in units of uS/nF
                ####
                B[n_comp-1] = -(gNa[n_comp-1]+gKd[n_comp-1]+gleak[n_comp-1]+gGABA[n_comp-1]
                                +gA[n_comp-1]+gH[n_comp-1]+gM[n_comp-1]+gCaL[n_comp-1]+gKCa[n_comp-1]
                                +g_ax_comp)/C_mem_comp
                ####
                C[n_comp-1] = 0.0 #in units of uS/nF
                ####
                D[n_comp-1] = (gNa[n_comp-1]*E_Na+gKd[n_comp-1]*E_K+gleak[n_comp-1]*E_leak+gGABA[n_comp-1]*E_GABA+I_stim_now
                               + gA[n_comp-1]*E_A + gH[n_comp-1]*E_H + gM[n_comp-1]*E_M + gCaL[n_comp-1]*E_CaL + gKCa[n_comp-1]*E_KCa)/C_mem_comp
                ####
                a[n_comp-1] = A[n_comp-1]*dt    
                b[n_comp-1] = B[n_comp-1]*dt
                c[n_comp-1] = C[n_comp-1]*dt
                d[n_comp-1] = (D[n_comp-1]+A[n_comp-1]*V[n_comp-2]+B[n_comp-1]*V[n_comp-1])*dt
                    
                b_p[n_comp-1] = b[n_comp-1] + a[n_comp-1]*c[n_comp-2]/(1-b_p[n_comp-2]) #equation 6.54 in D&A
                d_p[n_comp-1] = d[n_comp-1] + a[n_comp-1]*d_p[n_comp-2]/(1-b_p[n_comp-2]) #equation 6.55 in D&A
               
                
               ################
                deltaV[n_comp-1] = d_p[n_comp-1]/(1-b_p[n_comp-1]) #equation 6.56 in D&A, update voltage for last compartment
                V[n_comp-1] = V_old[n_comp-1] + deltaV[n_comp-1]
                        
                for i in range(n_comp-1, 0, -1): # step through the middle compartments backward to update voltages
                    deltaV[i-1] = (c[i-1]*deltaV[i]+d_p[i-1])/(1-b_p[i-1])
                    V[i-1] = V_old[i-1] + deltaV[i-1]
        
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
        
                ###branch compartments n-1 updated based on main axon compartment 0 voltage            
                


                I_stim.append(I_stim_now) #record stimulus current
                Cas_monitor_main.append(CaS[mid_ind])
                V_rec_first.append(V[n_comp-1]) #record voltage in stimulated compartment
                V_rec_second.append(V[n_comp-2]) #record voltage in compartment next to stimulus
                V_rec_middle.append(V[mid_ind]) #record voltage in middle compartment
                V_rec_last.append(V[0]) #record voltage in branch compartment
                V_rec_nexttoGABA.append(V[2]) #recorded voltage time course in compartment next to GABA in mV
                
                b1.V_rec_first.append(b1.V[b1.n_comp-1]) #record voltage in first compartment
                b1.V_rec_second.append(b1.V[b1.n_comp-2]) #record voltage in second compartment
                b1.V_rec_middle.append(b1.V[b1.mid_ind]) #record voltage in middle compartment
                b1.V_rec_last.append(b1.V[0]) #record voltage in last compartment    
                b1.V_rec_nexttoGABA.append(b1.V[b1.n_comp-3]) #recorded voltage time course in compartment next to GABA in mV
        
                b2.V_rec_first.append(b2.V[b2.n_comp-1]) #record voltage in first compartment
                b2.V_rec_second.append(b2.V[b2.n_comp-2]) #record voltage in second compartment
                b2.V_rec_middle.append(b2.V[b2.mid_ind]) #record voltage in middle compartment
                b2.V_rec_last.append(b2.V[0]) #record voltage in last compartment
                b2.V_rec_nexttoGABA.append(b2.V[b2.n_comp-3]) #recorded voltage time course in compartment next to GABA in mV
                main_First.findSpike(V_rec_first[-1])
                main_Middle.findSpike(V_rec_middle[-1]) 
                main_Last.findSpike(V_rec_last[-1]) 
                b1.First.findSpike(b1.V_rec_first[-1]) 
                b1.Middle.findSpike(b1.V_rec_middle[-1]) 
                b1.Last.findSpike(b1.V_rec_last[-1]) 
                b2.First.findSpike(b2.V_rec_first[-1])  
                b2.Middle.findSpike(b2.V_rec_middle[-1])  
                b2.Last.findSpike(b2.V_rec_last[-1]) 
        
                
        # END SIMULATION
        
        
        plotI_stim(t, I_stim, 'Main', t_total) #Comment out if you don't want to output the graph
        plotVfirst(t, V_rec_first, 'Main', t_total) #Comment out if you don't want to output the graph
        #plotVsecond(t, V_rec_second, 'Main', t_total) #Comment out if you don't want to output the graph
        plotVmiddle(t, V_rec_middle, 'Main', t_total) #Comment out if you don't want to output the graph
        #plotVnexttoGABA(t, V_rec_nexttoGABA, 'Main', t_total) #Comment out if you don't want to output the graph
        plotVlast(t, V_rec_last, 'Main', t_total) #Comment out if you don't want to output the graph
        
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
        plotCaS(t, Cas_monitor_main, 'Main_middle', t_total)
        print("Your simulations are successfully completed!") 
      
        
        print("main_First_Spike_Number: %d" % main_First.Seg_spike_num)
        print("main_Middle_Spike_Number: %d" % main_Middle.Seg_spike_num)
        print("main_Last_Spike_Number: %d" % main_Last.Seg_spike_num)
        print("b1.First_Spike_Number: %d" % b1.First.Seg_spike_num)
        print("b1.Middle_Spike_Number: %d" % b1.Middle.Seg_spike_num)
        print("b1.Last_Spike_Number: %d" % b1.Last.Seg_spike_num)
        print("b2.First_Spike_Number: %d" % b2.First.Seg_spike_num)
        print("b2.Middle_Spike_Number: %d" % b2.Middle.Seg_spike_num)
        print("b2.Last_Spike_Number: %d" % b2.Last.Seg_spike_num)
        #arrayR = [("mf", main_First.Seg_spike_num), ("mm", main_Middle.Seg_spike_num), ("ml", main_Last.Seg_spike_num), ("b1f", b1.First.Seg_spike_num), ("b1m", b1.Middle.Seg_spike_num),
         #         ("b1l", b1.Last.Seg_spike_num), ("b2f", b2.First.Seg_spike_num), ("b2m", b2.Middle.Seg_spike_num), ("b2l", b2.Last.Seg_spike_num)]
        #a = "b2.dim: " + str(round(index1 * 0.05, 2))
        #dict[a] = arrayR
        index1 += 1
        oneDarray.append(b2.Last.Seg_spike_num)
    twoDarray.append(oneDarray)
    index2 += 1
#for i in range(1, 11):
 #   string1 = "b2.dim: " + str(round(i * 0.05, 2))
  #  print(string1)
   # print(dict[string1])
   # print("-------")

#dataFrame = pd.DataFrame(twoDarray)
#dataFrame.to_csv('Temp_2dimensionalNew.csv')


# writing data to CSV file
#np.savetxt('Temp_secondtry_39_6.txt', np.column_stack([t,V_rec_second,V_rec_middle,V_rec_nexttoGABA,V_rec_last,b1.V_rec_first,b1.V_rec_nexttoGABA,b1.V_rec_middle,b1.V_rec_last,b2.V_rec_first,b2.V_rec_nexttoGABA,b2.V_rec_middle,b2.V_rec_last]))

