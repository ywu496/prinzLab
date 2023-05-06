import pylab as plt
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import signal
import csv

### ALL ARRAYS ACROSS COMPARTMENTS WILL USE INDICES 0 THROUGH n_comp-1 
### THIS IS DIFFERENT FROM DAYAN & ABBOTT, which uses 1 THROUGH N

# time parameters
dt = 0.01 #numerical integration time step
t_total = 30.0 #total simulation time in ms
t_now = 0.0 #time right now in ms
t = [t_now] #time array in ms

# stimulus parameters (stimulus is current pulse injected in first compartment)
t_stimstart = 2.0 #stimulation start time
t_stimend = 2.5 #stimulation end time
I_stim_amp = 0.3 #stimulation current amplitude in nA, 0.15 is good
I_stim_now = 0.0 #stimulus current in nA
I_stim = [0.0] #stimulus current time course

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
E_K_uniform_default = -90.0 #K reversal potential in mV, according to McKinnon, default is -90mV
E_K_uniform = E_K_uniform_default * temp_K / default_temp_K #adjust according to Nernst equation
E_leak_uniform = E_leak_uniform_default * temp_K / default_temp_K #adjust according to Nernst equation
E_GABA_uniform_default = -60.0 #GABA (i.e., Cl-) reversal potential in mV, see Prescott paper fig. 6, vary from -65mV (control) to -50mV (SCI)
E_GABA_uniform = E_GABA_uniform_default * temp_K / default_temp_K

G_Na_abs_uniform = 7.0 #Na conductance in nS, based on McKinnon Table 1, default is 300nS
G_Kd_abs_uniform = 100.0 #Kd conductance in nS, based on McKinnon Table 1, default is 2000nS
G_leak_abs_uniform = 1.0 #leak conductance in nS, based on McKinnon Table 1, default is 1nS
G_GABA_uniform = 0.0 #GABA conductance in nS

# model geometry settings
# main axon
L = 1000.0 #length of axon in um, distance between sympathetic ganglia is about 1mm
d = 0.5 #diameter of axon in um
n_comp = 100 #number of compartments
# branch 1
b1_L = 1000.0 #length of axon in um, distance between sympathetic ganglia is about 1mm
b1_d = 0.5 #diameter of axon in um
b1_n_comp = 100 #number of compartments
# branch2
b2_L = 1000.0 #length of axon in um, distance between sympathetic ganglia is about 1mm
b2_d = 0.1 #diameter of axon in um
b2_n_comp = 100 #number of compartments

# geometry-related
mid_ind = int(n_comp/2) #index of compartment roughly in middle of cable
l_comp = L/n_comp #length of compartment in um 
A_mem_comp = np.pi*d*l_comp*1e-8 #membrane surface area of compartment in square cm
A_cross_comp = np.pi*d*d*1e-8/4 #axon cross-sectional in square cm
b1_mid_ind = int(b1_n_comp/2) #index of compartment roughly in middle of cable
b1_l_comp = b1_L/b1_n_comp #length of compartment in um 
b1_A_mem_comp = np.pi*b1_d*b1_l_comp*1e-8 #membrane surface area of compartment in square cm
b1_A_cross_comp = np.pi*b1_d*b1_d*1e-8/4 #axon cross-sectional in square cm
b2_mid_ind = int(b2_n_comp/2) #index of compartment roughly in middle of cable
b2_l_comp = b2_L/b2_n_comp #length of compartment in um 
b2_A_mem_comp = np.pi*b2_d*b2_l_comp*1e-8 #membrane surface area of compartment in square cm
b2_A_cross_comp = np.pi*b2_d*b2_d*1e-8/4 #axon cross-sectional in square cm

# capacitance related
C_mem_comp = A_mem_comp*1e3 #membrane capacitace of individual compartment in nF, assuming 1uF/cm2
conductance_scaling_factor = 1e6*C_mem_comp/100 #factor used to scale conductances because McKinnon et al model has 100pF capacitance
b1_C_mem_comp = b1_A_mem_comp*1e3 #membrane capacitace of individual compartment in nF, assuming 1uF/cm2
b1_conductance_scaling_factor = 1e6*b1_C_mem_comp/100 #factor used to scale conductances because McKinnon et al model has 100pF capacitance
b2_C_mem_comp = b2_A_mem_comp*1e3 #membrane capacitace of individual compartment in nF, assuming 1uF/cm2
b2_conductance_scaling_factor = 1e6*b2_C_mem_comp/100 #factor used to scale conductances because McKinnon et al model has 100pF capacitance

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
# branch 1
b1_G_leak_abs = G_leak_abs_uniform #leak conductance in nS, based on McKinnon Table 1, default is 1nS
b1_g_mem_leak_comp = b1_conductance_scaling_factor*b1_G_leak_abs/1e3 #membrane leak conductance per compartment in uS
b1_E_leak = E_leak_uniform #leak reversal potential in mV, according to McKinnon, default is -55mV
# Na:
b1_G_Na_abs = G_Na_abs_uniform #Na conductance in nS, based on McKinnon Table 1, default is 300nS
b1_g_mem_Na_comp = b1_conductance_scaling_factor*b1_G_Na_abs/1e3 #membrane Na conductance per compartment in uS
b1_E_Na = E_Na_uniform #Na reversal potential in mV, according to McKinnon, default is 60mV
# Kd:
b1_G_Kd_abs = G_Kd_abs_uniform #Kd conductance in nS, based on McKinnon Table 1, default is 2000nS
b1_g_mem_Kd_comp = b1_conductance_scaling_factor*b1_G_Kd_abs/1e3 #membrane Kd conductance per compartment in uS
b1_E_K = E_K_uniform #K reversal potential in mV, according to McKinnon, default is -90mV
# GABA conductance for compartments proximal to (but not at) branch point
b1_G_GABA_abs = G_GABA_uniform #GABA conductance in nS
b1_g_mem_GABA = conductance_scaling_factor*G_GABA_abs/1e3 #GABA conductance per compartment in uS
b1_E_GABA = E_GABA_uniform #GABA (i.e., Cl-) reversal potential in mV, see Prescott paper fig. 6, vary from -65mV (control) to -50mV (SCI)
# branch 2
b2_G_leak_abs = G_leak_abs_uniform #leak conductance in nS, based on McKinnon Table 1, default is 1nS
b2_g_mem_leak_comp = b2_conductance_scaling_factor*b2_G_leak_abs/1e3 #membrane leak conductance per compartment in uS
b2_E_leak = E_leak_uniform #leak reversal potential in mV, according to McKinnon, default is -55mV
# Na:
b2_G_Na_abs = G_Na_abs_uniform #Na conductance in nS, based on McKinnon Table 1, default is 300nS
b2_g_mem_Na_comp = b2_conductance_scaling_factor*b2_G_Na_abs/1e3 #membrane Na conductance per compartment in uS
b2_E_Na = E_Na_uniform #Na reversal potential in mV, according to McKinnon, default is 60mV
# Kd:
b2_G_Kd_abs = G_Kd_abs_uniform #Kd conductance in nS, based on McKinnon Table 1, default is 2000nS
b2_g_mem_Kd_comp = b2_conductance_scaling_factor*b2_G_Kd_abs/1e3 #membrane Kd conductance per compartment in uS
b2_E_K = E_K_uniform #K reversal potential in mV, according to McKinnon, default is -90mV
# GABA conductance for compartments proximal to (but not at) branch point
b2_G_GABA_abs = G_GABA_uniform #GABA conductance in nS
b2_g_mem_GABA = conductance_scaling_factor*G_GABA_abs/1e3 #GABA conductance per compartment in uS
b2_E_GABA = E_GABA_uniform #GABA (i.e., Cl-) reversal potential in mV, see Prescott paper fig. 6, vary from -65mV (control) to -50mV (SCI)

# axial conductance related
# main axon
R_ax = 100.0 #axial resistivity in Ohm cm, from https://www.frontiersin.org/articles/10.3389/fncel.2019.00413/full
g_ax_comp = A_cross_comp*1e6/(R_ax*l_comp*1e-4) #axial conductance between compartments in uS
#g_ax_comp = 0.0 #uncouple compartments, for testing
# branch 1
b1_R_ax = 100.0 #axial resistivity in Ohm cm, from https://www.frontiersin.org/articles/10.3389/fncel.2019.00413/full
b1_g_ax_comp = b1_A_cross_comp*1e6/(b1_R_ax*b1_l_comp*1e-4) #axial conductance between compartments in uS
# branch 2
b2_R_ax = 100.0 #axial resistivity in Ohm cm, from https://www.frontiersin.org/articles/10.3389/fncel.2019.00413/full
b2_g_ax_comp = b2_A_cross_comp*1e6/(b2_R_ax*b2_l_comp*1e-4) #axial conductance between compartments in uS
# coupling between main axon and branches, see Dayan & Abbott page 219, fig 6.16
g_main_b1 = 2.0*g_ax_comp*b1_g_ax_comp/(g_ax_comp+b1_g_ax_comp)
g_main_b2 = 2.0*g_ax_comp*b2_g_ax_comp/(g_ax_comp+b2_g_ax_comp)

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

gNa_init = g_mem_Na_comp * np.power(mNa_init, 2) * hNa_init
gKd_init = g_mem_Kd_comp * np.power(nKd_init, 4)
gleak_init = g_mem_leak_comp

INa_init = gNa_init * (V_init - E_Na) #initialize Na currents
IKd_init = gKd_init * (V_init - E_K) #initialize Kd currents
Ileak_init = gleak_init * (V_init - E_leak) #initialize leak currents

#generate and fill in arrays of compartmental voltages, gating variables, and currents
V = [V_init] #array of compartment voltages in mV
mNa = [mNa_init] #array of Na activation variables
hNa = [hNa_init] #array of Na inactivation variables
nKd = [nKd_init] #array of Kd activation variables

gNa = [gNa_init] #array of Na conductances
gKd = [gKd_init] #array of Kd conductances
gleak = [gleak_init] #array of leak conductances
gGABA = [0.0] #array of GABA conductances, will be zeros except for compartments proximal to branch point

INa = [INa_init] #array of Na currents
IKd = [IKd_init] #array of Kd currents
Ileak = [Ileak_init] #array of leak currents

for i in range(1, n_comp):
    V.append(V_init) #initialize compartment voltage array
    mNa.append(mNa_init) #initialize compartment Na activation array
    hNa.append(hNa_init) #initialize compartment Na inactivation array
    nKd.append(nKd_init) #initialize compartment Kd activation array
    
    gNa.append(gNa_init)
    gKd.append(gKd_init)
    gleak.append(gleak_init)
    gGABA.append(0.0)
    
    INa.append(INa_init) #initialize compartment Na current array
    IKd.append(IKd_init) #initialize compartment Na current array
    Ileak.append(Ileak_init) #initialize compartment leak current array
gGABA[1] = g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)

# store previous time step values in _old arrays
V_old = V.copy() #array of previous time step's compartment voltages in mV
mNa_old = mNa.copy() #array of previous time step's compartment Na activations
hNa_old = hNa.copy() #array of previous time step's compartment Na inactivations
nKd_old = nKd.copy() #array of previous time step's compartment Kd activations

gNa_old = gNa.copy()
gKd_old = gKd.copy()
gleak_old = gleak.copy()
gGABA_old = gGABA.copy()

INa_old = INa.copy() #array of previous time step's compartment Na current
IKd_old = IKd.copy() #array of previous time step's compartment Kd current
Ileak_old = Ileak.copy() #array of previous time step's compartment leak current

#Branch b1

# compartmental voltage changes
b1_deltaV = [0.0]
for i in range(1, b1_n_comp, 1):
    b1_deltaV.append(0.0)
b1_deltaV = np.asarray(b1_deltaV)

# initial values for compartmental voltages, gating variables, conductances, and currents
b1_V_init = -65.0 #initialize all voltages
b1_mNa_init = 0.0 #initialize all Na channels to deactivated
b1_hNa_init = 1.0 #initialize all Na channels to deinactivated
b1_nKd_init = 0.0 #initialize all Kd channels to deactivated

b1_gNa_init = b1_g_mem_Na_comp * np.power(b1_mNa_init, 2) * b1_hNa_init
b1_gKd_init = b1_g_mem_Kd_comp * np.power(b1_nKd_init, 4)
b1_gleak_init = b1_g_mem_leak_comp

b1_INa_init = b1_gNa_init * (b1_V_init - b1_E_Na) #initialize Na currents
b1_IKd_init = b1_gKd_init * (b1_V_init - b1_E_K) #initialize Kd currents
b1_Ileak_init = b1_gleak_init * (b1_V_init - b1_E_leak) #initialize leak currents

#generate and fill in arrays of compartmental voltages, gating variables, and currents
b1_V = [b1_V_init] #array of compartment voltages in mV
b1_mNa = [b1_mNa_init] #array of Na activation variables
b1_hNa = [b1_hNa_init] #array of Na inactivation variables
b1_nKd = [b1_nKd_init] #array of Kd activation variables

b1_gNa = [b1_gNa_init] #array of Na conductances
b1_gKd = [b1_gKd_init] #array of Kd conductances
b1_gleak = [b1_gleak_init] #array of leak conductances
b1_gGABA = [0.0] #array of GABA conductances, will be zeros except for compartments proximal to branch point

b1_INa = [b1_INa_init] #array of Na currents
b1_IKd = [b1_IKd_init] #array of Kd currents
b1_Ileak = [b1_Ileak_init] #array of leak currents

for i in range(1, b1_n_comp):
    b1_V.append(b1_V_init) #initialize compartment voltage array
    b1_mNa.append(b1_mNa_init) #initialize compartment Na activation array
    b1_hNa.append(b1_hNa_init) #initialize compartment Na inactivation array
    b1_nKd.append(b1_nKd_init) #initialize compartment Kd activation array
    
    b1_gNa.append(b1_gNa_init)
    b1_gKd.append(b1_gKd_init)
    b1_gleak.append(b1_gleak_init)
    b1_gGABA.append(0.0)
    
    b1_INa.append(b1_INa_init) #initialize compartment Na current array
    b1_IKd.append(b1_IKd_init) #initialize compartment Na current array
    b1_Ileak.append(b1_Ileak_init) #initialize compartment leak current array
b1_gGABA[b1_n_comp-2] = b1_g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)

# store previous time step values in _old arrays
b1_V_old = b1_V.copy() #array of previous time step's compartment voltages in mV
b1_mNa_old = b1_mNa.copy() #array of previous time step's compartment Na activations
b1_hNa_old = b1_hNa.copy() #array of previous time step's compartment Na inactivations
b1_nKd_old = b1_nKd.copy() #array of previous time step's compartment Kd activations

b1_gNa_old = b1_gNa.copy()
b1_gKd_old = b1_gKd.copy()
b1_gleak_old = b1_gleak.copy()
b1_gGABA_old = b1_gGABA.copy()

b1_INa_old = b1_INa.copy() #array of previous time step's compartment Na current
b1_IKd_old = b1_IKd.copy() #array of previous time step's compartment Kd current
b1_Ileak_old = b1_Ileak.copy() #array of previous time step's compartment leak current

#branch b2

# compartmental voltage changes
b2_deltaV = [0.0]
for i in range(1, b2_n_comp, 1):
    b2_deltaV.append(0.0)
b2_deltaV = np.asarray(b2_deltaV)

# initial values for compartmental voltages, gating variables, conductances, and currents
b2_V_init = -65.0 #initialize all voltages
b2_mNa_init = 0.0 #initialize all Na channels to deactivated
b2_hNa_init = 1.0 #initialize all Na channels to deinactivated
b2_nKd_init = 0.0 #initialize all Kd channels to deactivated

b2_gNa_init = b2_g_mem_Na_comp * np.power(b2_mNa_init, 2) * b2_hNa_init
b2_gKd_init = b2_g_mem_Kd_comp * np.power(b2_nKd_init, 4)
b2_gleak_init = b2_g_mem_leak_comp

b2_INa_init = b2_gNa_init * (b2_V_init - b2_E_Na) #initialize Na currents
b2_IKd_init = b2_gKd_init * (b2_V_init - b2_E_K) #initialize Kd currents
b2_Ileak_init = b2_gleak_init * (b2_V_init - b2_E_leak) #initialize leak currents

#generate and fill in arrays of compartmental voltages, gating variables, and currents
b2_V = [b2_V_init] #array of compartment voltages in mV
b2_mNa = [b2_mNa_init] #array of Na activation variables
b2_hNa = [b2_hNa_init] #array of Na inactivation variables
b2_nKd = [b2_nKd_init] #array of Kd activation variables

b2_gNa = [b2_gNa_init] #array of Na conductances
b2_gKd = [b2_gKd_init] #array of Kd conductances
b2_gleak = [b2_gleak_init] #array of leak conductances
b2_gGABA = [0.0] #array of GABA conductances, will be zeros except for compartments proximal to branch point

b2_INa = [b2_INa_init] #array of Na currents
b2_IKd = [b2_IKd_init] #array of Kd currents
b2_Ileak = [b2_Ileak_init] #array of leak currents

for i in range(1, b2_n_comp):
    b2_V.append(b2_V_init) #initialize compartment voltage array
    b2_mNa.append(b2_mNa_init) #initialize compartment Na activation array
    b2_hNa.append(b2_hNa_init) #initialize compartment Na inactivation array
    b2_nKd.append(b2_nKd_init) #initialize compartment Kd activation array
    
    b2_gNa.append(b2_gNa_init)
    b2_gKd.append(b2_gKd_init)
    b2_gleak.append(b2_gleak_init)
    b2_gGABA.append(0.0)
    
    b2_INa.append(b2_INa_init) #initialize compartment Na current array
    b2_IKd.append(b2_IKd_init) #initialize compartment Na current array
    b2_Ileak.append(b2_Ileak_init) #initialize compartment leak current array
b2_gGABA[b2_n_comp-2] = b2_g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)

# store previous time step values in _old arrays
b2_V_old = b2_V.copy() #array of previous time step's compartment voltages in mV
b2_mNa_old = b2_mNa.copy() #array of previous time step's compartment Na activations
b2_hNa_old = b2_hNa.copy() #array of previous time step's compartment Na inactivations
b2_nKd_old = b2_nKd.copy() #array of previous time step's compartment Kd activations

b2_gNa_old = b2_gNa.copy()
b2_gKd_old = b2_gKd.copy()
b2_gleak_old = b2_gleak.copy()
b2_gGABA_old = b2_gGABA.copy()

b2_INa_old = b2_INa.copy() #array of previous time step's compartment Na current
b2_IKd_old = b2_IKd.copy() #array of previous time step's compartment Kd current
b2_Ileak_old = b2_Ileak.copy() #array of previous time step's compartment leak current

# initialize compartmental integration parameters
A = [(g_main_b1+g_main_b2)/C_mem_comp] #initialize branch compartment, in units of uS/nF
B = [-(gNa[0]+gKd[0]+gleak[0]+gGABA[0]+g_ax_comp+g_main_b1+g_main_b2)/C_mem_comp] #in units of uS/nF
C = [g_ax_comp/C_mem_comp] #in units of uS/nF
D = [(gNa[0]*E_Na+gKd[0]*E_K+gleak[0]*E_leak+gGABA[0]*E_GABA)/C_mem_comp] #in units of nA/nF
a = [A[0]*dt] #in units of uS*ms/nF
b = [B[0]*dt] #in units of uS*ms/nF
c = [C[0]*dt] #in units of uS*ms/nF
d = [(D[0]+(g_main_b1*b1_V[b1_n_comp-1]+g_main_b2*b2_V[b2_n_comp-1])/C_mem_comp+B[0]*V[0]+C[0]*V[1])*dt] #in units of nA*ms/nF

for i in range(1, n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
    A.append(g_ax_comp/C_mem_comp) #in units of uS/nF
    B.append(-(gNa[i]+gKd[i]+gleak[i]+gGABA[i]+2.0*g_ax_comp)/C_mem_comp) #in units of uS/nF
    C.append(g_ax_comp/C_mem_comp) #in units of uS/nF
    D.append((gNa[i]*E_Na+gKd[i]*E_K+gleak[i]*E_leak+gGABA[i]*E_GABA)/C_mem_comp) #in units of nA/nF
    a.append(A[i]*dt)    
    b.append(B[i]*dt) 
    c.append(C[i]*dt)
    d.append((D[i]+A[i]*V[i-1]+B[i]*V[i]+C[i]*V[i+1])*dt)

A.append(g_ax_comp/C_mem_comp) #in units of uS/nF, for last compartment
B.append(-(gNa[n_comp-1]+gKd[n_comp-1]+gleak[n_comp-1]+gGABA[n_comp-1]+g_ax_comp)/C_mem_comp) #in units of uS/nF
C.append(0.0) #in units of uS/nF
D.append((gNa[n_comp-1]*E_Na+gKd[n_comp-1]*E_K+gleak[n_comp-1]*E_leak+gGABA[n_comp-1]*E_GABA+I_stim_now)/C_mem_comp) #in units of nA/nF
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

# recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
V_rec_first = [V_init] #recorded voltage time course in first compartment in mV
V_rec_second = [V_init] #recorded voltage time course in second compartment in mV
V_rec_middle = [V_init] #recorded voltage time course in middle compartment in mV
V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV
V_rec_last = [V_init] #recorded voltage time course in last compartment in mV

#for testing
testsignal1 = [0.0]
testsignal2 = [0.0]


# initialize compartmental integration parameters
b1_A = [0.0] #initialize terminal compartment, in units of uS/nF
b1_B = [-(b1_gNa[0]+b1_gKd[0]+b1_gleak[0]+gGABA[0]+b1_g_ax_comp)/b1_C_mem_comp] #in units of uS/nF
b1_C = [b1_g_ax_comp/b1_C_mem_comp] #in units of uS/nF
b1_D = [(b1_gNa[0]*b1_E_Na+b1_gKd[0]*b1_E_K+b1_gleak[0]*b1_E_leak+b1_gGABA[0]*b1_E_GABA)/b1_C_mem_comp] #in units of nA/nF
b1_a = [b1_A[0]*dt] #in units of uS*ms/nF
b1_b = [b1_B[0]*dt] #in units of uS*ms/nF
b1_c = [b1_C[0]*dt] #in units of uS*ms/nF
b1_d = [(b1_D[0]+b1_B[0]*b1_V[0]+b1_C[0]*b1_V[1])*dt] #in units of nA*ms/nF

for i in range(1, b1_n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
    b1_A.append(b1_g_ax_comp/b1_C_mem_comp) #in units of uS/nF
    b1_B.append(-(b1_gNa[i]+b1_gKd[i]+b1_gleak[i]+b1_gGABA[i]+2.0*b1_g_ax_comp)/b1_C_mem_comp) #in units of uS/nF
    b1_C.append(b1_g_ax_comp/b1_C_mem_comp) #in units of uS/nF
    b1_D.append((b1_gNa[i]*b1_E_Na+b1_gKd[i]*b1_E_K+b1_gleak[i]*b1_E_leak+b1_gGABA[i]*b1_E_GABA)/b1_C_mem_comp) #in units of nA/nF
    b1_a.append(b1_A[i]*dt)    
    b1_b.append(b1_B[i]*dt) 
    b1_c.append(b1_C[i]*dt)
    b1_d.append((b1_D[i]+b1_A[i]*b1_V[i-1]+b1_B[i]*b1_V[i]+b1_C[i]*b1_V[i+1])*dt)

b1_A.append(b1_g_ax_comp/b1_C_mem_comp) #in units of uS/nF, for branch compartment
b1_B.append(-(b1_gNa[b1_n_comp-1]+b1_gKd[b1_n_comp-1]+b1_gleak[b1_n_comp-1]+b1_gGABA[b1_n_comp-1]+b1_g_ax_comp+g_main_b1)/b1_C_mem_comp) #in units of uS/nF
b1_C.append(g_main_b1/b1_C_mem_comp) #in units of uS/nF
b1_D.append((b1_gNa[b1_n_comp-1]*b1_E_Na+b1_gKd[b1_n_comp-1]*b1_E_K+b1_gleak[b1_n_comp-1]*b1_E_leak+b1_gGABA[b1_n_comp-1]*b1_E_GABA)/b1_C_mem_comp) #in units of nA/nF
b1_a.append(b1_A[b1_n_comp-1]*dt)    
b1_b.append(b1_B[b1_n_comp-1]*dt) 
b1_c.append(b1_C[b1_n_comp-1]*dt)
b1_d.append((b1_D[b1_n_comp-1]+b1_A[b1_n_comp-1]*b1_V[b1_n_comp-2]+b1_B[b1_n_comp-1]*b1_V[b1_n_comp-1]+b1_C[b1_n_comp-1]*V[0])*dt)

b1_A = np.asarray(b1_A)
b1_B = np.asarray(b1_B)
b1_C = np.asarray(b1_C)
b1_D = np.asarray(b1_D)
b1_a = np.asarray(b1_a)
b1_b = np.asarray(b1_b)
b1_c = np.asarray(b1_c)
b1_d = np.asarray(b1_d)

b1_b_p = b1_b.copy() 
b1_d_p = b1_d.copy()

# recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
b1_V_rec_first = [b1_V_init] #recorded voltage time course in first compartment in mV
b1_V_rec_second = [b1_V_init] #recorded voltage time course in second compartment in mV
b1_V_rec_middle = [b1_V_init] #recorded voltage time course in middle compartment in mV
b1_V_rec_last = [b1_V_init] #recorded voltage time course in last compartment in mV
b1_V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV

# initialize compartmental integration parameters
b2_A = [0.0] #initialize first compartment, in units of uS/nF
b2_B = [-(b2_gNa[0]+b2_gKd[0]+b2_gleak[0]+b2_gGABA[0]+b2_g_ax_comp)/b2_C_mem_comp] #in units of uS/nF
b2_C = [b2_g_ax_comp/b2_C_mem_comp] #in units of uS/nF
b2_D = [(b2_gNa[0]*b2_E_Na+b2_gKd[0]*b2_E_K+b2_gleak[0]*b2_E_leak+b2_gGABA[0]*b2_E_GABA)/b2_C_mem_comp] #in units of nA/nF
b2_a = [b2_A[0]*dt] #in units of uS*ms/nF
b2_b = [b2_B[0]*dt] #in units of uS*ms/nF
b2_c = [b2_C[0]*dt] #in units of uS*ms/nF
b2_d = [(b2_D[0]+b2_B[0]*b2_V[0]+b2_C[0]*b2_V[1])*dt] #in units of nA*ms/nF

for i in range(1, b2_n_comp-1, 1): #initialize compartment integration parameter arrays, for middle compartments
    b2_A.append(b2_g_ax_comp/b2_C_mem_comp) #in units of uS/nF
    b2_B.append(-(b2_gNa[i]+b2_gKd[i]+b2_gleak[i]+b2_gGABA[i]+2.0*b2_g_ax_comp)/b2_C_mem_comp) #in units of uS/nF
    b2_C.append(b2_g_ax_comp/b2_C_mem_comp) #in units of uS/nF
    b2_D.append((b2_gNa[i]*b2_E_Na+b2_gKd[i]*b2_E_K+b2_gleak[i]*b2_E_leak+b2_gGABA[i]*b2_E_GABA)/b2_C_mem_comp) #in units of nA/nF
    b2_a.append(b2_A[i]*dt)    
    b2_b.append(b2_B[i]*dt) 
    b2_c.append(b2_C[i]*dt)
    b2_d.append((b2_D[i]+b2_A[i]*b2_V[i-1]+b2_B[i]*b2_V[i]+b2_C[i]*b2_V[i+1])*dt)

b2_A.append(b2_g_ax_comp/b2_C_mem_comp) #in units of uS/nF, for last compartment
b2_B.append(-(b2_gNa[b2_n_comp-1]+b2_gKd[b2_n_comp-1]+b2_gleak[b2_n_comp-1]+b2_gGABA[b2_n_comp-1]+b2_g_ax_comp+g_main_b2)/b2_C_mem_comp) #in units of uS/nF
b2_C.append(g_main_b2/b2_C_mem_comp) #in units of uS/nF
b2_D.append((b2_gNa[b2_n_comp-1]*b2_E_Na+b2_gKd[b2_n_comp-1]*b2_E_K+b2_gleak[b2_n_comp-1]*b2_E_leak+b2_gGABA[b2_n_comp-1]*b2_E_GABA)/b2_C_mem_comp) #in units of nA/nF
b2_a.append(b2_A[b2_n_comp-1]*dt)    
b2_b.append(b2_B[b2_n_comp-1]*dt) 
b2_c.append(b2_C[b2_n_comp-1]*dt)
b2_d.append((b2_D[b2_n_comp-1]+b2_A[b2_n_comp-1]*b2_V[b2_n_comp-2]+b2_B[b2_n_comp-1]*b2_V[b2_n_comp-1]+b2_C[b2_n_comp-1]*V[0])*dt)

b2_A = np.asarray(b2_A)
b2_B = np.asarray(b2_B)
b2_C = np.asarray(b2_C)
b2_D = np.asarray(b2_D)
b2_a = np.asarray(b2_a)
b2_b = np.asarray(b2_b)
b2_c = np.asarray(b2_c)
b2_d = np.asarray(b2_d)

b2_b_p = b2_b.copy() 
b2_d_p = b2_d.copy()

# recording electrodes (located in first and last compartment)# _p stands for prime as in Dayan and Abbott appendix 6
b2_V_rec_first = [b2_V_init] #recorded voltage time course in first compartment in mV
b2_V_rec_second = [b2_V_init] #recorded voltage time course in second compartment in mV
b2_V_rec_middle = [b2_V_init] #recorded voltage time course in middle compartment in mV
b2_V_rec_last = [b2_V_init] #recorded voltage time course in last compartment in mV
b2_V_rec_nexttoGABA = [V_init] #recorded voltage time course in compartment next to GABA in mV

#for testing
b2_testsignal1 = [0.0]
b2_testsignal2 = [0.0]

##############################################################################
##############################################################################
##############################################################################

# BEGIN SIMULATION
while t_now+dt < t_total: 
        t_now += dt
        t.append(t_now)
        if t_now>=t_stimstart and t_now<t_stimend:
            I_stim_now=I_stim_amp #apply stimulus current
        else:
            I_stim_now=0.0;
        
        # store previous time step values in _old arrays
        V_old = V.copy() #array of previous time step's compartment voltages in mV
        mNa_old = mNa.copy() #array of previous time step's compartment Na activations
        hNa_old = hNa.copy() #array of previous time step's compartment Na inactivations
        nKd_old = nKd.copy() #array of previous time step's compartment Kd activations

        b1_V_old = b1_V.copy() #array of previous time step's compartment voltages in mV
        b1_mNa_old = b1_mNa.copy() #array of previous time step's compartment Na activations
        b1_hNa_old = b1_hNa.copy() #array of previous time step's compartment Na inactivations
        b1_nKd_old = b1_nKd.copy() #array of previous time step's compartment Kd activations
        
        b2_V_old = b2_V.copy() #array of previous time step's compartment voltages in mV
        b2_mNa_old = b2_mNa.copy() #array of previous time step's compartment Na activations
        b2_hNa_old = b2_hNa.copy() #array of previous time step's compartment Na inactivations
        b2_nKd_old = b2_nKd.copy() #array of previous time step's compartment Kd activations

        # integratinng branch 1 dynamic variables
        # first compartment
        # Na
        alpha = 0.36 * (b1_V_old[0] + 33) / (1 - np.exp(-(b1_V_old[0] + 33) / 3))
        beta = - 0.4 * (b1_V_old[0] + 42) / (1 - np.exp((b1_V_old[0] + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b1_mNa[0] = vinf + (b1_mNa_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
        alpha = - 0.1 * (b1_V_old[0] + 55) / (1 - np.exp((b1_V_old[0] + 55) / 6))
        beta = 4.5 / (1 + np.exp(-b1_V_old[0] / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b1_hNa[0] = vinf + (b1_hNa_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b1_gNa[0] = b1_g_mem_Na_comp * np.power(b1_mNa[0], 2) * b1_hNa[0]

        # Kd
        alpha = 0.0047 * (b1_V_old[0] - 8) / (1 - np.exp(-(b1_V_old[0] - 8) / 12))
        beta = np.exp(-(b1_V_old[0] + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (b1_V_old[0] + 12) / (1 - np.exp(-(b1_V_old[0] + 12) / 12))
        beta = np.exp(-(b1_V_old[0] + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        b1_nKd[0] = vinf + (b1_nKd_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b1_gKd[0] = b1_g_mem_Kd_comp * np.power(b1_nKd[0], 4)

        # leak
        b1_gleak[0] = b1_g_mem_leak_comp 

        # integration parameters
        b1_A[0] = 0.0 #first compartment, in units of uS/nF
        b1_B[0] = -(b1_gNa[0]+b1_gKd[0]+b1_gleak[0]+b1_gGABA[0]+b1_g_ax_comp)/b1_C_mem_comp #in units of uS/nF
        b1_C[0] = b1_g_ax_comp/b1_C_mem_comp #in units of uS/nF
        b1_D[0] = (b1_gNa[0]*b1_E_Na+b1_gKd[0]*b1_E_K+b1_gleak[0]*b1_E_leak+b1_gGABA[0]*b1_E_GABA)/b1_C_mem_comp #in units of nA/nF
        b1_a[0] = b1_A[0]*dt #in units of uS*ms/nF
        b1_b[0] = b1_B[0]*dt #in units of uS*ms/nF
        b1_c[0] = b1_C[0]*dt #in units of uS*ms/nF
        b1_d[0] = (b1_D[0]+b1_B[0]*b1_V[0]+b1_C[0]*b1_V[1])*dt #in units of nA*ms/nF

        b1_b_p[0] = b1_b[0] # _p stands for prime as in Dayan and Abbott appendix 6
        b1_d_p[0] = b1_d[0]
        
        for i in range(1, b1_n_comp-1, 1): #middle compartments
            # Na
            alpha = 0.36 * (b1_V_old[i] + 33) / (1 - np.exp(-(b1_V_old[i] + 33) / 3))
            beta = - 0.4 * (b1_V_old[i] + 42) / (1 - np.exp((b1_V_old[i] + 42) / 20))
            vinf = alpha / (alpha + beta)
            tau = 2 / (alpha + beta)
            tau = tau * taufac
            b1_mNa[i] = vinf + (b1_mNa_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
            alpha = - 0.1 * (b1_V_old[i] + 55) / (1 - np.exp((b1_V_old[i] + 55) / 6))
            beta = 4.5 / (1 + np.exp(-b1_V_old[i] / 10))
            vinf = alpha / (alpha + beta)
            tau = 2 / (alpha + beta)
            tau = tau * taufac
            b1_hNa[i] = vinf + (b1_hNa_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

            b1_gNa[i] = b1_g_mem_Na_comp * np.power(b1_mNa[i], 2) * b1_hNa[i]

            # Kd
            alpha = 0.0047 * (b1_V_old[i] - 8) / (1 - np.exp(-(b1_V_old[i] - 8) / 12))
            beta = np.exp(-(b1_V_old[i] + 127) / 30)
            vinf = alpha / (alpha + beta)
            alpha = 0.0047 * (b1_V_old[i] + 12) / (1 - np.exp(-(b1_V_old[i] + 12) / 12))
            beta = np.exp(-(b1_V_old[i] + 147) / 30)
            tau = 1 / (alpha + beta)
            tau = tau * taufac
            b1_nKd[i] = vinf + (b1_nKd_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

            b1_gKd[i] = b1_g_mem_Kd_comp * np.power(b1_nKd[i], 4)

            # leak
            b1_gleak[i] = b1_g_mem_leak_comp 
            
            # integration parameters
            b1_A[i] = b1_g_ax_comp/b1_C_mem_comp #in units of uS/nF
            b1_B[i] = -(b1_gNa[i]+b1_gKd[i]+b1_gleak[i]+b1_gGABA[i]+2.0*b1_g_ax_comp)/b1_C_mem_comp
            b1_C[i] = b1_g_ax_comp/b1_C_mem_comp #in units of uS/nF
            b1_D[i] = (b1_gNa[i]*b1_E_Na+b1_gKd[i]*b1_E_K+b1_gleak[i]*b1_E_leak+b1_gGABA[i]*b1_E_GABA)/b1_C_mem_comp
            b1_a[i] = b1_A[i]*dt    
            b1_b[i] = b1_B[i]*dt
            b1_c[i] = b1_C[i]*dt
            b1_d[i] = (b1_D[i]+b1_A[i]*b1_V[i-1]+b1_B[i]*b1_V[i]+b1_C[i]*b1_V[i+1])*dt
            
            b1_b_p[i] = b1_b[i] + b1_a[i]*b1_c[i-1]/(1-b1_b_p[i-1]) #equation 6.54 in D&A
            b1_d_p[i] = b1_d[i] + b1_a[i]*b1_d_p[i-1]/(1-b1_b_p[i-1]) #equation 6.55 in D&A 
        
        #last compartment
        # Na
        alpha = 0.36 * (b1_V_old[b1_n_comp-1] + 33) / (1 - np.exp(-(b1_V_old[b1_n_comp-1] + 33) / 3))
        beta = - 0.4 * (b1_V_old[b1_n_comp-1] + 42) / (1 - np.exp((b1_V_old[b1_n_comp-1] + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b1_mNa[b1_n_comp-1] = vinf + (b1_mNa_old[b1_n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
        alpha = - 0.1 * (b1_V_old[b1_n_comp-1] + 55) / (1 - np.exp((b1_V_old[b1_n_comp-1] + 55) / 6))
        beta = 4.5 / (1 + np.exp(-b1_V_old[b1_n_comp-1] / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b1_hNa[b1_n_comp-1] = vinf + (b1_hNa_old[b1_n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b1_gNa[b1_n_comp-1] = b1_g_mem_Na_comp * np.power(b1_mNa[b1_n_comp-1], 2) * b1_hNa[b1_n_comp-1]

        # Kd
        alpha = 0.0047 * (b1_V_old[b1_n_comp-1] - 8) / (1 - np.exp(-(b1_V_old[b1_n_comp-1] - 8) / 12))
        beta = np.exp(-(b1_V_old[b1_n_comp-1] + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (b1_V_old[b1_n_comp-1] + 12) / (1 - np.exp(-(b1_V_old[b1_n_comp-1] + 12) / 12))
        beta = np.exp(-(b1_V_old[b1_n_comp-1] + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        b1_nKd[b1_n_comp-1] = vinf + (b1_nKd_old[b1_n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b1_gKd[b1_n_comp-1] = b1_g_mem_Kd_comp * np.power(b1_nKd[b1_n_comp-1], 4)

        # leak
        b1_gleak[b1_n_comp-1] = b1_g_mem_leak_comp 

        # integration parameters
        b1_A[b1_n_comp-1] = b1_g_ax_comp/b1_C_mem_comp #in units of uS/nF
        b1_B[b1_n_comp-1] = -(b1_gNa[b1_n_comp-1]+b1_gKd[b1_n_comp-1]+b1_gleak[b1_n_comp-1]+b1_gGABA[b1_n_comp-1]+b1_g_ax_comp+g_main_b1)/b1_C_mem_comp
        b1_C[b1_n_comp-1] = g_main_b1/b1_C_mem_comp #in units of uS/nF
        b1_D[b1_n_comp-1] = (b1_gNa[b1_n_comp-1]*b1_E_Na+b1_gKd[b1_n_comp-1]*b1_E_K+b1_gleak[b1_n_comp-1]*b1_E_leak+b1_gGABA[b1_n_comp-1]*b1_E_GABA)/b1_C_mem_comp
        b1_a[b1_n_comp-1] = b1_A[b1_n_comp-1]*dt    
        b1_b[b1_n_comp-1] = b1_B[b1_n_comp-1]*dt
        b1_c[b1_n_comp-1] = b1_C[b1_n_comp-1]*dt
        b1_d[b1_n_comp-1] = (b1_D[b1_n_comp-1]+b1_A[b1_n_comp-1]*b1_V[b1_n_comp-2]+b1_B[b1_n_comp-1]*b1_V[b1_n_comp-1]+b1_C[b1_n_comp-1]*V[0])*dt
            
        b1_b_p[b1_n_comp-1] = b1_b[b1_n_comp-1] + b1_a[b1_n_comp-1]*b1_c[b1_n_comp-2]/(1-b1_b_p[b1_n_comp-2]) #equation 6.54 in D&A
        b1_d_p[b1_n_comp-1] = b1_d[b1_n_comp-1] + b1_a[b1_n_comp-1]*b1_d_p[b1_n_comp-2]/(1-b1_b_p[b1_n_comp-2]) #equation 6.55 in D&A

        # integrating branch 2 dynamic variables
        # first compartment
        # Na
        alpha = 0.36 * (b2_V_old[0] + 33) / (1 - np.exp(-(b2_V_old[0] + 33) / 3))
        beta = - 0.4 * (b2_V_old[0] + 42) / (1 - np.exp((b2_V_old[0] + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b2_mNa[0] = vinf + (b2_mNa_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
        alpha = - 0.1 * (b2_V_old[0] + 55) / (1 - np.exp((b2_V_old[0] + 55) / 6))
        beta = 4.5 / (1 + np.exp(-b2_V_old[0] / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b2_hNa[0] = vinf + (b2_hNa_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b2_gNa[0] = b2_g_mem_Na_comp * np.power(b2_mNa[0], 2) * b2_hNa[0]

        # Kd
        alpha = 0.0047 * (b2_V_old[0] - 8) / (1 - np.exp(-(b2_V_old[0] - 8) / 12))
        beta = np.exp(-(b2_V_old[0] + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (b2_V_old[0] + 12) / (1 - np.exp(-(b2_V_old[0] + 12) / 12))
        beta = np.exp(-(b2_V_old[0] + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        b2_nKd[0] = vinf + (b2_nKd_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b2_gKd[0] = b2_g_mem_Kd_comp * np.power(b2_nKd[0], 4)

        # leak
        b2_gleak[0] = b2_g_mem_leak_comp 

        # integration parameters
        b2_A[0] = 0.0 #first compartment, in units of uS/nF
        b2_B[0] = -(b2_gNa[0]+b2_gKd[0]+b2_gleak[0]+b2_gGABA[0]+b2_g_ax_comp)/b2_C_mem_comp #in units of uS/nF
        b2_C[0] = b2_g_ax_comp/b2_C_mem_comp #in units of uS/nF
        b2_D[0] = (b2_gNa[0]*b2_E_Na+b2_gKd[0]*b2_E_K+b2_gleak[0]*b2_E_leak+b2_gGABA[0]*b2_E_GABA)/b2_C_mem_comp #in units of nA/nF
        b2_a[0] = b2_A[0]*dt #in units of uS*ms/nF
        b2_b[0] = b2_B[0]*dt #in units of uS*ms/nF
        b2_c[0] = b2_C[0]*dt #in units of uS*ms/nF
        b2_d[0] = (b2_D[0]+b2_B[0]*b2_V[0]+b2_C[0]*b2_V[1])*dt #in units of nA*ms/nF

        b2_b_p[0] = b2_b[0] # _p stands for prime as in Dayan and Abbott appendix 6
        b2_d_p[0] = b2_d[0]
        
        for i in range(1, b2_n_comp-1, 1): #middle compartments
            # Na
            alpha = 0.36 * (b2_V_old[i] + 33) / (1 - np.exp(-(b2_V_old[i] + 33) / 3))
            beta = - 0.4 * (b2_V_old[i] + 42) / (1 - np.exp((b2_V_old[i] + 42) / 20))
            vinf = alpha / (alpha + beta)
            tau = 2 / (alpha + beta)
            tau = tau * taufac
            b2_mNa[i] = vinf + (b2_mNa_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
            alpha = - 0.1 * (b2_V_old[i] + 55) / (1 - np.exp((b2_V_old[i] + 55) / 6))
            beta = 4.5 / (1 + np.exp(-b2_V_old[i] / 10))
            vinf = alpha / (alpha + beta)
            tau = 2 / (alpha + beta)
            tau = tau * taufac
            b2_hNa[i] = vinf + (b2_hNa_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

            b2_gNa[i] = b2_g_mem_Na_comp * np.power(b2_mNa[i], 2) * b2_hNa[i]

            # Kd
            alpha = 0.0047 * (b2_V_old[i] - 8) / (1 - np.exp(-(b2_V_old[i] - 8) / 12))
            beta = np.exp(-(b2_V_old[i] + 127) / 30)
            vinf = alpha / (alpha + beta)
            alpha = 0.0047 * (b2_V_old[i] + 12) / (1 - np.exp(-(b2_V_old[i] + 12) / 12))
            beta = np.exp(-(b2_V_old[i] + 147) / 30)
            tau = 1 / (alpha + beta)
            tau = tau * taufac
            b2_nKd[i] = vinf + (b2_nKd_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

            b2_gKd[i] = b2_g_mem_Kd_comp * np.power(b2_nKd[i], 4)

            # leak
            b2_gleak[i] = b2_g_mem_leak_comp 
            
            # integration parameters
            b2_A[i] = b2_g_ax_comp/b2_C_mem_comp #in units of uS/nF
            b2_B[i] = -(b2_gNa[i]+b2_gKd[i]+b2_gleak[i]+b2_gGABA[i]+2.0*b2_g_ax_comp)/b2_C_mem_comp
            b2_C[i] = b2_g_ax_comp/b2_C_mem_comp #in units of uS/nF
            b2_D[i] = (b2_gNa[i]*b2_E_Na+b2_gKd[i]*b2_E_K+b2_gleak[i]*b2_E_leak+b2_gGABA[i]*b2_E_GABA)/b2_C_mem_comp
            b2_a[i] = b2_A[i]*dt    
            b2_b[i] = b2_B[i]*dt
            b2_c[i] = b2_C[i]*dt
            b2_d[i] = (b2_D[i]+b2_A[i]*b2_V[i-1]+b2_B[i]*b2_V[i]+b2_C[i]*b2_V[i+1])*dt
            
            b2_b_p[i] = b2_b[i] + b2_a[i]*b2_c[i-1]/(1-b2_b_p[i-1]) #equation 6.54 in D&A
            b2_d_p[i] = b2_d[i] + b2_a[i]*b2_d_p[i-1]/(1-b2_b_p[i-1]) #equation 6.55 in D&A
        
        
        #last compartment
        # Na
        alpha = 0.36 * (b2_V_old[b2_n_comp-1] + 33) / (1 - np.exp(-(b2_V_old[b2_n_comp-1] + 33) / 3))
        beta = - 0.4 * (b2_V_old[b2_n_comp-1] + 42) / (1 - np.exp((b2_V_old[b2_n_comp-1] + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b2_mNa[b2_n_comp-1] = vinf + (b2_mNa_old[b2_n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
        alpha = - 0.1 * (b2_V_old[b2_n_comp-1] + 55) / (1 - np.exp((b2_V_old[b2_n_comp-1] + 55) / 6))
        beta = 4.5 / (1 + np.exp(-b2_V_old[b2_n_comp-1] / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        b2_hNa[b2_n_comp-1] = vinf + (b2_hNa_old[b2_n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b2_gNa[b2_n_comp-1] = b2_g_mem_Na_comp * np.power(b2_mNa[b2_n_comp-1], 2) * b2_hNa[b2_n_comp-1]

        # Kd
        alpha = 0.0047 * (b2_V_old[b2_n_comp-1] - 8) / (1 - np.exp(-(b2_V_old[b2_n_comp-1] - 8) / 12))
        beta = np.exp(-(b2_V_old[b2_n_comp-1] + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (b2_V_old[b2_n_comp-1] + 12) / (1 - np.exp(-(b2_V_old[b2_n_comp-1] + 12) / 12))
        beta = np.exp(-(b2_V_old[b2_n_comp-1] + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        b2_nKd[b2_n_comp-1] = vinf + (b2_nKd_old[b2_n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        b2_gKd[b2_n_comp-1] = b2_g_mem_Kd_comp * np.power(b2_nKd[b2_n_comp-1], 4)

        # leak
        b2_gleak[b2_n_comp-1] = b2_g_mem_leak_comp 

        # integration parameters
        b2_A[b2_n_comp-1] = b2_g_ax_comp/b2_C_mem_comp #in units of uS/nF
        b2_B[b2_n_comp-1] = -(b2_gNa[b2_n_comp-1]+b2_gKd[b2_n_comp-1]+b2_gleak[b2_n_comp-1]+b2_gGABA[b2_n_comp-1]+b2_g_ax_comp+g_main_b2)/b2_C_mem_comp
        b2_C[b2_n_comp-1] = g_main_b2/b2_C_mem_comp #in units of uS/nF
        b2_D[b2_n_comp-1] = (b2_gNa[b2_n_comp-1]*b2_E_Na+b2_gKd[b2_n_comp-1]*b2_E_K+b2_gleak[b2_n_comp-1]*b2_E_leak+b2_gGABA[b2_n_comp-1]*b2_E_GABA)/b2_C_mem_comp
        b2_a[b2_n_comp-1] = b2_A[b2_n_comp-1]*dt    
        b2_b[b2_n_comp-1] = b2_B[b2_n_comp-1]*dt
        b2_c[b2_n_comp-1] = b2_C[b2_n_comp-1]*dt
        b2_d[b2_n_comp-1] = (b2_D[b2_n_comp-1]+b2_A[b2_n_comp-1]*b2_V[b2_n_comp-2]+b2_B[b2_n_comp-1]*b2_V[b2_n_comp-1]+b2_C[b2_n_comp-1]*V[0])*dt
            
        b2_b_p[b2_n_comp-1] = b2_b[b2_n_comp-1] + b2_a[b2_n_comp-1]*b2_c[b2_n_comp-2]/(1-b2_b_p[b2_n_comp-2]) #equation 6.54 in D&A
        b2_d_p[b2_n_comp-1] = b2_d[b2_n_comp-1] + b2_a[b2_n_comp-1]*b2_d_p[b2_n_comp-2]/(1-b2_b_p[b2_n_comp-2]) #equation 6.55 in D&A

        ###last branch compartments get axial current from main axon compartment 0

        # main axon
        # compartment 0 gets axial current from branch compartments n-1
        # Na
        alpha = 0.36 * (V_old[0] + 33) / (1 - np.exp(-(V_old[0] + 33) / 3))
        beta = - 0.4 * (V_old[0] + 42) / (1 - np.exp((V_old[0] + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        mNa[0] = vinf + (mNa_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
        alpha = - 0.1 * (V_old[0] + 55) / (1 - np.exp((V_old[0] + 55) / 6))
        beta = 4.5 / (1 + np.exp(-V_old[0] / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        hNa[0] = vinf + (hNa_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        gNa[0] = g_mem_Na_comp * np.power(mNa[0], 2) * hNa[0]

        # Kd
        alpha = 0.0047 * (V_old[0] - 8) / (1 - np.exp(-(V_old[0] - 8) / 12))
        beta = np.exp(-(V_old[0] + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (V_old[0] + 12) / (1 - np.exp(-(V_old[0] + 12) / 12))
        beta = np.exp(-(V_old[0] + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        nKd[0] = vinf + (nKd_old[0] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        gKd[0] = g_mem_Kd_comp * np.power(nKd[0], 4)

        # leak
        gleak[0] = g_mem_leak_comp 

        # integration parameters
        A[0] = (g_main_b1+g_main_b2)/C_mem_comp #in units of uS/nF
        B[0] = -(gNa[0]+gKd[0]+gleak[0]+gGABA[0]+g_ax_comp+g_main_b1+g_main_b2)/C_mem_comp #in units of uS/nF
        C[0] = g_ax_comp/C_mem_comp #in units of uS/nF
        D[0] = (gNa[0]*E_Na+gKd[0]*E_K+gleak[0]*E_leak+gGABA[0]*E_GABA)/C_mem_comp #in units of nA/nF
        a[0] = A[0]*dt #in units of uS*ms/nF
        b[0] = B[0]*dt #in units of uS*ms/nF
        c[0] = C[0]*dt #in units of uS*ms/nF              
        d[0] = (D[0]+(g_main_b1*b1_V[b1_n_comp-1]+g_main_b2*b2_V[b2_n_comp-1])/C_mem_comp+B[0]*V[0]+C[0]*V[1])*dt #in units of nA*ms/nF

        b_p[0] = b[0] # _p stands for prime as in Dayan and Abbott appendix 6
        d_p[0] = d[0]
        
        for i in range(1, n_comp-1, 1): #middle compartments
            # Na
            alpha = 0.36 * (V_old[i] + 33) / (1 - np.exp(-(V_old[i] + 33) / 3))
            beta = - 0.4 * (V_old[i] + 42) / (1 - np.exp((V_old[i] + 42) / 20))
            vinf = alpha / (alpha + beta)
            tau = 2 / (alpha + beta)
            tau = tau * taufac
            mNa[i] = vinf + (mNa_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
            alpha = - 0.1 * (V_old[i] + 55) / (1 - np.exp((V_old[i] + 55) / 6))
            beta = 4.5 / (1 + np.exp(-V_old[i] / 10))
            vinf = alpha / (alpha + beta)
            tau = 2 / (alpha + beta)
            tau = tau * taufac
            hNa[i] = vinf + (hNa_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

            gNa[i] = g_mem_Na_comp * np.power(mNa[i], 2) * hNa[i]

            # Kd
            alpha = 0.0047 * (V_old[i] - 8) / (1 - np.exp(-(V_old[i] - 8) / 12))
            beta = np.exp(-(V_old[i] + 127) / 30)
            vinf = alpha / (alpha + beta)
            alpha = 0.0047 * (V_old[i] + 12) / (1 - np.exp(-(V_old[i] + 12) / 12))
            beta = np.exp(-(V_old[i] + 147) / 30)
            tau = 1 / (alpha + beta)
            tau = tau * taufac
            nKd[i] = vinf + (nKd_old[i] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

            gKd[i] = g_mem_Kd_comp * np.power(nKd[i], 4)

            # leak
            gleak[i] = g_mem_leak_comp 
            
            # integration parameters
            A[i] = g_ax_comp/C_mem_comp #in units of uS/nF
            B[i] = -(gNa[i]+gKd[i]+gleak[i]+gGABA[i]+2.0*g_ax_comp)/C_mem_comp
            C[i] = g_ax_comp/C_mem_comp #in units of uS/nF
            D[i] = (gNa[i]*E_Na+gKd[i]*E_K+gleak[i]*E_leak+gGABA[i]*E_GABA)/C_mem_comp
            a[i] = A[i]*dt    
            b[i] = B[i]*dt
            c[i] = C[i]*dt
            d[i] = (D[i]+A[i]*V[i-1]+B[i]*V[i]+C[i]*V[i+1])*dt
            
            b_p[i] = b[i] + a[i]*c[i-1]/(1-b_p[i-1]) #equation 6.54 in D&A
            d_p[i] = d[i] + a[i]*d_p[i-1]/(1-b_p[i-1]) #equation 6.55 in D&A
        
        #compartment n_comp-1
        # Na
        alpha = 0.36 * (V_old[n_comp-1] + 33) / (1 - np.exp(-(V_old[n_comp-1] + 33) / 3))
        beta = - 0.4 * (V_old[n_comp-1] + 42) / (1 - np.exp((V_old[n_comp-1] + 42) / 20))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        mNa[n_comp-1] = vinf + (mNa_old[n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf
  
        alpha = - 0.1 * (V_old[n_comp-1] + 55) / (1 - np.exp((V_old[n_comp-1] + 55) / 6))
        beta = 4.5 / (1 + np.exp(-V_old[n_comp-1] / 10))
        vinf = alpha / (alpha + beta)
        tau = 2 / (alpha + beta)
        tau = tau * taufac
        hNa[n_comp-1] = vinf + (hNa_old[n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        gNa[n_comp-1] = g_mem_Na_comp * np.power(mNa[n_comp-1], 2) * hNa[n_comp-1]

        # Kd
        alpha = 0.0047 * (V_old[n_comp-1] - 8) / (1 - np.exp(-(V_old[n_comp-1] - 8) / 12))
        beta = np.exp(-(V_old[n_comp-1] + 127) / 30)
        vinf = alpha / (alpha + beta)
        alpha = 0.0047 * (V_old[n_comp-1] + 12) / (1 - np.exp(-(V_old[n_comp-1] + 12) / 12))
        beta = np.exp(-(V_old[n_comp-1] + 147) / 30)
        tau = 1 / (alpha + beta)
        tau = tau * taufac
        nKd[n_comp-1] = vinf + (nKd_old[n_comp-1] - vinf) * np.exp(-dt / tau) if dt < tau else vinf

        gKd[n_comp-1] = g_mem_Kd_comp * np.power(nKd[n_comp-1], 4)

        # leak
        gleak[n_comp-1] = g_mem_leak_comp 

        # integration parameters
        A[n_comp-1] = g_ax_comp/C_mem_comp #in units of uS/nF
        B[n_comp-1] = -(gNa[n_comp-1]+gKd[n_comp-1]+gleak[n_comp-1]+gGABA[n_comp-1]+g_ax_comp)/C_mem_comp
        C[n_comp-1] = 0.0 #in units of uS/nF
        D[n_comp-1] = (gNa[n_comp-1]*E_Na+gKd[n_comp-1]*E_K+gleak[n_comp-1]*E_leak+gGABA[n_comp-1]*E_GABA+I_stim_now)/C_mem_comp
        a[n_comp-1] = A[n_comp-1]*dt    
        b[n_comp-1] = B[n_comp-1]*dt
        c[n_comp-1] = C[n_comp-1]*dt
        d[n_comp-1] = (D[n_comp-1]+A[n_comp-1]*V[n_comp-2]+B[n_comp-1]*V[n_comp-1])*dt
            
        b_p[n_comp-1] = b[n_comp-1] + a[n_comp-1]*c[n_comp-2]/(1-b_p[n_comp-2]) #equation 6.54 in D&A
        d_p[n_comp-1] = d[n_comp-1] + a[n_comp-1]*d_p[n_comp-2]/(1-b_p[n_comp-2]) #equation 6.55 in D&A
       
        deltaV[n_comp-1] = d_p[n_comp-1]/(1-b_p[n_comp-1]) #equation 6.56 in D&A, update voltage for last compartment
        V[n_comp-1] = V_old[n_comp-1] + deltaV[n_comp-1]
                
        for i in range(n_comp-1, 0, -1): # step through the middle compartments backward to update voltages
            deltaV[i-1] = (c[i-1]*deltaV[i]+d_p[i-1])/(1-b_p[i-1])
            V[i-1] = V_old[i-1] + deltaV[i-1]

        # branch 1     
        b1_deltaV[b1_n_comp-1] = b1_d_p[b1_n_comp-1]/(1-b1_b_p[b1_n_comp-1]) #equation 6.56 in D&A, update voltage for last compartment
        b1_V[b1_n_comp-1] = b1_V_old[b1_n_comp-1] + b1_deltaV[b1_n_comp-1]
                
        for i in range(b1_n_comp-1, 0, -1): # step through the middle compartments backward to update voltages
            b1_deltaV[i-1] = (b1_c[i-1]*b1_deltaV[i]+b1_d_p[i-1])/(1-b1_b_p[i-1])
            b1_V[i-1] = b1_V_old[i-1] + b1_deltaV[i-1]       

        # branch 2            
        b2_deltaV[b2_n_comp-1] = b2_d_p[b2_n_comp-1]/(1-b2_b_p[b2_n_comp-1]) #equation 6.56 in D&A, update voltage for last compartment
        b2_V[b2_n_comp-1] = b2_V_old[b2_n_comp-1] + b2_deltaV[b2_n_comp-1]
                
        for i in range(b2_n_comp-1, 0, -1): # step through the middle compartments backward to update voltages
            b2_deltaV[i-1] = (b2_c[i-1]*b2_deltaV[i]+b2_d_p[i-1])/(1-b2_b_p[i-1])
            b2_V[i-1] = b2_V_old[i-1] + b2_deltaV[i-1]
           

        ###branch compartments n-1 updated based on main axon compartment 0 voltage            
        
        testsignal1.append(d_p[2])
        testsignal2.append(b_p[2])
        
        I_stim.append(I_stim_now) #record stimulus current
        
        V_rec_first.append(V[n_comp-1]) #record voltage in stimulated compartment
        V_rec_second.append(V[n_comp-2]) #record voltage in compartment next to stimulus
        V_rec_middle.append(V[mid_ind]) #record voltage in middle compartment
        V_rec_last.append(V[0]) #record voltage in branch compartment
        V_rec_nexttoGABA.append(V[2]) #recorded voltage time course in compartment next to GABA in mV
        
        b1_V_rec_first.append(b1_V[b1_n_comp-1]) #record voltage in first compartment
        b1_V_rec_second.append(b1_V[b1_n_comp-2]) #record voltage in second compartment
        b1_V_rec_middle.append(b1_V[b1_mid_ind]) #record voltage in middle compartment
        b1_V_rec_last.append(b1_V[0]) #record voltage in last compartment    
        b1_V_rec_nexttoGABA.append(b1_V[b1_n_comp-3]) #recorded voltage time course in compartment next to GABA in mV

        b2_V_rec_first.append(b2_V[b2_n_comp-1]) #record voltage in first compartment
        b2_V_rec_second.append(b2_V[b2_n_comp-2]) #record voltage in second compartment
        b2_V_rec_middle.append(b2_V[b2_mid_ind]) #record voltage in middle compartment
        b2_V_rec_last.append(b2_V[0]) #record voltage in last compartment
        b2_V_rec_nexttoGABA.append(b2_V[b2_n_comp-3]) #recorded voltage time course in compartment next to GABA in mV

        
# END SIMULATION

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

plotI_stim(t, I_stim, 'Main', t_total) #Comment out if you don't want to output the graph
plotVfirst(t, V_rec_first, 'Main', t_total) #Comment out if you don't want to output the graph
#plotVsecond(t, V_rec_second, 'Main', t_total) #Comment out if you don't want to output the graph
plotVmiddle(t, V_rec_middle, 'Main', t_total) #Comment out if you don't want to output the graph
#plotVnexttoGABA(t, V_rec_nexttoGABA, 'Main', t_total) #Comment out if you don't want to output the graph
plotVlast(t, V_rec_last, 'Main', t_total) #Comment out if you don't want to output the graph

plotVfirst(t, b1_V_rec_first, 'b1', t_total) #Comment out if you don't want to output the graph
#plotVsecond(t, b1_V_rec_second, 'b1', t_total) #Comment out if you don't want to output the graph
#plotVnexttoGABA(t, b1_V_rec_nexttoGABA, 'Main', t_total) #Comment out if you don't want to output the graph
plotVmiddle(t, b1_V_rec_middle, 'b1', t_total) #Comment out if you don't want to output the graph
plotVlast(t, b1_V_rec_last, 'b1', t_total) #Comment out if you don't want to output the graph

plotVfirst(t, b2_V_rec_first, 'b2', t_total) #Comment out if you don't want to output the graph
#plotVsecond(t, b2_V_rec_second, 'b2', t_total) #Comment out if you don't want to output the graph
#plotVnexttoGABA(t, b2_V_rec_nexttoGABA, 'Main', t_total) #Comment out if you don't want to output the graph
plotVmiddle(t, b2_V_rec_middle, 'b2', t_total) #Comment out if you don't want to output the graph
plotVlast(t, b2_V_rec_last, 'b2', t_total) #Comment out if you don't want to output the graph
print("Your simulations are successfully completed!") 

# writing data to CSV file
np.savetxt('Temp_secondtry_39_6.txt', np.column_stack([t,V_rec_second,V_rec_middle,V_rec_nexttoGABA,V_rec_last,b1_V_rec_first,b1_V_rec_nexttoGABA,b1_V_rec_middle,b1_V_rec_last,b2_V_rec_first,b2_V_rec_nexttoGABA,b2_V_rec_middle,b2_V_rec_last]))

