
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:34:56 2023

@author: wuyuxuan
"""
import numpy as np

class axBranch:

    #constructor, what varaibles to use
    def __init__(self, L, d, l_comp, R_ax, V_init, \
                 mNa_init, hNa_init, nKd_init, mA_init, hA_init, mH_init, mM_init, mCaL_init, hCaL_init, mKCa_init, CaS_init, \
                 G_leak_abs_uniform,E_leak_uniform, G_Na_abs_uniform, E_Na_uniform, G_Kd_abs_uniform, E_K_uniform, \
                 G_GABA_uniform, E_GABA_uniform, \
                 G_A_abs_uniform, E_A_uniform, G_H_abs_uniform, E_H_uniform, G_M_abs_uniform, E_M_uniform, \
                 G_CaL_abs_uniform, E_CaL_uniform, G_KCa_abs_uniform, E_KCa_uniform):

        self.L = L #length of axon in um, distance between sympathetic ganglia is about 1mm
        self.d = d #diameter of axon in um
        self.l_comp = l_comp #length of each compartment in um
        self.n_comp = int(self.L / self.l_comp) #number of compartments
       
        #self.comp_vol = np.pi * ((self.d/2) ** 2) * self.l_comp #um^3
        
        self.mid_ind = int(self.n_comp/2) #index of compartment roughly in middle of cable
       
        #self.A_mem_comp = np.pi*self.d*self.l_comp*1e-8 #membrane surface area of compartment in square cm
        
        #self.A_cross_comp = np.pi*self.d*self.d*1e-8/4 #axon cross-sectional in square cm
       
        #tapering implementations
        self.comp_vol = [np.pi * ((self.d[x]/2) ** 2) * self.l_comp for x in range(self.n_comp)]#um^3
        self.A_mem_comp = [ np.pi*self.d[x]*self.l_comp*1e-8 for x in range(self.n_comp)]#membrane surface area of compartment in square cm
        self.A_cross_comp = [ np.pi*self.d[x]*self.d[x]*1e-8/4 for x in range(self.n_comp)]#axon cross-sectional in square cm
        
        #self.C_mem_comp = self.A_mem_comp*1e3 #membrane capacitace of individual compartment in nF, assuming 1uF/cm2
        self.C_mem_comp = [self.A_mem_comp[x]*1e3 for x in range(self.n_comp)]
        #self.conductance_scaling_factor = 1e6*self.C_mem_comp/100 #factor used to scale conductances because McKinnon et al model has 100pF capacitance
        self.conductance_scaling_factor = [1e6*self.C_mem_comp[x]/100 for x in range(self.n_comp) ]

        # branch 1
        self.G_leak_abs = G_leak_abs_uniform #leak conductance in nS, based on McKinnon Table 1, default is 1nS
        self.g_mem_leak_comp = [self.conductance_scaling_factor[x]*self.G_leak_abs/1e3 for x in range(self.n_comp)] #membrane leak conductance per compartment in uS
        self.E_leak = E_leak_uniform #leak reversal potential in mV, according to McKinnon, default is -55mV
        # Na:
        self.G_Na_abs = G_Na_abs_uniform #Na conductance in nS, based on McKinnon Table 1, default is 300nS
        self.g_mem_Na_comp = [self.conductance_scaling_factor[x]*self.G_Na_abs/1e3 for x in range(self.n_comp)] #membrane Na conductance per compartment in uS
        self.E_Na = E_Na_uniform #Na reversal potential in mV, according to McKinnon, default is 60mV
        # Kd:
        self.G_Kd_abs = G_Kd_abs_uniform #Kd conductance in nS, based on McKinnon Table 1, default is 2000nS
        self.g_mem_Kd_comp = [self.conductance_scaling_factor[x]*self.G_Kd_abs/1e3 for x in range(self.n_comp)] #membrane Kd conductance per compartment in uS
        self.E_K = E_K_uniform #K reversal potential in mV, according to McKinnon, default is -90mV
        # GABA conductance for compartments proximal to (but not at) branch point
        self.G_GABA_abs = G_GABA_uniform #GABA conductance in nS
        self.g_mem_GABA = [self.conductance_scaling_factor[x]*self.G_GABA_abs/1e3 for x in range(self.n_comp)] #GABA conductance per compartment in uS
        self.E_GABA = E_GABA_uniform #GABA (i.e., Cl-) reversal potential in mV, see Prescott paper fig. 6, vary from -65mV (control) to -50mV (SCI)
        
        
        self.G_A_abs = G_A_abs_uniform #default 50 nS
        self.g_mem_A_comp = [self.conductance_scaling_factor[x]*self.G_A_abs/1e3 for x in range(self.n_comp)]
        self.E_A = E_A_uniform#  default of -90.0 in mV
        
        
        self.G_H_abs = G_H_abs_uniform #default 1nS
        self.g_mem_H_comp = [self.conductance_scaling_factor[x]*self.G_H_abs/1e3 for x in range(self.n_comp)]
        self.E_H = E_H_uniform   #  default of -32.0 in mV
        
        
        self.G_M_abs = G_M_abs_uniform #default 50 nS
        self.g_mem_M_comp = [self.conductance_scaling_factor[x]*self.G_M_abs/1e3 for x in range(self.n_comp)]
        self.E_M = E_M_uniform#  default of -90.0 in mV
        
        self.G_CaL_abs = G_CaL_abs_uniform #default 1.2 nS
        self.g_mem_CaL_comp = [self.conductance_scaling_factor[x]*self.G_CaL_abs/1e3 for x in range(self.n_comp)]
        self.E_CaL = E_CaL_uniform# default of 120.0 in mV
        ###########
        
        self.G_KCa_abs = G_KCa_abs_uniform #default 50 nS
        self.g_mem_KCa_comp = [self.conductance_scaling_factor[x]*self.G_KCa_abs/1e3 for x in range(self.n_comp)]
        self.E_KCa = E_KCa_uniform# at default already in mV
        
        
        self.R_ax = R_ax #axial resistivity in Ohm cm, from https://www.frontiersin.org/articles/10.3389/fncel.2019.00413/full
        self.g_ax_comp = [self.A_cross_comp[x]*1e6/(self.R_ax*self.l_comp*1e-4) for x in range(self.n_comp)] #axial conductance between compartments in uS
        
        
        self.g_cross_comp = [(0.0 , 2.0*self.g_ax_comp[0]*self.g_ax_comp[1]/(self.g_ax_comp[0]+self.g_ax_comp[1]))]
        for i in range(1, self.n_comp - 1):
            left = 2.0*self.g_ax_comp[i]*self.g_ax_comp[i-1]/(self.g_ax_comp[i]+self.g_ax_comp[i-1])
            right = 2.0*self.g_ax_comp[i]*self.g_ax_comp[i+1]/(self.g_ax_comp[i]+self.g_ax_comp[i+1])
            self.g_cross_comp.append((left, right))

        self.g_cross_comp.append((2.0*self.g_ax_comp[self.n_comp-1]*self.g_ax_comp[self.n_comp-2]/(self.g_ax_comp[self.n_comp-1]+self.g_ax_comp[self.n_comp-2]),0.0))
        #Branch b1
        
        # compartmental voltage changes
        self.deltaV = np.full((self.n_comp,), 0.0)
        #self.deltaV = [0.0] 
        
        #for i in range(1, self.n_comp, 1):
         #   self.deltaV.append(0.0)
        #self.deltaV = np.asarray(self.deltaV)
        
        # initial values for compartmental voltages, gating variables, conductances, and currents
        self.V_init = V_init #initialize all voltages
        #print(V_init)
        self.mNa_init = mNa_init #initialize all Na channels to deactivated
        self.hNa_init = hNa_init #initialize all Na channels to deinactivated
        self.nKd_init = nKd_init #initialize all Kd channels to deactivated
        
        
        self.mA_init = mA_init
        self.hA_init = hA_init
        self.mH_init = mH_init
        self.mM_init = mM_init
        self.mCaL_init = mCaL_init
        self.hCaL_init = hCaL_init
        self.mKCa_init = mKCa_init
        self.CaS_init = CaS_init
        
        
        self.gNa = [self.g_mem_Na_comp[x] * np.power(self.mNa_init, 2) * self.hNa_init for x in range(self.n_comp)]
        self.gKd = [self.g_mem_Kd_comp[x] * np.power(self.nKd_init, 4) for x in range(self.n_comp)]
        self.gleak = [self.g_mem_leak_comp[x] for x in range(self.n_comp)]
        
        
        self.gA = [self.g_mem_A_comp[x] * np.power(self.mA_init, 3) * self.hA_init for x in range(self.n_comp)]
        self.gH = [self.g_mem_H_comp[x] * self.mH_init for x in range(self.n_comp)]
        self.gM = [self.g_mem_M_comp[x] * np.power(self.mM_init, 2) for x in range(self.n_comp)]
        self.gCaL = [self.g_mem_CaL_comp[x] * self.mCaL_init * self.hCaL_init for x in range(self.n_comp)]
        self.gKCa = [self.g_mem_KCa_comp[x] * self.mKCa_init for x in range(self.n_comp)]
        
        
        self.gGABA = [] #array of GABA conductances, will be zeros except for compartments proximal to branch point

        
  
        
        #generate and fill in arrays of compartmental voltages, gating variables, and currents
        self.V = [] #array of compartment voltages in mV
        self.mNa = [] #array of Na activation variables
        self.hNa = [] #array of Na inactivation variables
        self.nKd = [] #array of Kd activation variables
        self.mA = []
        self.hA = []
        self.mH = []
        self.mM = []
        self.mCaL = []
        self.hCaL = []
        self.mKCa= []
        self.CaS = []
        

        for i in range(0, self.n_comp):
            self.V.append(self.V_init) #initialize compartment voltage array
            self.mNa.append(self.mNa_init) #initialize compartment Na activation array
            self.hNa.append(self.hNa_init) #initialize compartment Na inactivation array
            self.nKd.append(self.nKd_init) #initialize compartment Kd activation array
            
            
            
            self.mA.append(self.mA_init) #initialize compartment Kd activation array
            self.hA.append(self.hA_init) #initialize compartment Kd activation array
            self.mH.append(self.mH_init)
            self.mCaL.append(self.mCaL_init)
            self.hCaL.append(self.hCaL_init)
            self.mM.append(self.mM_init)
            self.mKCa.append(self.mKCa_init)
            self.CaS.append(self.CaS_init)
            
            self.gGABA.append(0.0)

            
        #this relates to conneciton and should be taken out right now
        #self.gGABA[self.n_comp-2] = self.g_mem_GABA #put GABA conductance only in compartment proximal to branch point (not at branch point)
        
        
        # store previous time step values in _old arrays
        


    #object related functions
    #??? look into class functions that are independent of an object but contained in a class    
    #??? look into field placement, instantiation, abstraction, inheritance, polymorphism, and more
    
