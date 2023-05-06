#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 08:33:22 2023

@author: wuyuxuan
"""


import numpy as np


#axon c1 connects to axon c2 where c1 compartment 5 connects to c2 comparment 4
# c1 has a total compartment of 10, while c2 has a total compartment of 15
#originally thought about the whole process
#[(a[(value, v)], c[value, v])]
class Integration:
        
        
    def IntegParamInit(axo, ACsettingArray, dt, stimDict=None):
        axo.A = []
        axo.B = []
        axo.C = []
        axo.D = []
        axo.a = []
        axo.b = []
        axo.c = []
        axo.d = []
        
        for i in range(len(ACsettingArray)):
            temp_B = (-(axo.gNa[i]+axo.gKd[i]+axo.gleak[i]+axo.gGABA[i]+axo.gA[i] + axo.gH[i] + axo.gM[i] + axo.gCaL[i] + axo.gKCa[i])/axo.C_mem_comp[i])#in units of uS/nF

            A_condVpairArr, C_condVpairArr = ACsettingArray[i]
            temp_A = 0.0
            temp_d = 0.0

            for aPair in A_condVpairArr:
                conductance, comp_voltage = aPair
                temp_A += conductance / axo.C_mem_comp[i]
                temp_d += conductance / axo.C_mem_comp[i] * comp_voltage
                temp_B += (-conductance/axo.C_mem_comp[i])
                
                
            temp_C = 0.0
            for cPair in C_condVpairArr:
                conductance, comp_voltage = cPair
                temp_C += conductance / axo.C_mem_comp[i]
                temp_d += conductance / axo.C_mem_comp[i] * comp_voltage
                temp_B += (-conductance/axo.C_mem_comp[i])
                
            axo.A.append(temp_A)
            axo.B.append(temp_B)
            axo.C.append(temp_C)
            
            if stimDict == None:
                stim_temp = 0.0
            else:
                stim_temp = stimDict.get(i, 0.0)
                
            axo.D.append((axo.gNa[i]*axo.E_Na+axo.gKd[i]*axo.E_K+axo.gleak[i]*axo.E_leak+axo.gGABA[i]*axo.E_GABA + stim_temp
                     +axo.gA[i]*axo.E_A + axo.gH[i]*axo.E_H + axo.gM[i]*axo.E_M + axo.gCaL[i]*axo.E_CaL + axo.gKCa[i]*axo.E_KCa)/axo.C_mem_comp[i]) #in units of nA/nF
    
            axo.a.append(temp_A * dt)
            axo.b.append(temp_B * dt)
            axo.c.append(temp_C * dt)
        
            axo.d.append((temp_d + axo.D[i] + axo.B[i] * axo.V[i]) * dt)
            

        axo.A = np.asarray(axo.A)
        axo.B = np.asarray(axo.B)
        axo.C = np.asarray(axo.C)
        axo.D = np.asarray(axo.D)
        axo.a = np.asarray(axo.a)
        axo.b = np.asarray(axo.b)
        axo.c = np.asarray(axo.c)
        axo.d = np.asarray(axo.d)
        
        axo.b_p = axo.b.copy()
        axo.d_p = axo.d.copy()
    


                
        

        

