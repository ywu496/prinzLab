#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:33:40 2022

@author: yuxuanwu
"""



class segmentSpikeTest:
    #to avoid too much parameter inputs, use default values in the parameter configuration
    def __init__(self, spikeThreshold):
        self.Seg_inSpike = False
        self.spikeThreshold = spikeThreshold 
        self.Seg_spike_num = 0
        
    def findSpike(self, Seg_V):
        if(Seg_V >= self.spikeThreshold):
            if not self.Seg_inSpike:
                self.Seg_inSpike = True
                self.Seg_spike_num += 1
        else: 
            if self.Seg_inSpike:
                self.Seg_inSpike = False








    



  


            
        
