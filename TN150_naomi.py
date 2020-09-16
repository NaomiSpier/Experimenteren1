# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:10:44 2020

@author: naomi
"""

import numpy as np
import fplsqlib as fp

Uklem = np.array([19,18,17,16])             #placeholder data V
I = np.array([0.01,0.0095,0.009,0.0085])    #placeholder data I

dU = Uklem * 0.05   #gegevens uit schematics
dI = I * 0.05       #gegevens uit schematics

fout = Uklem * ((dU/Uklem)**2 + (dI/I)**2)**0.5 #absolute fout totaal

fp.fplsqGUI(I, Uklem, yerr=fout, xlabel='Stroom I [A]', ylabel='Voltage U [V]')