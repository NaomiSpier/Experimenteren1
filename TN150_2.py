# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:32:57 2020

@author: naomi
"""
import numpy as np
import fplsqlib as fp

R = np.array([1700,1800,1900,2000]) #placeholders
verschilSpanning = np.array([ 1, 2, 3, 4])  #placeholders

Urimpel = 0.5 * verschilSpanning
fout = Urimpel * 0.005 #placeholder

fp.fplsqGUI(R, Urimpel, yerr=fout, xlabel='Belasting(R)', ylabel='Urimpel(V)')

Ugelijk = np.array([19, 18, 17, 16]) #placeholder
foutGelijk = Ugelijk * 0.005 #placeholder

fp.fplsqGUI(R, Ugelijk, yerr=foutGelijk, xlabel='Belasting(R)', ylabel='Ugelijk(V)')


