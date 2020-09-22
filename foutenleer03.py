# -*- coding: utf-8 -*-
"""
Ik kan beter coderen dan je moeder
"""

import numpy as np
import fplsqlib as fp

L = np.array([1,1.1,1.2,1.3])                 #placeholder data m
T = np.array([1,2,3,4])                      #placeholder data s

x = L**0.5

T_err = np.array([0.01,0.01,0.01,0.01])


fp.fplsqGUI(x, T, yerr=T_err, xlabel='wortel van L (m^0.5)', ylabel='Slingertijd (s)')