# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:33:34 2020

@author: Yorick Vos en Naomi Spier

Description
"""


#import standard modules
import numpy as np
import matplotlib.pyplot as plt
import fplsqlib as fp

# Aantal metingen
n = 2

# Meetwaarden
E = np.linspace(20.0,20.0, n)
Urb = np.array([0.20, 0.40])
I = np.array([0.01, 0.80])

# Formules
Uri = E - Urb                       # Spanning over Ri
Ri = (E - Urb) / I                  # Formule voor berekenen Ri

# Fouten in volt- en amperemeter (metrahit extra) 
iV = 0.0000009
iI = 0.0000010
dUrb = Urb * 0.005 + iV             # Fout in voltagemeting
dI = I * 0.01 + iI                  # Fout in stroommeting

# Externe fout functiegenerator zonder belasting
dE = np.linspace(0.01, 0.01, n)

dU = np.sqrt(dE**2 + dUrb**2)       # Doorgerekende fout in spanning E - Urb

# Externe fout Ri
dRi = np.mean((Ri * np.sqrt((dU/Uri)**2 + (dI/I)**2)))

# Plot van I tegen U met fplsqlib met een fit
fp.fplsqGUI(I, Uri, yerr=dU, xlabel='Stroom I [A]', ylabel='Voltage U [V]')
# De a van de fit is Ri
Ri = 50
# De t95-value van de fit is
t95_value = 2.77

