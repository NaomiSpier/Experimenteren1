# -*- coding: utf-8 -*-

# meetwaardes
n = float(1)
l = 532e-9      # golflengte meter
x = 0.5         # afstand meter
y = 0.445         # afstand meter

# fouten
dl = 10e-9      # golflengte meter
dx = 5e-3       # afstand meter
dy = 5e-3       # afstand meter

# berekening d
d = n*l*(1 + (x/y)**2)**0.5

# berekening fout
dd = ( n**2*(1 + (x/y)**2)*dl**2 + ((n**2 * l**2)/(4*(1 + (x/y)**2)))*((dy**2 / y**6) + 4 * x**2 * dx**2) )**0.5


