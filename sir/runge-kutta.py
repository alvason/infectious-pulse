# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
author: Alvason Zhenhua Li
date:   03/23/2015
'''

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import time
import IPython.display as idisplay
from mpl_toolkits.mplot3d.axes3d import Axes3D

AlvaFontSize = 23;
AlvaFigSize = (16, 6);
numberingFig = 0;

# Runge-Kutta numerical algorithm 
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12, 6))
plt.axis('off')
plt.title(r'$ the \ 4th-order \ Runge-Kutta \ algorithm (interpolating \ polynomial) $', fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ \frac{\partial y}{\partial x} = f(y,x) $', fontsize = 1.2*AlvaFontSize)

plt.text(0, 4.0/6, r'$ y_{n+1} = \
         y_n + \Delta x(\frac{ f_1}{6} + \frac{f_2}{3} + \frac{f_3}{3} +\frac{f_4}{6})  $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/6, r'$ f_1 = f(y_n,\ x_n) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/6, r'$ f_2 = f(y_n + f_1 \frac{\Delta x}{2},\ x_n + \frac{\Delta x}{2}) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 1.0/6, r'$ f_3= f(y_n + f_2 \frac{\Delta x}{2},\ x_n + \frac{\Delta x}{2}) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 0.0/6, r'$ f_4 = f(y_n + f_3 \Delta x,\ x_n + \Delta x) $', fontsize = 1.2*AlvaFontSize)
plt.show()

# <codecell>

def AlvaRungeKutta4Yx(dYdx, gridY, gridX, dx, xn):
    yyy = gridY[xn]; xxx = gridX[xn]; # keep initial value
    dydx1 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx1*dx/2; gridX[xn] = xxx + dx/2;
    dydx2 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx2*dx/2; gridX[xn] = xxx + dx/2;
    dydx3 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx3*dx; gridX[xn] = xxx + dx;
    dydx4 = dYdx(gridY[xn], gridX[xn]); 
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6); 
    gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridY[xn + 1]);

# Effectiveness of the algorithm
def Y_function(x):
    Y = (1.0 - x**2)**(1.0/2) # circle
    return Y

def dYdx(y, x):
    dY_dx = -x/(1.0 - x**2)**(1.0/2)
    return dY_dx

# numerical griding
totalGPoint_X = int(18**1 + 1);
minX = float(-0.9); maxX = float(0.9);
gridX = np.linspace(minX, maxX, totalGPoint_X);
dx = (maxX - minX)/totalGPoint_X;

# Analytic solution
gridY_A = np.zeros(totalGPoint_X)
for xn in range(totalGPoint_X):
    gridY_A[xn] = Y_function(gridX[xn])

# Euler solution  
gridY_E = np.zeros(totalGPoint_X)
gridY_E[0] = gridY_A[0]
for xn in range(totalGPoint_X - 1):
    gridY_E[xn + 1] = gridY_E[xn] + dx*dYdx(gridY_E[xn], gridX[xn])

# RungeKutta solution
gridY = np.zeros(totalGPoint_X)
gridY[0] = gridY_A[0];
for xn in range(totalGPoint_X - 1):
    AlvaRungeKutta4Yx(dYdx, gridY, gridX, dx, xn)
   
gridX = np.linspace(minX, maxX, totalGPoint_X);      
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridY_A, label = r'$ Analytic $')
plt.plot(gridX, gridY_E, label = r'$ Euler $', marker = '^')
plt.plot(gridX, gridY, label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Effectiveness \ of \ Runge-Kutta \ (total \ steps = %i) $'%(totalGPoint_X)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

# <codecell>

def AlvaRungeKutta4(dYdx, gridZ, gridY, gridX, dx, xn):
    zzz = gridZ[xn] ; yyy = gridY[xn]; xxx = gridX[xn]; # keep initial value
    dydx1 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx1*dx/2; gridX[xn] = xxx + dx/2;
    dydx2 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx2*dx/2; gridX[xn] = xxx + dx/2;
    dydx3 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx3*dx; gridX[xn] = xxx + dx;
    dydx4 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); 
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6); 
    gridZ[xn] = zzz; gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridY[xn + 1]);

# <codecell>


