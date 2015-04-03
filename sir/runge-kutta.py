# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Infectious Pulse
# https://github.com/alvason/infectious-pulse
# 
# ### Effectiveness of Runge-Kutta algorithm for solving coupled differential equations 

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
plt.text(0, 1.0/6, r'$ f_3 = f(y_n + f_2 \frac{\Delta x}{2},\ x_n + \frac{\Delta x}{2}) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 0.0/6, r'$ f_4 = f(y_n + f_3 \Delta x,\ x_n + \Delta x) $', fontsize = 1.2*AlvaFontSize)
plt.show()

# <codecell>

# define RK4 for one diferrential equation
def AlvaRungeKutta4Yyx(dYdx, gridY, gridX, dx, xn):
    yyy = gridY[xn]; xxx = gridX[xn]; # keep initial value
    dydx1 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx1*dx/2; gridX[xn] = xxx + dx/2;
    dydx2 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx2*dx/2; gridX[xn] = xxx + dx/2;
    dydx3 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx3*dx; gridX[xn] = xxx + dx;
    dydx4 = dYdx(gridY[xn], gridX[xn]); 
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6); 
    gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridY[xn + 1]);

# Effectiveness of the algorithm
def dYdx(y, x):
    dY_dx = -x/y
    return dY_dx

def Y_function(x):
    Y = (1.0 - x**2)**(1.0/2) # circle
    return Y

# numerical griding
totalGPoint_X = int(18**1 + 1);
minX = float(-0.9); maxX = float(0.9);
spacingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = spacingX[0]
dx = spacingX[1]

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
    AlvaRungeKutta4Yyx(dYdx, gridY, gridX, dx, xn)

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridY_A, label = r'$ Analytic $', linewidth = 6.0, alpha = 0.3)
plt.plot(gridX, gridY_E, label = r'$ Euler $', marker = '^')
plt.plot(gridX, gridY, label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Effectiveness \ of \ Runge-Kutta-one \ (h = %f) $'%(dx)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 0));
plt.text(maxX*1.2, 1, r'$ \frac{\partial y(x)}{\partial x} = \frac{-x}{y} = \frac{-x}{(1 - x^2)^{1/2}} $'
         , fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, 4.0/5, r'$ y = (1 - x^2)^{1/2} $', fontsize = 1.2*AlvaFontSize);
plt.show();

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridY - gridY_A, label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Local \ error \ of \ Runge-Kutta-4th-one (\O(h^5) = %f) $'%(dx**5)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y - y_{exact} $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 0));
plt.show();

# <codecell>

# define RK4 for one diferrential equation
def AlvaRungeKutta4Yyx(dYdx, gridY, gridX, dx, xn):
    yyy = gridY[xn]; xxx = gridX[xn]; # keep initial value
    dydx1 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx1*dx/2; gridX[xn] = xxx + dx/2;
    dydx2 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx2*dx/2; gridX[xn] = xxx + dx/2;
    dydx3 = dYdx(gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx3*dx; gridX[xn] = xxx + dx;
    dydx4 = dYdx(gridY[xn], gridX[xn]); 
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6); 
    gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridY[xn + 1]);

# Effectiveness of the algorithm
def dYdx(y, x):
    dY_dx = np.cos(x)
    return dY_dx

def Y_function(x):
    Y = np.sin(x)
    return Y

# numerical griding
totalGPoint_X = int(40 + 1);
minX = float(0); maxX = float(4*np.pi);
spacingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = spacingX[0]
dx = spacingX[1]

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
    AlvaRungeKutta4Yyx(dYdx, gridY, gridX, dx, xn)

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridY_A, label = r'$ Analytic $', linewidth = 6.0, alpha = 0.3)
plt.plot(gridX, gridY_E, label = r'$ Euler $', marker = '^')
plt.plot(gridX, gridY, label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Effectiveness \ of \ Runge-Kutta-4th-one \ (h = %f) $'%(dx)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 0));
plt.text(maxX*1.2, 3.0/5, r'$ \frac{\partial y(x)}{\partial x} = cos(x)} $'
         , fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, 1.0/5, r'$ y = sin(x) $', fontsize = 1.2*AlvaFontSize);
plt.show();

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridY - gridY_A, label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Local \ error \ of \ Runge-Kutta-4th-one (\O(h^5) = %f) $'%(dx**5)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y - y_{exact} $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 0));
plt.show();

# <codecell>

# define RK4 for two coupled differential equations
def AlvaRungeKutta4ZYx(dZdx, dYdx, gridZ, gridY, gridX, dx, xn):
    zzz = gridZ[xn] ; yyy = gridY[xn]; xxx = gridX[xn]; # keep initial value
    dzdx1 = dZdx(gridZ[xn], gridY[xn], gridX[xn]); 
    dydx1 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); 
    gridZ[xn] = zzz + dzdx1*dx/2; gridY[xn] = yyy + dydx1*dx/2; gridX[xn] = xxx + dx/2;
    dzdx2 = dZdx(gridZ[xn], gridY[xn], gridX[xn]);
    dydx2 = dYdx(gridZ[xn], gridY[xn], gridX[xn]);
    gridZ[xn] = zzz + dzdx2*dx/2; gridY[xn] = yyy + dydx2*dx/2; gridX[xn] = xxx + dx/2;
    dzdx3 = dZdx(gridZ[xn], gridY[xn], gridX[xn]);
    dydx3 = dYdx(gridZ[xn], gridY[xn], gridX[xn]);
    gridZ[xn] = zzz + dzdx3*dx; gridY[xn] = yyy + dydx3*dx; gridX[xn] = xxx + dx;
    dzdx4 = dZdx(gridZ[xn], gridY[xn], gridX[xn]); 
    dydx4 = dYdx(gridZ[xn], gridY[xn], gridX[xn]);  
    gridZ[xn + 1] = zzz + dx*(dzdx1/6 + dzdx2/3 + dzdx3/3 + dzdx4/6); 
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6);
    gridZ[xn] = zzz; gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridZ[xn + 1], gridY[xn + 1]);

# Effectiveness of the algorithm
def dZdx(z, y, x):
    dZ_dx = -y
    return dZ_dx

def dYdx(z, y, x):
    dY_dx = z
    return dY_dx
 
def Z_function(x):
    Z = np.cos(x) # circle
    return Z

def Y_function(x):
    Y = np.sin(x) # circle
    return Y

# numerical griding
totalGPoint_X = int(20**1 + 1);
minX = float(0); maxX = float(2*np.pi);
spacingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = spacingX[0]
dx = spacingX[1]


# Analytic solution
gridZ_A = np.zeros(totalGPoint_X)
gridY_A = np.zeros(totalGPoint_X)
for xn in range(totalGPoint_X):
    gridZ_A[xn] = Z_function(gridX[xn])
    gridY_A[xn] = Y_function(gridX[xn])

# Euler solution  
gridZ_E = np.zeros(totalGPoint_X)
gridZ_E[0] = gridZ_A[0]
gridY_E = np.zeros(totalGPoint_X)
gridY_E[0] = gridY_A[0]
for xn in range(totalGPoint_X - 1):
    gridZ_E[xn + 1] = gridZ_E[xn] + dx*dZdx(gridZ_E[xn], gridY_E[xn], gridX[xn])
    gridY_E[xn + 1] = gridY_E[xn] + dx*dYdx(gridZ_E[xn], gridY_E[xn], gridX[xn])

# RungeKutta solution
gridZ = np.zeros(totalGPoint_X)
gridY = np.zeros(totalGPoint_X)
gridZ[0] = gridZ_A[0];
gridY[0] = gridY_A[0];
for xn in range(totalGPoint_X - 1):
    AlvaRungeKutta4ZYx(dZdx, dYdx, gridZ, gridY, gridX, dx, xn)

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridZ_A, label = r'$ Analytic $', linewidth = 6.0, alpha = 0.3)
plt.plot(gridX, gridY_A, label = r'$ Analytic $', linewidth = 6.0, alpha = 0.3)
plt.plot(gridX, gridZ_E, label = r'$ Euler $', marker = '^')
plt.plot(gridX, gridY_E, label = r'$ Euler $', marker = '^')
plt.plot(gridX, gridZ, label = r'$ RungeKutta $', marker = 'o')
plt.plot(gridX, gridY, label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Effectiveness \ of \ Runge-Kutta-4th-coupled \ (h = %f) $'%(dx)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 1));
plt.text(maxX*1.2, 6.0/6, r'$ \frac{\partial z(x)}{\partial x} = -y $'
         , fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, 1.0/6, r'$ \frac{\partial y(x)}{\partial x} = z $'
         , fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, -3.0/6, r'$ z = cos(x) $', fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, -6.0/6, r'$ y = sin(x) $', fontsize = 1.2*AlvaFontSize);
plt.show()

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
plt.plot(gridX, gridZ - gridZ_A, label = r'$ z(x) $', marker = '^')
plt.plot(gridX, gridY - gridY_A, label = r'$ y(x) $', marker = 'o')
plt.grid(True)
plt.title(r'$ Local \ error \ of \ Runge-Kutta-4th-coupled \ (\O(h^5) = %f) $'%(dx**5)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y - y_{exact} $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 0));
plt.show();

# <codecell>

# define RK4 for a list of coupled differential equations
def AlvaRungeKutta4List(dYdx_array, gridY_array, gridX, dx, xn):
    m = dYdx_array.size # size of a list of equation
    # keep initial value
    yyy_array = np.copy(gridY_array[:, xn])
    xxx = gridX[xn]
    # first step
    dydx1_array = np.zeros(m)
    gridYX_array = np.append(gridY_array[:, xn], gridX[xn]) # reunion with new values
    for i in range(m):
        dydx1_array[i] = dYdx_array[i](gridYX_array) # computing ratio
        gridY_array[i, xn] = yyy_array[i] + dydx1_array[i]*dx/2 # computing new high
    gridX[xn] = xxx + dx/2
    # second half step
    dydx2_array = np.zeros(m)
    gridYX_array = np.append(gridY_array[:, xn], gridX[xn]) # reunion with new values
    for i in range(m):
        dydx2_array[i] = dYdx_array[i](gridYX_array) 
        gridY_array[i, xn] = yyy_array[i] + dydx2_array[i]*dx/2 
    gridX[xn] = xxx + dx/2
    # third half step
    dydx3_array = np.zeros(m)
    gridYX_array = np.append(gridY_array[:, xn], gridX[xn]) # reunion with new values
    for i in range(m):
        dydx3_array[i] = dYdx_array[i](gridYX_array)
        gridY_array[i, xn] = yyy_array[i] + dydx3_array[i]*dx
    gridX[xn] = xxx + dx
    # final step
    dydx4_array = np.zeros(m)
    gridYX_array = np.append(gridY_array[:, xn], gridX[xn]) # reunion with new values
    for i in range(m):
        dydx4_array[i] = dYdx_array[i](gridYX_array)
    # computing the next move by accumulate all the steps
    for i in range(m):
        gridY_array[i, xn + 1] = yyy_array[i] + dx*(dydx1_array[i]/6 + dydx2_array[i]/3 + dydx3_array[i]/3 + dydx4_array[i]/6)
    # restore to initial value
    gridY_array[:, xn] = np.copy(yyy_array)
    gridX[xn] = xxx; 
    return (gridY_array[:, xn + 1]);

# Effectiveness of the algorithm
def dZdx(zyx = [], *args):
    dZ_dx = -zyx[1]
    return dZ_dx

def dYdx(zyx = [], *args):
    dY_dx = zyx[0]
    return dY_dx
 
def Z_function(x):
    Z = np.cos(x) 
    return Z

def Y_function(x):
    Y = np.sin(x) 
    return Y

# numerical griding
totalGPoint_X = int(10**2 + 1);
minX = float(0); maxX = float(10*np.pi);
spacingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = spacingX[0]
dx = spacingX[1]

# Analytic solution
gridZ_A = np.zeros(totalGPoint_X)
gridY_A = np.zeros(totalGPoint_X)
for xn in range(totalGPoint_X):
    gridZ_A[xn] = Z_function(gridX[xn])
    gridY_A[xn] = Y_function(gridX[xn])

# RungeKutta solution
gridZ = np.zeros(totalGPoint_X)
gridY = np.zeros(totalGPoint_X)
gridZ[0] = gridZ_A[0];
gridY[0] = gridY_A[0];
dYdx_array = np.array([dZdx, dYdx])
gridY_array = np.array([gridZ, gridY])
for xn in range(totalGPoint_X - 1):
    AlvaRungeKutta4List(dYdx_array, gridY_array, gridX, dx, xn)

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (16, 5))
plt.plot(gridX, gridZ_A, label = r'$ Analytic $', linewidth = 6.0, alpha = 0.3)
plt.plot(gridX, gridY_A, label = r'$ Analytic $', linewidth = 6.0, alpha = 0.3)

plt.plot(gridX, gridY_array[0], label = r'$ RungeKutta $', marker = 'o')
plt.plot(gridX, gridY_array[1], label = r'$ RungeKutta $', marker = 'o')
plt.grid(True)
plt.title(r'$ Effectiveness \ of \ Runge-Kutta-4th-coupled \ (\O(h) = %f) $'%(dx)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 1));
plt.text(maxX*1.2, 4.0/6, r'$ \frac{\partial z(x)}{\partial x} = -y $'
         , fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, 1.0/6, r'$ \frac{\partial y(x)}{\partial x} = z $'
         , fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, -2.0/6, r'$ z = cos(x) $', fontsize = 1.2*AlvaFontSize);
plt.text(maxX*1.2, -4.0/6, r'$ y = sin(x) $', fontsize = 1.2*AlvaFontSize);
plt.show() 

# plotting
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (16, 5))
plt.plot(gridX, gridY_array[0] - gridZ_A, label = r'$ z(x) $', marker = '^')
plt.plot(gridX, gridY_array[1] - gridY_A, label = r'$ y(x) $', marker = 'o')
plt.grid(True)
plt.title(r'$ Local \ error \ of \ Runge-Kutta-4th-coupled \ (\O(h^5) = %f) $'%(dx**5)
          , fontsize = AlvaFontSize);
plt.xlabel(r'$ x $', fontsize = AlvaFontSize);
plt.ylabel(r'$ y - y_{exact} $', fontsize = AlvaFontSize);
plt.legend(loc = (1, 0));
plt.show();

# <codecell>


