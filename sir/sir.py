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


numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,3))
plt.axis('off')
plt.title(r'$ Susceptible-Infectious-Recovered \ equation $',fontsize = AlvaFontSize)
plt.text(0,2.0/3,r'$ \frac{\partial S(t)}{\partial t} = \
         -\beta S(t)I(t) +\mu N -\mu S(t)$', fontsize = 1.2*AlvaFontSize)
plt.text(0,1.0/3,r'$ \frac{\partial I(t)}{\partial t} = \
         +\beta S(t)I(t) - \gamma I(t) -\mu I(t) $', fontsize = 1.2*AlvaFontSize)
plt.text(0,0.0/3,r'$ \frac{\partial R(t)}{\partial t} = \
         +\gamma I(t) - \mu R(t) $', fontsize = 1.2*AlvaFontSize)

plt.show()

# <codecell>

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

def RungeKutta4(gridY, gridX, dYdx, dx, xn):
    yyy = gridY[xn]; 
    xxx = gridX[xn]; # keep initial value
    dydx1 = dYdx[xn]; print dYdx[xn]
    gridY[xn] = yyy + dydx1*dx/2; 
    gridX[xn] = xxx + dx/2;
    dydx2 = dYdx[xn]; print dYdx[xn]
    gridY[xn] = yyy + dydx2*dx/2; 
    gridX[xn] = xxx + dx/2;
    dydx3 = dYdx[xn]; print dYdx[xn]
    gridY[xn] = yyy + dydx3*dx; 
    gridX[xn] = xxx + dx;
    dydx4 = dYdx[xn]; print dYdx[xn]
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6); 
    gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridY[xn + 1]);


# <codecell>

# Effectiveness of the algorithm
def Y_function(x):
    Y = (1.0 - x**2)**(1.0/2) # circle
    Y1 = x*np.sin(x)
    return (Y + Y1)

def Y_derivative(y, x):
    dY_dx = -x/(1.0 - x**2)**(1.0/2)
    dY_dx1 = np.sin(x) + x*np.cos(x)
    return (dY_dx + dY_dx1)

# numerical griding
totalGPoint_X = int(10**1);
minX = float(-0.99); maxX = float(0.9);
gridX = np.linspace(minX, maxX, totalGPoint_X);
dx = (maxX - minX)/totalGPoint_X;

# Analytic solution
gridY_A = np.zeros(totalGPoint_X)
for xn in range(totalGPoint_X - 1):
    gridY_A[xn] = Y_function(gridX)[xn]

# Euler solution  
gridY_E = np.zeros(totalGPoint_X)
dYdx = Y_derivative(gridY_E, gridX)
gridY_E[0] = gridY_A[0]
for xn in range(totalGPoint_X - 1):
    gridY_E[xn + 1] = gridY_E[xn] + dx*dYdx[xn]

# RungeKutta solution
gridY = np.zeros(totalGPoint_X)
dYdx = Y_derivative(gridY, gridX)
gridY[0] = gridY_A[0];
for xn in range(totalGPoint_X - 1):
    RungeKutta4(gridY, gridX, dYdx, dx, xn)
    
    
rangeA = totalGPoint_X/1 - 60;
rangeB = totalGPoint_X/1 - 30;
gridX = np.linspace(minX, maxX, totalGPoint_X);    
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = (12, 6))
#plt.plot(gridX[rangeA:rangeB], gridY_A[rangeA:rangeB], label = r'$ Analytic $')
#plt.plot(gridX[rangeA:rangeB], gridY_E[rangeA:rangeB], label = r'$ Euler $')
#plt.plot(gridX[rangeA:rangeB], gridY[rangeA:rangeB], label = r'$ RungeKutta $')

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

# setting parameter
timeUnit = 'day'
if timeUnit == 'day':
    day = 1; year = 365; 
elif timeUnit == 'year':
    year = 1; day = float(1)/365; 
    
totalSIR = float(1); # total population
inOutRate = float(1)/(30*year) # birth rate per year
reprodNum = 1.8 # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
infecRate = reprodNum*recovRate/totalSIR # per year, per person, per total-population

# initial condition
minT = float(0); maxT = float(1*year);
totalGPoint_T = int(10**4);
gridT = np.linspace(minT, maxT, totalGPoint_T);
dt = (maxT - minT)/totalGPoint_T;

gridS = np.zeros(totalGPoint_T);
gridI = np.zeros(totalGPoint_T);
gridR = np.zeros(totalGPoint_T);

gridI[0] = float(1)/10**6;
gridR[0] = float(0);
gridS[0] = totalSIR - gridI[0] - gridR[0];

# Runge Kutta numerical scheme
for tn in range(totalGPoint_T - 1):
    dSdt = -infecRate*gridS*gridI + inOutRate*totalSIR - inOutRate*gridS;
    dIdt = +infecRate*gridS*gridI - recovRate*gridI - inOutRate*gridI;
    dRdt = +recovRate*gridI - inOutRate*gridR;
    RungeKutta4(gridS, gridT, dSdt, dt, tn);
    RungeKutta4(gridI, gridT, dIdt, dt, tn);
    RungeKutta4(gridR, gridT, dRdt, dt, tn);   
    
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI*day, label = r'$ \beta \ S(t)I(t) $', linestyle = 'dotted', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ Prevalence \ and \ incidence \ of \ SIR \ (R_0 = %f,\ \gamma = %f,\ \beta = %f) $'%(reprodNum, recovRate, infecRate), fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

# <codecell>

# setting parameter
timeUnit = 'year'
if timeUnit == 'day':
    day = 1; year = 365; 
elif timeUnit == 'year':
    year = 1; day = float(1)/365; 
    
totalSIR = float(1); # total population
inOutRate = float(1)/(30*year) # birth rate per year
reprodNum = 1.8 # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
infecRate = reprodNum*recovRate/totalSIR # per year, per person, per total-population

# initial condition
minT = float(0); maxT = float(100*year);
totalGPoint_T = int(10**4);
gridT = np.linspace(minT, maxT, totalGPoint_T);
dt = (maxT - minT)/totalGPoint_T;

gridS = np.zeros(totalGPoint_T);
gridI = np.zeros(totalGPoint_T);
gridR = np.zeros(totalGPoint_T);

gridI[0] = float(1)/10**6;
gridR[0] = float(0);
gridS[0] = totalSIR - gridI[0] - gridR[0];

# Runge Kutta numerical scheme
for tn in range(totalGPoint_T - 1):
    dSdt = -infecRate*gridS*gridI + inOutRate*totalSIR - inOutRate*gridS;
    dIdt = +infecRate*gridS*gridI - recovRate*gridI - inOutRate*gridI;
    dRdt = +recovRate*gridI - inOutRate*gridR;
    RungeKutta4(gridS, gridT, dSdt, dt, tn);
    RungeKutta4(gridI, gridT, dIdt, dt, tn);
    RungeKutta4(gridR, gridT, dRdt, dt, tn);   
    
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI*day, label = r'$ \beta S(t)I(t) $', linestyle = 'dotted', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ Revival \ of \ SIR \ (R_0 = %f,\ \gamma = %f,\ \beta = %f) $'%(reprodNum, recovRate, infecRate), fontsize = AlvaFontSize);
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

# <codecell>


