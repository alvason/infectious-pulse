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

# setting parameter
day = 1;
year = 365; # 365 days per year
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

gridS[0] = totalSIR;
gridI[0] = float(1)/10**4;
gridR[0] = float(0);

# simple numerical scheme
for tn in range(totalGPoint_T - 1):
    gridS[tn + 1] = gridS[tn] + dt*(-infecRate*gridS[tn]*gridI[tn] + inOutRate*(totalSIR - gridS[tn]));
    gridI[tn + 1] = gridI[tn] + dt*(+infecRate*gridS[tn]*gridI[tn] - recovRate*gridI[tn] - inOutRate*gridI[tn]);
    gridR[tn + 1] = gridR[tn] + dt*(+recovRate*gridI[tn] - inOutRate*gridR[tn]);
    
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI, label = r'$ \beta S(t)I(t) $', linestyle = 'dotted', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ SIR \ model \ (R_0 = %f,\ \gamma = %f,\ \beta = %f) $'%(reprodNum, recovRate, infecRate), fontsize = AlvaFontSize);
plt.xlabel(r'$time \ (day)$', fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

# <codecell>

# Runge-Kutta numerical method 
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12, 6))
plt.axis('off')
plt.title(r'$ the \ 4th-order \ Runge-Kutta \ algorithm (interpolating \ polynomial) $', fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ \frac{\partial y_n}{\partial x_n} = f(y_n,x_n) $', fontsize = 1.2*AlvaFontSize)

plt.text(0, 4.0/6, r'$ y_{n+1} = \
         y_n + \Delta x * (\frac{k_1}{6} + \frac{k_2}{3} + \frac{k_3}{3} +\frac{k_4}{6})  $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/6, r'$ k_1 = \Delta x * f(y_n,\ x_n) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/6, r'$ k_2 = \Delta x * f(y_n + \frac{k_1}{2},\ x_n + \frac{\Delta x}{2}) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 1.0/6, r'$ k_3 = \Delta x * f(y_n + \frac{k_2}{2},\ x_n + \frac{\Delta x}{2}) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 0.0/6, r'$ k_4 = \Delta x * f(y_n + k_3,\ x_n + \Delta x) $', fontsize = 1.2*AlvaFontSize)
plt.show()

# <codecell>

# simple numerical scheme
for tn in range(totalGPoint_T - 1):
    gridS[tn + 1] = gridS[tn] + dt*(-infecRate*gridS[tn]*gridI[tn] + inOutRate*(totalSIR - gridS[tn]));
    gridI[tn + 1] = gridI[tn] + dt*(+infecRate*gridS[tn]*gridI[tn] - recovRate*gridI[tn] - inOutRate*gridI[tn]);
    gridR[tn + 1] = gridR[tn] + dt*(+recovRate*gridI[tn] - inOutRate*gridR[tn]);
    
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI, label = r'$ \beta S(t)I(t) $', linestyle = 'dotted', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ SIR \ model \ (R_0 = %f,\ \gamma = %f,\ \beta = %f) $'%(reprodNum, recovRate, infecRate), fontsize = AlvaFontSize);
plt.xlabel(r'$time \ (day)$', fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

# <codecell>


