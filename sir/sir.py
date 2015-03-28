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

# define RungeKutta solver for numerical solution
def AlvaRungeKutta4YZ(dYdx, gridZ, gridY, gridX, dx, xn):
    zzz = gridZ[xn] ; yyy = gridY[xn]; xxx = gridX[xn]; # keep initial value
    dydx1 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx1*dx/2; gridX[xn] = xxx + dx/2;
    dydx2 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx2*dx/2; gridX[xn] = xxx + dx/2;
    dydx3 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); gridY[xn] = yyy + dydx3*dx; gridX[xn] = xxx + dx;
    dydx4 = dYdx(gridZ[xn], gridY[xn], gridX[xn]); 
    gridY[xn + 1] = yyy + dx*(dydx1/6 + dydx2/3 + dydx3/3 + dydx4/6); 
    gridZ[xn] = zzz; gridY[xn] = yyy; gridX[xn] = xxx; # restore to initial value
    return (gridY[xn + 1]);

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

def dSdt(I, S, t):
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S;
    return dS_dt
def dIdt(S, I, t):    
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I;
    return dI_dt
def dRdt(I, R, t):
    dR_dt = +recovRate*I - inOutRate*R;
    return dR_dt

# Runge Kutta numerical scheme
for tn in range(totalGPoint_T - 1):
    AlvaRungeKutta4YZ(dSdt, gridI, gridS, gridT, dt, tn);
    AlvaRungeKutta4YZ(dIdt, gridS, gridI, gridT, dt, tn);
    AlvaRungeKutta4YZ(dRdt, gridI, gridR, gridT, dt, tn);   
    
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
minT = float(0); maxT = float(60*year);
totalGPoint_T = int(20**4);
gridT = np.linspace(minT, maxT, totalGPoint_T);
dt = (maxT - minT)/totalGPoint_T;

gridS = np.zeros(totalGPoint_T);
gridI = np.zeros(totalGPoint_T);
gridR = np.zeros(totalGPoint_T);

gridI[0] = float(1)/10**6;
gridR[0] = float(0);
gridS[0] = totalSIR - gridI[0] - gridR[0];

def dSdt(I, S, t):
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S;
    return dS_dt
def dIdt(S, I, t):    
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I;
    return dI_dt
def dRdt(I, R, t):
    dR_dt = +recovRate*I - inOutRate*R;
    return dR_dt

# Runge Kutta numerical scheme
for tn in range(totalGPoint_T - 1):
    AlvaRungeKutta4YZ(dSdt, gridI, gridS, gridT, dt, tn);
    AlvaRungeKutta4YZ(dIdt, gridS, gridI, gridT, dt, tn);
    AlvaRungeKutta4YZ(dRdt, gridI, gridR, gridT, dt, tn);   
    
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


