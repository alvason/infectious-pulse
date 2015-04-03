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

# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'day':
    day = 1; year = 365; 
elif timeUnit == 'year':
    year = 1; day = float(1)/365; 
    
totalSIR = float(1); # total population
reprodNum = 1.8 # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
infecRate = reprodNum*recovRate/totalSIR # per year, per person, per total-population
inOutRate = float(1)/(30*year) # birth rate per year

# initial condition
minT = float(0); maxT = float(1*year);
totalGPoint_T = int(10**3 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

gridS = np.zeros(totalGPoint_T);
gridI = np.zeros(totalGPoint_T);
gridR = np.zeros(totalGPoint_T);

gridI[0] = float(1)/10**6;
gridR[0] = float(0);
gridS[0] = totalSIR - gridI[0] - gridR[0];

def dSdt(SIRT = [], *args):
    S = SIRT[0]; I = SIRT[1]; R = SIRT[2]; T = SIRT[3];
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S;
    return dS_dt
def dIdt(SIRT = [], *args):
    S = SIRT[0]; I = SIRT[1]; R = SIRT[2]; T = SIRT[3];    
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I;
    return dI_dt
def dRdt(SIRT = [], *args):
    S = SIRT[0]; I = SIRT[1]; R = SIRT[2]; T = SIRT[3];    
    dR_dt = recovRate*I - inOutRate*R;
    return dR_dt

# Runge Kutta numerical solution
dYdt_array = np.array([dSdt, dIdt, dRdt])
gridY_array = np.array([gridS, gridI, gridR])
for tn in range(totalGPoint_T - 1):
    AlvaRungeKutta4List(dYdt_array, gridY_array, gridT, dt, tn) 

gridS = gridY_array[0]  
gridI = gridY_array[1]
gridR = gridY_array[2]
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI*day, label = r'$ \beta \ S(t)I(t) $', linestyle = 'dashed', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ Prevalence \ and \ incidence \ of \ SIR $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.text(maxT, 6.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
plt.text(maxT, 5.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
plt.text(maxT, 4.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
plt.text(maxT, 3.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
plt.show()

# <codecell>

# setting parameter
timeUnit = 'year'
if timeUnit == 'day':
    day = 1; year = 365; 
elif timeUnit == 'year':
    year = 1; day = float(1)/365; 
    
totalSIR = float(1); # total population
reprodNum = 1.8 # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
infecRate = reprodNum*recovRate/totalSIR # per year, per person, per total-population
inOutRate = float(1)/(30*year) # birth rate per year

# initial condition
minT = float(0); maxT = float(200*year);
totalGPoint_T = int(10**4 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

gridS = np.zeros(totalGPoint_T);
gridI = np.zeros(totalGPoint_T);
gridR = np.zeros(totalGPoint_T);

gridI[0] = float(1)/10**6;
gridR[0] = float(0);
gridS[0] = totalSIR - gridI[0] - gridR[0];

def dSdt(SIRT = [], *args):
    S = SIRT[0]; I = SIRT[1]; R = SIRT[2]; T = SIRT[3];
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S;
    return dS_dt
def dIdt(SIRT = [], *args):
    S = SIRT[0]; I = SIRT[1]; R = SIRT[2]; T = SIRT[3];    
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I;
    return dI_dt
def dRdt(SIRT = [], *args):
    S = SIRT[0]; I = SIRT[1]; R = SIRT[2]; T = SIRT[3];    
    dR_dt = recovRate*I - inOutRate*R;
    return dR_dt

# Runge Kutta numerical solution
dYdt_array = np.array([dSdt, dIdt, dRdt])
gridY_array = np.array([gridS, gridI, gridR])
for tn in range(totalGPoint_T - 1):
    AlvaRungeKutta4List(dYdt_array, gridY_array, gridT, dt, tn) 

gridS = gridY_array[0]  
gridI = gridY_array[1]
gridR = gridY_array[2]
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI*day, label = r'$ \beta \ S(t)I(t) $', linestyle = 'dashed', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ Revival \ and \ equilibrium \ of \ SIR $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.text(maxT, 6.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
plt.text(maxT, 5.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
plt.text(maxT, 4.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
plt.text(maxT, 3.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
plt.show()

# <codecell>


