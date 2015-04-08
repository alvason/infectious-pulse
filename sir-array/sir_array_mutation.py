# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Infectious Pulse
# https://github.com/alvason/infectious-pulse/
# 
# ### Many-strain SIR evolution --- its equilibrium state and infectious pulse due to mutation

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

import alva_machinery as alva

AlvaFontSize = 23;
AlvaFigSize = (9, 6);
numberingFig = 0;

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,3))
plt.axis('off')
plt.title(r'$ Many-strain \ SIR \ equations (mutation \ only) $',fontsize = AlvaFontSize)
plt.text(0,2.0/3,r'$ \frac{\partial S_n(t)}{\partial t} = \
         -\beta S_n(t)I(t) +\mu N -\mu S_n(t)$', fontsize = 1.2*AlvaFontSize)
plt.text(0,1.0/3,r'$ \frac{\partial I_n(t)}{\partial t} = \
         +\beta S_n(t)I(t) - \gamma I_n(t) -\mu I_n(t) + m \frac{I_{n-1}(t) - 2I_n(t) + I_{n+1}(t)}{(1)^2}$'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0,0.0/3,r'$ \frac{\partial R_n(t)}{\partial t} = \
         +\gamma I_n(t) - \mu R_n(t) $', fontsize = 1.2*AlvaFontSize)
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
mutatRate = float(1)/10**6 # mutation rate

# time boundary and griding condition
minT = float(0); maxT = float(1*year);
totalGPoint_T = int(10**3 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

# space boundary and griding condition
minX = float(0); maxX = float(9);
totalGPoint_X = int(10);
gridX = np.linspace(minX, maxX, totalGPoint_X);
gridingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = gridingX[0]
dx = gridingX[1]

gridS_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridI_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridR_array = np.zeros([totalGPoint_X, totalGPoint_T])

# initial output condition
gridI_array[0:1, 0] = float(1)/10**6
gridR_array[0, 0] = float(0)
gridS_array[:, 0] = totalSIR - gridI_array[:, 0] - gridR_array[:, 0]

def dSdt_array(SIRxt = [], *args):
    # naming
    S = SIRxt[0]
    I = SIRxt[1]
    R = SIRxt[2]
    x_totalPoint = SIRxt.shape[1]
    # there are n dSdt
    dS_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    for xn in range(x_totalPoint):
        dS_dt_array[xn] = -infecRate*S[xn]*I[xn] + inOutRate*totalSIR - inOutRate*S[xn]
    return(dS_dt_array)
def dIdt_array(SIRxt = [], *args):
    # naming
    S = SIRxt[0]
    I = SIRxt[1]
    R = SIRxt[2]
    x_totalPoint = SIRxt.shape[1]
    # there are n dIdt
    dI_dt_array = np.zeros(x_totalPoint)
    # each dIdt with the same equation form
    Icopy = np.copy(I)
    centerX = Icopy[:]
    leftX = np.roll(Icopy[:], 1)
    rightX = np.roll(Icopy[:], -1)
    leftX[0] =centerX[0]
    rightX[-1] = centerX[-1]
    for xn in range(x_totalPoint):
        dI_dt_array[xn] = +infecRate*S[xn]*I[xn] - recovRate*I[xn] - inOutRate*I[xn] + mutatRate*(leftX[xn]
                                                                                                  - 2*centerX[xn]
                                                                                                  + rightX[xn])/(dx**2)
    return(dI_dt_array)
def dRdt_array(SIRxt = [], *args):
    # naming
    S = SIRxt[0]
    I = SIRxt[1]
    R = SIRxt[2]
    x_totalPoint = SIRxt.shape[1]
    # there are n dRdt
    dR_dt_array = np.zeros(x_totalPoint)
    # each dIdt with the same equation form
    for xn in range(x_totalPoint):
        dR_dt_array[xn] = +recovRate*I[xn] - inOutRate*R[xn]
    return(dR_dt_array)

# Runge Kutta numerical solution
pde_array = np.array([dSdt_array, dIdt_array, dRdt_array])
startingOut_Value = np.array([gridS_array, gridI_array, gridR_array])
gridOut_array = alva.AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)

# plotting
gridS = gridOut_array[0]  
gridI = gridOut_array[1]
gridR = gridOut_array[2]

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.contourf(gridT, gridX, gridI, levels = np.arange(0, 0.12, 0.002))
plt.title(r'$ Many \ strains \ infectious \ pulse $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Strain \ space $', fontsize = AlvaFontSize);
plt.colorbar()
plt.show()

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS.T)
plt.plot(gridT, gridR.T)
plt.plot(gridT, gridI.T)
plt.plot(gridT, gridS.T + gridI.T + gridR.T, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.title(r'$ Many-strain \ SIR $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
plt.text(maxT, totalSIR*6.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*5.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*4.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*3.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*2.0/6, r'$ m = %f $'%(mutatRate), fontsize = AlvaFontSize)
plt.show()

# <codecell>

# plot by listing each strain 
numberingFig = numberingFig + 1;
for i in range(totalGPoint_X):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridS[i], label = r'$ S_{%i}(t) $'%(i))
    plt.plot(gridT, gridR[i], label = r'$ R_{%i}(t) $'%(i))
    plt.plot(gridT, gridI[i], label = r'$ I_{%i}(t) $'%(i))
    plt.plot(gridT, infecRate*gridS[i].T*gridI[i].T*day, label = r'$ \beta \ S_{%i}(t)I_{%i}(t) $'%(i, i)
             , linestyle = 'dashed', color = 'red')
    plt.plot(gridT, (gridS[i] + gridI[i] + gridR[i]).T, label = r'$ S_{%i}(t)+I_{%i}(t)+R_{%i}(t) $'%(i, i, i)
             , color = 'black')
    plt.grid(True)
    plt.title(r'$ Prevalence \ and \ incidence \ of \ SIR $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize);
    plt.text(maxT, totalSIR*7.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
    plt.text(maxT, totalSIR*6.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
    plt.text(maxT, totalSIR*5.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
    plt.text(maxT, totalSIR*4.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
    plt.text(maxT, totalSIR*3.0/6, r'$ m = %f $'%(mutatRate), fontsize = AlvaFontSize)
    plt.legend(loc = (1,0))
    plt.show()

# <codecell>

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.pcolormesh(gridT, gridX, gridI)
plt.title(r'$ Many-strain \ SIR $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Strain \ space $', fontsize = AlvaFontSize);
plt.colorbar()
plt.show()

