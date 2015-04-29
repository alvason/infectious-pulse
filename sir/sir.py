# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Infectious Pulse
# https://github.com/alvason/infectious-pulse/
# 
# ### One-strain SIR evolution --- its revival and equilibrium state 

# <codecell>

'''
author: Alvason Zhenhua Li
date:   03/23/2015
'''

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

AlvaFontSize = 23
AlvaFigSize = (16, 6)
numberingFig = 0


numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(12, 3))
plt.axis('off')
plt.title(r'$ Susceptible-Infectious-Recovered \ equation $', fontsize = AlvaFontSize)
plt.text(0, 2.0/3, r'$ \frac{\partial S(t)}{\partial t} = \
         -\beta S(t)I(t) +\mu N -\mu S(t)$', fontsize = 1.2*AlvaFontSize)
plt.text(0, 1.0/3, r'$ \frac{\partial I(t)}{\partial t} = \
         +\beta S(t)I(t) - \gamma I(t) -\mu I(t) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 0.0/3, r'$ \frac{\partial R(t)}{\partial t} = \
         +\gamma I(t) - \mu R(t) $', fontsize = 1.2*AlvaFontSize)
plt.show()

# define RK4 for a list of coupled differential equations
def AlvaRungeKutta4List(pde_array, startingOut_Value, min_Input, max_Input, totalGPoint_Input):
    # size of a list of equations
    outWay = pde_array.size
    # initialize the whole memory-space for output and input
    inWay = 1
    gridOutIn_array = np.zeros([outWay + inWay, totalGPoint_Input])
    # loading starting output values (define the first m arrays as output memory-space)
    for i in range(outWay):
        gridOutIn_array[i, 0] = startingOut_Value[i]
    # griding input value 
    gridingInput = np.linspace(min_Input, max_Input, num = totalGPoint_Input, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gridOutIn_array[-inWay] = gridingInput[0]
    # step-size (increment of input)
    dx = gridingInput[1]
    # starting
    # initialize the memory-space for local try-step 
    dydx1_array = np.zeros(outWay)
    dydx2_array = np.zeros(outWay)
    dydx3_array = np.zeros(outWay)
    dydx4_array = np.zeros(outWay)
    for inPoint in range(totalGPoint_Input - 1):
        # keep initial value
        currentOut_Value = np.copy(gridOutIn_array[:-inWay, inPoint])
        currentIn_Value = np.copy(gridOutIn_array[-inWay, inPoint])
        # first try-step
        for i in range(outWay):
            dydx1_array[i] = pde_array[i](gridOutIn_array[:, inPoint]) # computing ratio   
        gridOutIn_array[:-inWay, inPoint] = currentOut_Value[:] + dydx1_array[:]*dx/2 # update output
        gridOutIn_array[-inWay, inPoint] = currentIn_Value + dx/2 # update input
        # second half try-step
        for i in range(outWay):
            dydx2_array[i] = pde_array[i](gridOutIn_array[:, inPoint]) # computing ratio
        gridOutIn_array[:-inWay, inPoint] = currentOut_Value[:] + dydx2_array[:]*dx/2 # update output
        gridOutIn_array[-inWay, inPoint] = currentIn_Value + dx/2
        # third half try-step
        for i in range(outWay):
            dydx3_array[i] = pde_array[i](gridOutIn_array[:, inPoint]) # computing ratio
        gridOutIn_array[:-inWay, inPoint] = currentOut_Value[:] + dydx3_array[:]*dx # update output
        gridOutIn_array[-inWay, inPoint] = currentIn_Value + dx
        # fourth try-step
        for i in range(outWay):
            dydx4_array[i] = pde_array[i](gridOutIn_array[:, inPoint]) # computing ratio
        # solid step (update the next output) by accumulate all the try-steps with proper adjustment
        gridOutIn_array[:-inWay, inPoint + 1] = currentOut_Value[:] + dx*(dydx1_array[:]/6 
                                                                                      + dydx2_array[:]/3 
                                                                                      + dydx3_array[:]/3 
                                                                                      + dydx4_array[:]/6)
        # restore to initial value
        gridOutIn_array[:-inWay, inPoint] = np.copy(currentOut_Value)
        gridOutIn_array[-inWay, inPoint] = np.copy(currentIn_Value)
        # end of loop
    return (gridOutIn_array[:-inWay])

# <codecell>

''' starting from one infected '''
# setting parameter
timeUnit = 'day'
if timeUnit == 'day':
    day = 1
    year = 365 
elif timeUnit == 'year':
    year = 1
    day = float(1)/365 
    
totalSIR = float(1) # total population
reprodNum = float(1.8) # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
inOutRate = float(1)/(30*year) # birth rate per year
infecRate = reprodNum*(recovRate + inOutRate)/totalSIR # per year, per person, per total-population

# initial boundary condition
minT = float(0)
maxT = float(1*year)
totalGPoint_T = int(10**4 + 1)
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

gridS = np.zeros(totalGPoint_T)
gridI = np.zeros(totalGPoint_T)
gridR = np.zeros(totalGPoint_T)

# initial output condition
gridI[0] = float(1)/10**6
gridR[0] = float(0)
gridS[0] = totalSIR - gridI[0] - gridR[0]

def dSdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3]
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S
    return dS_dt
def dIdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3] 
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I
    return dI_dt
def dRdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3]
    dR_dt = recovRate*I - inOutRate*R
    return dR_dt

# Runge Kutta numerical solution
pde_array = np.array([dSdt, dIdt, dRdt])
startingOut_Value = np.array([gridS[0], gridI[0], gridR[0]])
gridOut_array = AlvaRungeKutta4List(pde_array, startingOut_Value, minT, maxT, totalGPoint_T)
    
gridS = gridOut_array[0]  
gridI = gridOut_array[1]
gridR = gridOut_array[2]
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI*day, label = r'$ \beta \ S(t)I(t) $', linestyle = 'dashed', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ Prevalence \ and \ incidence \ of \ SIR $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize)
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize)
plt.ylim(0, totalSIR*1.1)
plt.legend(loc = (1,0))
plt.text(maxT, totalSIR*6.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*5.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*4.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*3.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
plt.show()

# <codecell>

''' approaching equilibrium state from one infected '''
# setting parameter
timeUnit = 'year'
if timeUnit == 'day':
    day = 1
    year = 365 
elif timeUnit == 'year':
    year = 1
    day = float(1)/365 
    
totalSIR = float(1) # total population
reprodNum = float(1.8) # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
inOutRate = float(1)/(30*year) # birth rate per year
infecRate = reprodNum*(recovRate + inOutRate)/totalSIR # per year, per person, per total-population

# initial boundary condition
minT = float(0)
maxT = float(200*year)
totalGPoint_T = int(10**4 + 1)
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

gridS = np.zeros(totalGPoint_T)
gridI = np.zeros(totalGPoint_T)
gridR = np.zeros(totalGPoint_T)

# initial output condition
gridI[0] = float(1)/10**6
gridR[0] = float(0)
gridS[0] = totalSIR - gridI[0] - gridR[0]

def dSdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3]
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S
    return dS_dt
def dIdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3] 
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I
    return dI_dt
def dRdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3]
    dR_dt = recovRate*I - inOutRate*R
    return dR_dt

# Runge Kutta numerical solution
pde_array = np.array([dSdt, dIdt, dRdt])
startingOut_Value = np.array([gridS[0], gridI[0], gridR[0]])
gridOut_array = AlvaRungeKutta4List(pde_array, startingOut_Value, minT, maxT, totalGPoint_T)
    
gridS = gridOut_array[0]  
gridI = gridOut_array[1]
gridR = gridOut_array[2]
numberingFig = numberingFig + 1;
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
axis = figure.add_subplot(1, 1, 1)
axis.plot(gridT, gridS, label = r'$ S(t) $')
axis.plot(gridT, gridR, label = r'$ R(t) $')
axis.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
axis.set_xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize)
axis.set_ylabel(r'$ S \ , \ R $', fontsize = AlvaFontSize)
axis.set_ylim(0, totalSIR*1.1)
axis.text(maxT*1.1, totalSIR*6.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
axis.text(maxT*1.1, totalSIR*5.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
axis.text(maxT*1.1, totalSIR*4.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
axis.text(maxT*1.1, totalSIR*3.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
axis.legend(loc = (1.1, 0))

axis2 = axis.twinx()
axis2.plot(gridT, gridI, label = r'$ I(t) $', color = 'red')
axis2.set_ylabel(r'$ I $', fontsize = AlvaFontSize, color = 'red')
for tl in axis2.get_yticklabels(): tl.set_color('red')
axis2.legend(loc = (1.1, 0.5))

plt.legend(loc = (1.1, 0.3))
plt.title(r'$ Revival \ and \ equilibrium \ of \ SIR $', fontsize = AlvaFontSize)
plt.grid(True)
plt.show()

# <codecell>

''' starting from equilibrium infected number '''
# setting parameter
timeUnit = 'day'
if timeUnit == 'day':
    day = 1
    year = 365 
elif timeUnit == 'year':
    year = 1
    day = float(1)/365 
    
totalSIR = float(1) # total population
reprodNum = float(1.8) # basic reproductive number R0: one infected person will transmit to 1.8 person 
recovRate = float(1)/(4*day) # 4 days per period ==> rate/year = 365/4
inOutRate = float(1)/(30*year) # birth rate per year
infecRate = reprodNum*(recovRate + inOutRate)/totalSIR # per year, per person, per total-population

# initial boundary condition
minT = float(0)
maxT = float(1*year)
totalGPoint_T = int(10**4 + 1)
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

gridS = np.zeros(totalGPoint_T)
gridI = np.zeros(totalGPoint_T)
gridR = np.zeros(totalGPoint_T)

# initial output condition (equlibrium state)
gridS[0] = (recovRate + inOutRate)/infecRate
gridI[0] = inOutRate*(totalSIR - gridS[0])/(infecRate*gridS[0])
gridR[0] = totalSIR - gridS[0] - gridI[0]

def dSdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3]
    dS_dt = -infecRate*S*I + inOutRate*totalSIR - inOutRate*S
    return dS_dt
def dIdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3] 
    dI_dt = +infecRate*S*I - recovRate*I - inOutRate*I
    return dI_dt
def dRdt(SIRT = [], *args):
    S = SIRT[0]
    I = SIRT[1]
    R = SIRT[2]
    T = SIRT[3]
    dR_dt = recovRate*I - inOutRate*R
    return dR_dt

# Runge Kutta numerical solution
pde_array = np.array([dSdt, dIdt, dRdt])
startingOut_Value = np.array([gridS[0], gridI[0], gridR[0]])
gridOut_array = AlvaRungeKutta4List(pde_array, startingOut_Value, minT, maxT, totalGPoint_T)
    
gridS = gridOut_array[0]  
gridI = gridOut_array[1]
gridR = gridOut_array[2]
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(gridT, gridS, label = r'$ S(t) $')
plt.plot(gridT, gridR, label = r'$ R(t) $')
plt.plot(gridT, gridI, label = r'$ I(t) $')
plt.plot(gridT, infecRate*gridS*gridI*day, label = r'$ \beta \ S(t)I(t) $', linestyle = 'dashed', color = 'red')
plt.plot(gridT, gridS + gridI + gridR, label = r'$ S(t)+I(t)+R(t) $', color = 'black')
plt.grid(True)
plt.title(r'$ Prevalence \ and \ incidence \ of \ SIR \ (equilibrium \ state)$', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize)
plt.ylabel(r'$ Proportion \ of \ population $', fontsize = AlvaFontSize)
plt.legend(loc = (1,0))
plt.text(maxT, totalSIR*6.0/6, r'$ R_0 = %f $'%(reprodNum), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*5.0/6, r'$ \gamma = %f $'%(recovRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*4.0/6, r'$ \beta = %f $'%(infecRate), fontsize = AlvaFontSize)
plt.text(maxT, totalSIR*3.0/6, r'$ \mu = %f $'%(inOutRate), fontsize = AlvaFontSize)
plt.text(maxT/2, totalSIR*3.5/6, r'$ S_0 = %f $'%(gridS[0]), fontsize = AlvaFontSize)
plt.text(maxT/2, totalSIR*2.0/6, r'$ R_0 = %f $'%(gridR[0]), fontsize = AlvaFontSize)
plt.text(maxT/2, totalSIR*0.5/6, r'$ I_0 = %f $'%(gridI[0]), fontsize = AlvaFontSize)
plt.show()

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(12,4))
plt.axis('off')
plt.title(r'$ Susceptible-Infectious-Recovered \ equation \ (equilibrium \ state) $',fontsize = AlvaFontSize)
plt.text(0, 3.0/4,r'$ \frac{\partial S(t)}{\partial t} = \
         -\beta S(t)I(t) +\mu N -\mu S(t) = 0 \Longrightarrow I(t) = \frac{\mu}{\beta S(t)} (N - S(t)) \
         \Longrightarrow I_0 = \frac{\mu N}{\beta} (\frac{N}{S_0} - 1) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/4,r'$ \frac{\partial I(t)}{\partial t} = \
         +\beta S(t)I(t) - \gamma I(t) -\mu I(t) = 0 \Longrightarrow \beta S(t) - \gamma - \mu = 0 \
         \Longrightarrow S_0 = \frac{\gamma + \mu}{\beta} $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 1.0/4, r'$ \frac{\partial R(t)}{\partial t} = \
         +\gamma I(t) - \mu R(t) \Longrightarrow R_0 = \frac{\gamma}{\mu} I_0 $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 0.0/4,r'$ since \ \beta = \frac{\Re_0 (\gamma + \mu)}{N} \Longrightarrow S_0 = \frac{N}{\Re_0} \
         \ and \ I_0 = \frac{\mu N}{\beta} (\Re_0 - 1) \
         \ and \ R_0 = \frac{\gamma N}{\beta} (\Re_0 - 1)$', fontsize = 1.2*AlvaFontSize)
plt.show()

# <codecell>


