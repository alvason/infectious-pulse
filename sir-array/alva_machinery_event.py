# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Infectious Pulse
# https://github.com/alvason/infectious-pulse/
# 
# ## Homemade machinery for solving partial differential equations
# ### Runge-kutta algorithm for an array of coupled partial differential equation 

# <codecell>

'''
author: Alvason Zhenhua Li
date:   03/23/2015

Home-made machinery for solving partial differential equations
'''
import numpy as np

# define RK4 for an array (3, n) of coupled differential equations
def AlvaRungeKutta4XT(pde_array, initial_Out, minX_In, maxX_In, totalPoint_X, minT_In, maxT_In, totalPoint_T, event_tn_In):
    global eventName
    # primary size of pde equations
    outWay = pde_array.shape[0]
    # initialize the whole memory-space for output and input
    inWay = 1; # one layer is enough for storing "x" and "t" (only two list of variable)
    # define the first part of array as output memory-space
    gOutIn_array = np.zeros([outWay + inWay, totalPoint_X, totalPoint_T])
    # loading starting output values
    for i in range(outWay):
        gOutIn_array[i, :, :] = initial_Out[i, :, :]
    # griding input X value  
    gridingInput_X = np.linspace(minX_In, maxX_In, num = totalPoint_X, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gOutIn_array[-inWay, :, 0] = gridingInput_X[0]
    # step-size (increment of input X)
    dx = gridingInput_X[1]
    # griding input T value  
    gridingInput_T = np.linspace(minT_In, maxT_In, num = totalPoint_T, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gOutIn_array[-inWay, 0, :] = gridingInput_T[0]
    # step-size (increment of input T)
    dt = gridingInput_T[1]
    # starting
    # initialize the memory-space for local try-step 
    dydt1_array = np.zeros([outWay, totalPoint_X])
    dydt2_array = np.zeros([outWay, totalPoint_X])
    dydt3_array = np.zeros([outWay, totalPoint_X])
    dydt4_array = np.zeros([outWay, totalPoint_X])
    # initialize the memory-space for keeping current value
    currentOut_Value = np.zeros([outWay, totalPoint_X])
    for tn in range(totalPoint_T - 1):
        eventName = event_tn_In[0, 1] 
        if tn > int(totalPoint_T*(event_tn_In[1, 0]/(maxT_In - minT_In))):
            eventName = event_tn_In[1, 1]
        # keep initial value at the moment of tn
        currentOut_Value[:, :] = np.copy(gOutIn_array[:-inWay, :, tn])
        currentIn_T_Value = np.copy(gOutIn_array[-inWay, 0, tn])
        # first try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                dydt1_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt1_array[:, :]*dt/2 # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # second half try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                dydt2_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt2_array[:, :]*dt/2 # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # third half try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                dydt3_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt3_array[:, :]*dt # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt # update input
        # fourth try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                dydt4_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio 
        # solid step (update the next output) by accumulate all the try-steps with proper adjustment
        gOutIn_array[:-inWay, :, tn + 1] = currentOut_Value[:, :] + dt*(dydt1_array[:, :]/6 
                                                                                      + dydt2_array[:, :]/3 
                                                                                      + dydt3_array[:, :]/3 
                                                                                      + dydt4_array[:, :]/6)
        # restore to initial value
        gOutIn_array[:-inWay, :, tn] = np.copy(currentOut_Value[:, :])
        gOutIn_array[-inWay, 0, tn] = np.copy(currentIn_T_Value)
        # end of loop
    return (gOutIn_array[:-inWay, :])

# <codecell>


