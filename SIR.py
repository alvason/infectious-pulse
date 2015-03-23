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
AlvaFigSize = (6, 6);
numberingFig = 0;


numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,3))
plt.axis('off')
plt.title(r'$ Susceptible-Infectious-Recovered \ equation $',fontsize = AlvaFontSize)
plt.text(0,2.0/3,r'$ \frac{\partial S(t)}{\partial t} = \
         -\beta S(t) + \gamma I(t) +\mu (N-S(t))$', fontsize = 1.2*AlvaFontSize)
plt.text(0,1.0/3,r'$ \frac{\partial I(t)}{\partial t} = \
         +\beta S(t) - \gamma I(t) -\mu I(t) $', fontsize = 1.2*AlvaFontSize)
plt.text(0,0.0/3,r'$ \frac{\partial R(t)}{\partial t} = \
         +\gamma R(t) - \mu R(t) $', fontsize = 1.2*AlvaFontSize)

plt.show()

# <codecell>

# Initial condition
N = 1; # total population
inOutRate = float(1)/30 # birth rate per year
reproNumb = 1.8 # basic reproductive number R0: one infected person will transmitt to 1.8 person 
infecRate =  # per year, per person
recovRate = float(365)/4 # 4 days per period ==> rate/year = 365/4

