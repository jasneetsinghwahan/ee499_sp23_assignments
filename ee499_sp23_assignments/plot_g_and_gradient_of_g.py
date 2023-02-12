# Problem 3.1 Sub-part 1, 2 and 3 of Watt, et. al.:

import numpy as np
import matplotlib.pyplot as plt
import math         # Use math.log function from the 'math' module
import time
from matplotlib import cm
import copy
# import automatic differentiator to compute gradient module
from autograd import grad
from PIL import Image
from scipy.signal import find_peaks
#from scipy.special import sech     # to calculate the hyperbolic secant (sech) function
#from scipy.special import cosh      # to calculate the hyperbolic secant (sech) function
from math import cosh

fig, (ax1) = plt.subplots(3, 1, figsize=(10, 9))
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# the function g sub-part 1
def g_1(w):
    return w*math.log(w) + (1-w)*math.log(1-w) 

# the derivative of g sub-part 1
def gradientg_1(w):
    return math.log((w)/(1-w))

ax1[0].set_title('g(w) = w.log(w) + (1-w).log(w)', color = 'k',loc='left')
ax1[0].annotate('d(g(w))/dw = log(w/(1-w))', xy=(1,1), \
    xycoords="axes fraction", xytext=(10,10), textcoords="offset points", \
        ha="right", va="top",color = 'g')
ax1[0].annotate('d(g(w))/dw = log(w/(1-w))', xy=(1,0.5),ha='right',va='top',color = 'g')
r_min, r_max = 0.0, 1.0
step = 0.1
inputs = np.arange(r_min + step , r_max - step, step)
function = [g_1(x) for x in inputs]
ax1[0].plot(inputs, function, 'k')
ax1[0].set_xlabel('w', fontsize=12)
ax1[0].set_ylabel('g(w)', fontsize=12)
derivative = [gradientg_1(x) for x in inputs]
ax1[0].plot(inputs, derivative, 'g')
ax1[0].plot(0.5,0,'gD' )


# the function g sub-part 2
def g_2(w):
    return math.log(1+(math.e)**w) 

# the derivative of g sub-part 2
def gradientg_2(w):
    return (((math.e)**w) / 1 + (math.e)**w)

ax1[1].set_title('g(w) = log(1+e**w)', color = 'k',loc='left')
ax1[1].annotate('d(g(w))/dw = (e**w)/(1+e**w)', xy=(1,1), \
    xycoords="axes fraction", xytext=(10,10), textcoords="offset points", \
        ha="right", va="top",color = 'g')
r_min, r_max = -10.0, 10.0
step = 0.1
inputs_2 = np.arange(r_min + step , r_max - step, step)
function_2 = [g_2(x) for x in inputs_2]
ax1[1].plot(inputs_2, function_2, 'k')
ax1[1].set_xlabel('w', fontsize=12)
ax1[1].set_ylabel('g(w)', fontsize=12)
derivative_2 = [gradientg_2(x) for x in inputs_2]
ax1[1].plot(inputs_2, derivative_2, 'g')
ax1[1].plot(-10.0,0,'gD' )

# user-defined sech function
def sech(x):
    return 1/cosh(x)

# the function g sub-part 3
def g_3(w):
    return w*math.tanh(w) 

# the derivative of g sub-part 3
def gradientg_3(w):
    return w*(sech(w))**2 + math.tanh(w) 


ax1[2].set_title('g(w) = w.tanh(w)', color = 'k',loc='left')
ax1[2].annotate('d(g(w))/dw = w.sech2(w) + tanh(w)', xy=(1,1), \
    xycoords="axes fraction", xytext=(10,10), textcoords="offset points", \
        ha="right", va="top",color = 'g')
r_min, r_max = -10.0, 10.0
step = 0.1
inputs_3 = np.arange(r_min + step , r_max - step, step)
function_3 = [g_3(x) for x in inputs_3]
ax1[2].plot(inputs_3, function_3, 'k')
ax1[2].set_xlabel('w', fontsize=12)
ax1[2].set_ylabel('g(w)', fontsize=12)
derivative_2 = [gradientg_3(x) for x in inputs_3]
ax1[2].plot(inputs_3, derivative_2, 'g')
ax1[2].plot(-10.0,0,'gD' )
ax1[2].plot(10.0,0,'gD' )

plt.show()