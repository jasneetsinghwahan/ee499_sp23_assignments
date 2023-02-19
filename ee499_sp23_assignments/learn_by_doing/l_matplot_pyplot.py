import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt

def sigma(v):
    return expit(v)

def derivative_sigma(v):
    return expit(v)*(1-expit(v))

# plotting these two functions    
#plt.figure(1)                   # the first figure
#fig, ax = plt.subplots()        # a figure with a single Axes
r_min, r_max = -4.0, 4.0
inputs = np.arange(r_min, r_max, 0.1)
sigma_func_values = sigma(inputs)
derivative_sigma_values = derivative_sigma(inputs)

# Using objectove orinetded explicit interface style
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')              # ax stores the element returned by subplot
ax.plot(inputs, sigma_func_values, label=r'$\sigma(v)$')                    # Plot more data on the axes...
ax.plot(inputs, derivative_sigma_values, label=r"$\frac{d \sigma(v)}{dv}$") # ... and some more.
ax.set_xlabel('inputs')  # Add an x-label to the axes.
ax.set_ylabel('value')  # Add a y-label to the axes.
custom_title = "plot of {} and {}".format(r'$\sigma(v)$', r"$\frac{d \sigma(v)}{dv}$")
ax.set_title(custom_title)  # Add a title to the axes.

ax.legend()  # Add a legend
ax.grid()       # draws the grid
plt.show()