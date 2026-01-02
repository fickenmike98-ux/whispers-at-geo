import sys
import matplotlib

# 1. Force the backend explicitly BEFORE importing pyplot
# If 'TkAgg' fails, try 'Qt5Agg' (case sensitive)
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# 2. Verify backend actually changed
print(f"Operational Backend: {matplotlib.get_backend()}")

# 3. Simple test to force window focus
plt.ion() # Turn on interactive mode
fig = plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.draw()
plt.pause(0.1) # Gives the window a moment to spawn

plt.show(block=True)