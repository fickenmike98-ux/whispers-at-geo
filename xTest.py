import matplotlib
matplotlib.use('Qt5Agg') # Or 'TkAgg' if you prefer
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("SDA Connection Test")
plt.show(block=True)