import numpy as np
import matplotlib.pyplot as plt
from constant import DATA_DIR

data = np.load(DATA_DIR+"energy_variation.npy")
xs = np.load(DATA_DIR+"shield_height.npy")
gfds = np.load(DATA_DIR+"gradients_fd.npy")

data_ad = np.load(DATA_DIR+"energy_variation_ad.npy")
xs_ad = np.load(DATA_DIR+"shield_height_ad.npy")
ads = np.load(DATA_DIR+"gradients_fd_ad.npy")

# Create a figure and two axes objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 8))  # 2 rows, 1 column

# Plotting the sales data on the first axis
ax1.plot(xs, data, 'b-o', label='baseline')  # 'b-o' is a blue line with circle markers
ax1.plot(xs, data_ad, 'r-o', label="with reparam")
ax1.set_title('values w.r.t shielding height')
ax1.set_xlabel('height')
ax1.set_ylabel('tally')
ax1.legend()

# Plotting the temperature data on the second axis
ax2.plot(xs, gfds, 'b-s', label='fd')  # 'r-s' is a red line with square markers
ax2.plot(xs, ads, 'r-s', label='reparam_ad')  # 'r-s' is a red line with square markers
ax2.set_title('gradient of tally w.r.t height')
ax2.set_xlabel('height')
ax2.set_ylabel('gradient')
ax2.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()

#TODO: plot correct value
#TODO: plot finite difference gradient 
#TODO: plot auto-diff gradient computed with reparameteration

