import numpy as np
import matplotlib.pyplot as plt
from constant import DATA_DIR

data = np.load(DATA_DIR+"energy_variation.npy")
xs = np.load(DATA_DIR+"shield_height.npy")
gfds = np.load(DATA_DIR+"gradients_fd.npy")

data_ad = np.load(DATA_DIR+"energy_variation_ad.npy")
xs_ad = np.load(DATA_DIR+"shield_height_ad.npy")
ads = np.load(DATA_DIR+"gradients_fd_ad.npy")

data_ad_re = np.load(DATA_DIR+"energy_variation_ad_reparam.npy")
xs_ad_re = np.load(DATA_DIR+"shield_height_ad_reparam.npy")
ads_re = np.load(DATA_DIR+"gradients_fd_ad_reparam.npy")

# Create a figure and two axes objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 8))  # 2 rows, 1 column

# Plotting the sales data on the first axis
ax1.plot(xs, data, 'b-o', alpha=0.5, label='baseline')  # 'b-o' is a blue line with circle markers
ax1.plot(xs, data_ad, 'r-o', alpha=0.5, label="w.o reparam")
ax1.plot(xs, data_ad_re, 'g-o', alpha=0.5, label="w reparam")
ax1.set_title('values w.r.t shielding height')
ax1.set_xlabel('height')
ax1.set_ylabel('tally')
ax1.legend()

# Plotting the temperature data on the second axis
# print(ads)
ax2.plot(xs, gfds, 'b', alpha=0.5, label='fd')  # 'r-s' is a red line with square markers
ax2.plot(xs, ads, 'r', alpha=0.5, label='ad')  # 'r-s' is a red line with square markers
ax2.plot(xs, ads_re, 'g', alpha=0.5, label='reparam_ad (ours)')  # 'r-s' is a red line with square markers
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

