import numpy as np
import matplotlib.pyplot as plt
from constant import DATA_DIR
from matplotlib import style
plt.rcParams['font.family'] ='Serif'
plt.rcParams['font.weight']="normal"
    # plt.rcParams.update({'font.size': 88})
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

expname = "height"

plt.style.use("bmh")
n = 8
colors = plt.cm.Dark2(np.linspace(0.0, 1.0, n))

PARAM = "y offset k=2"

data = np.load(DATA_DIR+f"{PARAM}_energy_variation.npy")
xs = np.load(DATA_DIR+f"{PARAM}_shield_height.npy")
gfds = np.load(DATA_DIR+f"{PARAM}_gradients_fd.npy")

data_ad = np.load(DATA_DIR+f"{PARAM}_energy_variation_ad.npy")
xs_ad = np.load(DATA_DIR+f"{PARAM}_shield_height_ad.npy")
ads = np.load(DATA_DIR+f"{PARAM}_gradients_fd_ad.npy")

data_ad_re = np.load(DATA_DIR+f"{PARAM}_energy_variation_ad_reparam.npy")
xs_ad_re = np.load(DATA_DIR+f"{PARAM}_shield_height_ad_reparam.npy")
ads_re = np.load(DATA_DIR+f"{PARAM}_gradients_fd_ad_reparam.npy")

# Create a figure and two axes objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 2 rows, 1 column

# Plotting the sales data on the first axis
ax1.plot(xs, data, color=colors[0], alpha=0.7, linewidth=5, label='baseline', linestyle='dashed')  # 'b-o' is a blue line with circle markers
ax1.plot(xs, data_ad, color=colors[1], alpha=0.7, linewidth=5, label="w.o reparam")
ax1.plot(xs, data_ad_re, color=colors[2], alpha=0.7, linewidth=5, label="w reparam")
ax1.set_title(f'values w.r.t shielding {expname}')
ax1.set_xlabel(f'{expname}')
ax1.set_ylabel('tally')
ax1.legend()

# Plotting the temperature data on the second axis
# print(ads)

ax2.plot(xs, ads, color=colors[1], alpha=0.7, linewidth=5, label='auto diff (ad)')  # 'r-s' is a red line with square markers
ax2.plot(xs, ads_re, color=colors[2], alpha=0.7, linewidth=5, label='reparam_ad (ours)')  # 'r-s' is a red line with square markers
ax2.plot(xs, gfds, color=colors[0], alpha=0.7, linewidth=5, label='finite diff', linestyle='dashed')  # 'r-s' is a red line with square markers
ax2.set_title(f'gradient of tally w.r.t {expname}')
ax2.set_xlabel(f'{expname}')
ax2.set_ylabel('gradient')
ax2.legend()

#ax2.semilogx(xs[:], ads_re[:])
#ax2.semilogx(xs[:], ads[:])

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.legend(fontsize="22")
# Show the plots
plt.show()
#plt.savefig("E:/Research/NeutronInv/exp1_gardient_validation_5b.png", bbox_inches='tight')

#TODO: plot correct value
#TODO: plot finite difference gradient 
#TODO: plot auto-diff gradient computed with reparameteration

