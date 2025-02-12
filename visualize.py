import numpy as np
import matplotlib.pyplot as plt
from constant import DATA_DIR, TEMP_DIR
from matplotlib import style
plt.rcParams['font.family'] ='Serif'
plt.rcParams['font.weight']="normal"
    # plt.rcParams.update({'font.size': 88})
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

# n = 8
colors = plt.cm.Dark2(np.linspace(0.0, 1.0, 8))

def optimization_plot():
    # expname = "height"
    plt.style.use("bmh")

    data = np.load(TEMP_DIR+f"enery.npy")
    volume = np.load(TEMP_DIR+f"volumes.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.plot(data, color=colors[0], alpha=0.7, linewidth=5, label='enery') # linestyle='dashed'
    ax2.plot(volume, color=colors[1], alpha=0.7, linewidth=5, label='volume')
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.legend(fontsize="22")
    # Show the plots
    plt.show()

def gradient_space_validation():
    expname = "height"

    plt.style.use("bmh")
    

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
def y1(x):
    # return (3 * (x-3.0) - 4.0) ** 2
    return (np.exp(-x * 1.0) - np.exp(-0.5 * 1.0)) ** 2.0

def y2(x):
    return (np.exp(-x * .5) - np.exp(-0.5 * .5)) ** 2.0 * 0.1


def invx( noise, x, yfunc):
    return yfunc(x) + noise

def incx( noise, x, yfunc):
    return yfunc(x) + noise

def oneline(mean, std_dev, x, size, spp, noisefunc, yfunc):
    count = 0
    for i in range(spp):
        M2 = 0
        xn = np.random.normal(loc=mean, scale=std_dev, size=size)
        print(xn[0])
        ns = noisefunc(xn, x, yfunc)
        count += 1
        if i == 0:
            mean_x = ns
        delta = ns - mean_x
        mean_x += delta / count
        delta2 = ns - mean_x
        M2 += delta * delta2

    mean_value, variance, sample_variance = (mean_x, M2/count, M2/(count-1))
    std3 = np.sqrt(variance)
    # print(std3)
    # print(std3)
    weight = 1.0 / (std3 + 0.0001)
    return weight

# import random
# optimization_plot()
def presentationPlot():
    plt.style.use("dark_background")

    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=44)
    plt.rc('ytick', labelsize=44)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # 2 rows, 1 column
    mean = 0.0
    std_dev = 0.02
    size = 1000
    noise1 = np.random.normal(loc=mean, scale=std_dev, size=size)
    x = np.linspace(0, 10, size)

    noise2 = np.random.normal(loc=mean, scale=std_dev , size=size)
    # print(noise1)
    y_1 = y1(x)
    y_2 = y2(x)

    noisey = invx(noise1, x, y1)
    noisey2 = incx(noise2, x, y2)

    weight1 = 0.5
    # oneline(mean, std_dev, x, size, 100, invx, y1)
    weight2 = 0.1
    # oneline(mean, std_dev * 10.0, x, size, 100, incx, y2)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
 
    ax.plot(x, noisey2, color=colors[1], linewidth=2, alpha=0.5 )
    ax.plot(x, (noisey), color=colors[0], linewidth=2, alpha=0.5)
    ax.plot(x, y_1, color=colors[0], label="pixel 1")
    ax.plot(x, y_2, color=colors[1], label="pixel 2")
    # ax.plot(x, (noisey2 + noisey), color=colors[2], linewidth=2, alpha=0.5, label="sum")
    # ax.plot(x, (noisey2 * weight2 + noisey * weight1), color=colors[3], linewidth=2, alpha=0.5, label="weighted sum")
    
    # ax.plot(x, y_1 + y_2, color=colors[2])
    ax.set_ylabel("loss")
    ax.set_xlabel("parameter")
    ax.legend()
    plt.ylim(-0.1, 0.5)
    plt.show()
    
  

presentationPlot()