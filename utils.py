import numpy as np
# import sys
# sys.path = ["."] + sys.path[2:]

import mitsuba as mi
# import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR, TEMP_DIR

from scipy import stats

# Independent t-test
def t_p(group1, group2):
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Set equal_var=False for Welch’s t-test
    return t_stat, p_value
    # Paired t-test
    # t_stat, p_value = stats.ttest_rel(before, after)

# print("t-statistic:", t_stat)
# print("p-value:", p_value)

def render_result(scenefile, iters):
    scene = mi.load_file(DATA_DIR + scenefile)
    param = mi.traverse(scene)
    #for i in range(iters)
    for i in range(iters):
        print(i)
        id = i * 1
        objs = mi.load_dict({
            'type': 'ply',
            'filename':TEMP_DIR + f"A_iter{id}_csg.ply",
            'bsdf':{
                'type':'roughdielectric',
                'alpha':0.3
                # 'reflectance':{'type':'rgb', 'value':(0.5, 0.8, 0.65)}
            }
        })
        ltemp_ply = mi.traverse(objs)
   
        param['shield.vertex_positions'] = ltemp_ply["vertex_positions"]
        param['shield.faces'] = ltemp_ply["faces"]
        param['shield.vertex_normals'] = ltemp_ply["vertex_normals"]
        param['shield.vertex_texcoords'] = ltemp_ply["vertex_texcoords"]
        # param['shield.faces'] = ltemp_ply["faces"]
        param.update()
        print(f"rendering image {i}")
        image = mi.render(scene, spp=1024)
        dr.eval(image)
        print(f"rendered image {i}")
        mi.util.write_bitmap(TEMP_DIR + f"opt_{i}_csg_constrain_energy.png", image)
        print(f"image wrote")
        del objs, image

# render_result("result.xml", 200)
# sum = 0.0
import matplotlib.pyplot as plt

def get_data(height_ad, range_ad, ad_, init):
    grad_list = []
    std_list = []
    x = []

    # init = (0.01 * -7)

    for h in range(height_ad):
        ad_gs = np.zeros([0,1], dtype=np.float32)
        height = h * 0.0045 + init
        length = 0
        for i in range(range_ad):
            gradients = np.load(f"delta-gradient_{ad_}_{i}_{height:.3f}.npy")
            ad_gs = np.concatenate([ad_gs, gradients])

        ad_gs[np.isnan(ad_gs)] = 0.0
        grad = np.mean(ad_gs)
        std = np.std(ad_gs)

        grad_list.append(grad)
        std_list.append(std)
        x.append(height)
        print(f"gradient at {height} is : ",  grad)


    return grad_list, std_list, x

def load_data(height_ad, range_ad, init):
    # init = (0.01 * -5)
    for h in range(height_ad):
        ad_gs = np.zeros([0,1], dtype=np.float32)
        height = h *0.0045 + init
        length = 0
        group_ad = np.zeros((0,1), dtype=np.float32)
        group_fd = np.zeros((0,1), dtype=np.float32)
        for i in range(range_ad):
            Ad = np.load(f"delta-gradient_ad_{i}_{height:.3f}.npy")
            Fd = np.load(f"delta-gradient_fd_{i}_{height:.3f}.npy")

            group_ad = np.concatenate([group_ad, Ad], axis=0)
            group_fd = np.concatenate([group_fd, Fd], axis=0)

        
        t_value, p_value = t_p(group_ad, group_fd)
        print("t value, p value", t_value, p_value)
            

def plot_test_hemisphere():
    fd = np.load("hemisphere_test_fd.npy")
    ad = np.load("hemisphere_test_ad.npy")
    vd = np.load("hemisphere_test_vd.npy")
    x = np.linspace(0, 0.87, 30)
    plt.plot(x, fd, 'o-', label="finite_difference")
    plt.plot(x, ad, 'x-', label="auto_diff")
    plt.plot(x, vd, '+-', label="analytic")
    plt.legend()
    plt.show()

plot_test_hemisphere()

def plotgradients():
    grad_ad, std_ad, x_ad = get_data(32, 3, "ad", -0.08)
    grad_fd, std_fd, x_fd = get_data(32, 3, "fd", -0.08)
    idx = 5
    plt.errorbar(np.array(x_ad)[:-idx], np.array(grad_ad)[:-idx], yerr=std_ad[:-idx], fmt='o-', label="ad")
    # plt.errorbar(np.array(x_f2d), np.array(grad_f2d), yerr=std_f2d, fmt='+-', label="fd 20000 sample")
    plt.errorbar(np.array(x_fd)[:-idx], np.array(grad_fd)[:-idx], yerr=std_fd[:-idx], fmt='x-', label="fd")
    # plt.errorbar(np.array(x_f1d), np.array(grad_f1d), yerr=std_f1d, fmt='+-', label="fd 40000 sample")
    plt.legend()
    plt.show()

#load_data(17, 3, -0.08)
#plotgradients()