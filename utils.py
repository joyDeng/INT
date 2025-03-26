import numpy as np
import sys
sys.path = ["."] + sys.path[2:]

import mitsuba as mi
# import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR, TEMP_DIR

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
        mi.util.write_bitmap(TEMP_DIR + f"opt_{i}_csg_wo_volume.png", image)
        print(f"image wrote")
        del objs, image

render_result("result.xml", 200)
# sum = 0.0
import matplotlib.pyplot as plt

def get_data(height_ad, range_ad, ad_):
    grad_list = []
    std_list = []
    x = []

    for h in range(height_ad):
        ad_gs = np.zeros([0,1], dtype=np.float32)
        height = h * 0.005
        length = 0
        for i in range(range_ad):
            gradients = np.load(f"gradient_{ad_}_{i}_{height}.npy")
            ad_gs = np.concatenate([ad_gs, gradients])
        print(ad_gs.shape)
        grad = np.mean(ad_gs)
        std = np.std(ad_gs)
        print(std)
        grad_list.append(grad)
        std_list.append(std)
        x.append(height)
        print(f"gradient at {height} is : ",  grad)

    return grad_list, std_list, x

# grad_ad, std_ad, x_ad = get_data(4, 2, "ad")
# grad_fd, std_fd, x_fd = get_data(4, 4, "fd")
# grad_f1d, std_f1d, x_f1d = get_data(4, 3, "fd")
# grad_f2d, std_f2d, x_f2d = get_data(4, 2, "fd")
# plt.errorbar(np.array(x_ad), np.array(grad_ad), yerr=std_ad, fmt='o-', label="ad")
# plt.errorbar(np.array(x_f2d), np.array(grad_f2d), yerr=std_f2d, fmt='+-', label="fd 20000 sample")
# plt.errorbar(np.array(x_fd), np.array(grad_fd), yerr=std_fd, fmt='x-', label="fd 30000 sample")
# plt.errorbar(np.array(x_f1d), np.array(grad_f1d), yerr=std_f1d, fmt='+-', label="fd 40000 sample")
# plt.legend()
# plt.show()
