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
                'type':'diffuse',
                'reflectance':{'type':'rgb', 'value':(0.5, 0.8, 0.65)}
            }
        })
        ltemp_ply = mi.traverse(objs)
        # print(ltemp_ply)
        # exit(0)
   
        param['shield.vertex_positions'] = ltemp_ply["vertex_positions"]
        param['shield.faces'] = ltemp_ply["faces"]
        param['shield.vertex_normals'] = ltemp_ply["vertex_normals"]
        param['shield.vertex_texcoords'] = ltemp_ply["vertex_texcoords"]
        # param['shield.faces'] = ltemp_ply["faces"]
        param.update()
        image = mi.render(scene, spp=64)
        mi.util.write_bitmap(TEMP_DIR + f"opt_{i}_csg.png", image)
        del objs

# render_result("result.xml", 75)
sum = 0.0
length = 0
for i in range(9):
    gradients = np.load(f"gradient_fd_{i}.npy")
    print(gradients)
    sum += np.sum(gradients)
    length += gradients.shape[0]
 
print("avg", sum / length)
