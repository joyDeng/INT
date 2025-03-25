from area_tally import *
import torch
from gpytoolbox import remesh_botsch

from constant import DATA_DIR, TEMP_DIR

def mse(current, target):
    return dr.sqr(current - target)

def get_vf(params, key):
    v_np = params[f'{key}.vertex_positions'].numpy().reshape((-1,3)).astype(np.float64)
    f_np = params[f'{key}.faces'].numpy().reshape((-1,3))

    l0 = np.linalg.norm(v_np[f_np[:,0]] - v_np[f_np[:,1]], axis=1)
    l1 = np.linalg.norm(v_np[f_np[:,1]] - v_np[f_np[:,2]], axis=1)
    l2 = np.linalg.norm(v_np[f_np[:,2]] - v_np[f_np[:,0]], axis=1)
    target_l = np.mean([l0, l1, l2])
    return v_np, f_np, target_l

def get_vertices_face_list(params):
    # watch out for the order of the keys here
    keys = params.keys()
    vertices_list = []
    faces_list = []
    for k in keys:
        idx = k.find("vertex_positions")
        if idx > 0:
            name = k[:idx-1]
            # print(name)
            # exit(0)
            V = dr.unravel(mi.Point3f, params[f'{name}.vertex_positions'])
            F = dr.unravel(mi.Vector3i, params[f'{name}.faces'])
            vertices_list.append(V)
            faces_list.append(F)
    # print(vertices_list, faces_list)
    return vertices_list, faces_list
    

# @dr.wrap(source='torch', target='drjit')
def opt(iteration_count, key, remesh):
    # print("optimization")
    scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)
    
    # initialize the offset to 0.1
    params["offset.data"] += 0.1
    dr.enable_grad(params["offset.data"])

    opt = mi.ad.Adam(lr=0.001)
    opt["offset.data"] = params["offset.data"]
    params.update(opt)

    # offset A along the normal using a texture map
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    aN = dr.unravel(mi.Vector3f, params['A.vertex_normals'])
    aUV = dr.unravel(mi.Vector2f, params['A.vertex_texcoords'])
    dr.enable_grad(params[f'A.vertex_positions'])

    v_np, f_np, target_length = get_vf(params, key)
    
    

    errors = []
    volumes = []
    for it in range(iteration_count):
        rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=it, initseq=it*2)

        if it % 1 == 0:
            shapes = scene.shapes()
            shapes[0].write_ply(TEMP_DIR + f"{key}_iter{it}_csg.ply")

        # if (it % remesh == 1) and (it > 1) and remesh > 0:
        #     print("try to reparamerize")
        #     v_np, f_np, avglength = get_vf(params, key)

        #     if (it // remesh) in [3, 5]:
        #         updatelength =  avglength * 0.5
        #     else:
        #         updatelength = avglength

        #     v_new, f_new = remesh_botsch(v_np, f_np, i=5, h=updatelength, project=True)

        #     params[f'{key}.vertex_positions'] =  mi.Float(v_new.flatten().astype(np.float32))
        #     params[f'{key}.faces'] = mi.Int(f_new.flatten())
        #     params.update()
            
            # print(help(opt))
            # del opt
            # ls = mi.ad.LargeSteps(params[f'{key}.vertex_positions'], params[f'{key}.faces'], lambda_)
            
            # # exit(0)
            # # print(target_length * 0.5)
            # # exit(0)
            # opt = mi.ad.Adam(lr=updatelength * 0.07)
            # dr.enable_grad(params[f'{key}.vertex_positions'])
            # params.update()
            # opt['u'] = ls.to_differential(params[f'{key}.vertex_positions'])
            # params.update(opt)

        # compute the vertex of mesh A from the distplacement texture
        tensorxf = TensorXfD(opt["offset.data"])
        heights_map = mi.Texture2f(tensorxf)
        mi.util.write_bitmap(TEMP_DIR + f"opt_height_{it}.exr",  mi.Bitmap(heights_map.tensor()))
        offsets = heights_map.eval_cubic(aUV)[0]
        offsetedV = aV + aN * offsets
        params['A.vertex_positions'] = dr.ravel(offsetedV)
        params.update()

        vertices_list, faces_list = get_vertices_face_list(params)

        ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
        # ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
        ray_current = mi.Ray3f(ray_origin, ray_vec)

        
        energy = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, True)
        loss = energy 
        dr.backward(loss)
        
        # print("\n\nvalue before step", opt["offset.data"].numpy())
        # print("\n\ngradient", dr.grad(opt["offset.data"]).numpy())
        opt.step()
        # print("\n\nvalue after step", opt["offset.data"].numpy())
        # exit(0)



        # heights_map = mi.Bitmap(heights_map.tensor())
        # print(height_bitmap)
        

        print(f"Iteration {it:02d}: energy = {energy.numpy()[0]:6f}")  #end='\r'
        
        del energy, loss
        torch.cuda.empty_cache()

    np.save(TEMP_DIR+"enery_csg.npy", np.array(errors))
    print('\nOptimization complete.')


opt(70, "A", 10)