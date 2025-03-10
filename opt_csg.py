from area_tally import *

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
    

def opt(iteration_count, key, remesh):
    
    scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)
    # help(params)
    # print("items", params.items())
    # print("params", params.keys())
    
    # exit(0)

    
    lambda_ = 15
    ls = mi.ad.LargeSteps(params[f'{key}.vertex_positions'], params[f'{key}.faces'], lambda_)
    
    
    # set up tally that exit the shape
  
    
    # generate rays
    # ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
    
    
    # load scene
    # temp example, a scene with torus
    scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)
    # get scene parameter for optimization
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # Fa = dr.unravel(mi.Vector3i, params['A.faces'])
    # Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    # Fb = dr.unravel(mi.Vector3i, params['B.faces'])
    # Va.y = Va.y + height

    # params['A.vertex_positions'] = dr.ravel(Va)
    # params['B.vertex_positions'] = dr.ravel(Vb)
    # dr.enable_grad(params['A.vertex_positions'])
    # dr.enable_grad(params['B.vertex_positions'])
    # params.update()

    # vertices_list = [Va, Vb]
    # faces_list = [Fa, Fb]



    opt = mi.ad.Adam(lr=0.005)
    opt['u'] = ls.to_differential(params[f'{key}.vertex_positions'])
    
    # F = dr.unravel(mi.Vector3i, params[f'{key}.faces'])
    # V = dr.unravel(mi.Point3f, params[f'{key}.vertex_positions'])

    # init_volume = dr.detach(compute_volume(V, F))

    # negative_delta_volme = init_volume_abs - init_volume
    # print(init_volume.numpy())
    # exit(0)
    dr.enable_grad(params[f'{key}.vertex_positions'])
    v_np, f_np, target_length = get_vf(params, key)
    
    params.update(opt)
    errors = []
    volumes = []
    for it in range(iteration_count):
        rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=it, initseq=it*2)

        if it % 1 == 0:
            shapes = scene.shapes()
            shapes[0].write_ply(TEMP_DIR + f"{key}_iter{it}_csg.ply")

        if (it % remesh == 1) and (it > 1) and remesh > 0:
            print("try to reparamerize")
            v_np, f_np, avglength = get_vf(params, key)

            if (it // remesh) in [3, 5]:
                updatelength =  avglength * 0.5
            else:
                updatelength = avglength

            v_new, f_new = remesh_botsch(v_np, f_np, i=5, h=updatelength, project=True)

            params[f'{key}.vertex_positions'] =  mi.Float(v_new.flatten().astype(np.float32))
            params[f'{key}.faces'] = mi.Int(f_new.flatten())
            params.update()
            
            # print(help(opt))
            del opt
            ls = mi.ad.LargeSteps(params[f'{key}.vertex_positions'], params[f'{key}.faces'], lambda_)
            
            # exit(0)
            # print(target_length * 0.5)
            # exit(0)
            opt = mi.ad.Adam(lr=updatelength * 0.07)
            dr.enable_grad(params[f'{key}.vertex_positions'])
            params.update()
            opt['u'] = ls.to_differential(params[f'{key}.vertex_positions'])
            params.update(opt)

        params[f'{key}.vertex_positions'] = ls.from_differential(opt['u'])
        params.update()
        
        # exit(0)

        vertices_list, faces_list = get_vertices_face_list(params)
        # V = dr.unravel(mi.Point3f, params[f'{key}.vertex_positions'])
        # F = dr.unravel(mi.Vector3i, params[f'{key}.faces'])

        # vtemp = compute_volume(V, F)
        # range_loss = compute_range_loss(V, F)
        # volume_mse = mse(vtemp, init_volume)
        # negative_volume_mse = mse(dvx, negative_delta_volme)
        # print("render nuetron in csg shape")

        ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
        ray_current = mi.Ray3f(ray_origin, ray_vec)

        energy = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, True)
        # calculate_tally_energy_light_connection(scene, V, F, TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A, it, True)

        loss = energy 
        # + range_loss * 2.0 + volume_mse * 5.0 
        dr.backward(loss)
        opt.step()

        print(f"Iteration {it:02d}: energy = {energy.numpy()[0]:6f}")  #end='\r'
        # errors.append(energy.numpy())
        # volumes.append(vtemp.numpy())
        del energy, loss
        # volume_mse, range_loss, vtemp

    np.save(TEMP_DIR+"enery_csg.npy", np.array(errors))
    # np.save(TEMP_DIR+"volumes.npy", np.array(volumes))
    print('\nOptimization complete.')

opt(100, "A", 10)