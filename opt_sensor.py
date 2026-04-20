from simulator import *
from area_tally import sample_dir_from_unit_ring
from gpytoolbox import remesh_botsch

from constant import DATA_DIR, TEMP_DIR

dr.set_flag(dr.JitFlag.Debug, True)

def set_parameter(param_dict):
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    # sig_t_shell = [FloatD(0.2), FloatD(0.2), FloatD(0.2)]
    # albedo_shell = [0.8 / 0.9, 0.8 / 0.9, 0.8 / 0.9]

    # # sig_s[g_out, g_in]
    # Phase_shell = np.array([
    #     [0.75, 0.125, 0.125],
    #     [0.00, 0.50, 0.50],
    #     [0.00, 0.00, 1.00],
    # ])

    sig_t_inner = [param_dict["sig_t"], FloatD(1.0), FloatD(1.0)]
    albedo_inner = [0.95, 0.95, 0.95]
    Phase_inner =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    params = {}
    params["sig_t"] = TensorXfD([sig_t_inner])
    params["albedo"] = [albedo_inner]
    params["phase"] = [Phase_inner]

    return params

def load_reactor_scene(params):
    scene_dict = {
        'type': 'scene',
        # 'A': {
        #     'id': 'A',
        #     'type': 'obj',
        #     # 'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
        #     'filename': f"{DATA_DIR}/scene/reactor_outer.obj",
        #     'bsdf': {'type': 'diffuse',
        #             'reflectance': {
        #             'type': 'rgb',
        #             'value': [0.2, 0.25, 0.7]
        #         },
        #     }
        # },
        # 'B': {
        #     'id': 'B',
        #     'type': 'obj',
        #     # 'to_world': mi.ScalarTransform4f().scale([3.0, 3.0, 3.0]),
        #     'filename': f"{DATA_DIR}/scene/reactor_inner.obj",
        #     'bsdf': {'type': 'diffuse',
        #             'reflectance': {
        #             'type': 'rgb',
        #             'value': [0.23, 0.25, 0.7]
        #         },
        #     }
        # },
        'C': {
            'id': 'C',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([2.0, 2.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/sensor_inner.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.6, 0.25, 0.7]
                },
            }
        },
        'D': {
            'id': 'D',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([2.1, 2.1, 1.0]),
            'filename': f"{DATA_DIR}/scene/sensor_outer_shell.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.9, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    
    # router = CSGLeaf(3)
    # rinner = CSGLeaf(2)
    sinner = CSGLeaf(1)
    souter = CSGLeaf(0)
    
    # node2 = CSGNode("difference", CSGNode("difference", shape2, shape0), shape1)
    # node1 = CSGNode("difference", CSGNode("intersection", shape2, shape1), shape0)
    # reactor_shell = CSGNode("difference", router, CSGNode("intersection", rinner, CSGNode("intersection", souter, sinner)))
    # sensor_shell = CSGNode("intersection", router, CSGNode("intersection", rinner, CSGNode("difference", souter, sinner)))
    sensor_shell = CSGNode("difference", souter, sinner)

    scm = SceneMaterial([sensor_shell], [[FloatD(0.9)]], [[0.8]], 2, 3)
    # scm.print_materials()
    # exit(0)
    
    scm.set_material_sigma(TensorXfD(params["sig_t"]))
    scm.set_material_ald(TensorXfD(params["albedo"]))
    phase_function = TensorXf([
        params["phase"]
    ])
    
    scm.set_phase_function(phase_function)

    sensor_dict = {
        'type': 'scene',
         "mesh_emitter": {
            "type": "obj",
            "filename": f"{DATA_DIR}/scene/sensor_inner.obj",

            # Optional transforms (uncomment/tune as needed)
            # "to_world":  mi.ScalarTransform4f.translate([1.0, 1.0, 1.0]) @ mi.ScalarTransform4f.scale([0.05, 0.05, 0.05]),
            

            # You can still attach a BSDF (useful for importance / consistency)
            "bsdf": {"type": "diffuse", "reflectance": 0.0},

            # Turn the mesh into an area emitter
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [30.0, 30.0, 30.0]},
            },
        },
    }
    
    sensor = mi.load_dict(sensor_dict)
    return scene, scm, sensor

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
    keys = params.keys()
    vertices_list = []
    faces_list = []
    for k in keys:
        idx = k.find("vertex_positions")
        if idx > 0:
            name = k[:idx-1]
            V = dr.unravel(mi.Point3f, params[f'{name}.vertex_positions'])
            F = dr.unravel(mi.Vector3i, mi.Int(params[f'{name}.faces']))
            vertices_list.append(V)
            faces_list.append(F)
    # print(vertices_list, faces_list)
    return vertices_list, faces_list

def value_range_loss(opt):
    height_1 = opt["offsetB"]
    height_2 = 0.4
    difference = height_2 - height_1
    loss = dr.sum(dr.select(height_1 < 0.0, (0.0 - height_1) * (0.0 - height_1), 0.0))
    loss += dr.select(difference < 0.0, (0.0 - difference) * (0.0 - difference), 0.0)
    return loss
    
def compute_range_loss(opt):
    height_1 = opt["BitmapTextureImpl.data"]
    height_2 = opt["BitmapTextureImpl_1.data"]
    difference = height_2 - height_1
    loss = dr.sum(dr.select(height_1 < 0.0, (0.0 - height_1) * (0.0 - height_1), 0.0))
    loss += dr.select(difference < 0.0, (0.0 - difference) * (0.0 - difference), 0.0)
    return loss

def compute_volume(v, faces):
    # dr.gather(mi.Point3f, v, faces.x, active)
    p0 = dr.gather(mi.Point3f, v, faces.x, True)
    p1 = dr.gather(mi.Point3f, v, faces.y, True)
    p2 = dr.gather(mi.Point3f, v, faces.z, True)

    delta_1 = p1 - p0
    delta_2 = p2 - p0
    corss_delta12 = dr.cross(delta_1, delta_2)
    
    volume = dr.sum((p0.y + p1.y + p2.y) * corss_delta12.y) / 6.0
    
    return volume

def compute_differentiable_csg_volume(params, key1, key2):
    # if csg_node.op != "difference":
    #     raise ValueError("Only CSG difference nodes are supported.")


    outer_mesh_v = dr.unravel(mi.Point3f, params[f'{key1}.vertex_positions'])
    outer_mesh_f = dr.unravel(mi.Vector3i, mi.Int(params[f'{key1}.faces']))
    inner_mesh_v = dr.unravel(mi.Point3f, params[f'{key2}.vertex_positions'])
    inner_mesh_f =  dr.unravel(mi.Vector3i, mi.Int(params[f'{key2}.faces']))

    # help(outer_mesh)
    # exit(0)

    # help(inner_mesh)
    # Explicitly enable gradients for both meshes
    # outer_vertex_buf = outer_mesh.vertex_positions_buffer()
    # inner_vertex_buf = inner_mesh.vertex_positions_buffer()

    # dr.enable_grad(outer_vertex_buf)
    # dr.enable_grad(inner_vertex_buf)

    # Compute volumes with gradient tracking
    vol_outer = compute_volume(outer_mesh_v, outer_mesh_f)
    vol_inner = compute_volume(inner_mesh_v, inner_mesh_f)

    return vol_outer - vol_inner

def opt_energy_dependent_two_layers(iteration_count, nuetron_number):
    scene, scm = load_scene_node_energy_dependent()
    params = mi.traverse(scene)

    # params["BitmapTextureImpl.data"] += 0.1
    # params["BitmapTextureImpl_1.data"] += 0.3
    # dr.enable_grad(params["BitmapTextureImpl.data"])
    # dr.enable_grad(params["BitmapTextureImpl_1.data"])

    offsetB = FloatD(0.1)
    offsetC = FloatD(0.4)
    dr.enable_grad(offsetB)
    dr.enable_grad(offsetC)

    opt = mi.ad.Adam(lr=0.001)
    opt["offsetB"] = offsetB
    dr.make_opaque(opt["offsetB"])
    # dr.make_opaque(opt["offsetC"])
    
    # opt["BitmapTextureImpl.data"] = params["BitmapTextureImpl.data"]
    # opt["BitmapTextureImpl_1.data"] = params["BitmapTextureImpl_1.data"]
    # params.update(opt)

    # offset A along the normal using a texture map
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    aN = dr.unravel(mi.Vector3f, params['A.vertex_normals'])
    aUV = dr.unravel(mi.Vector2f, params['A.vertex_texcoords'])
    dr.enable_grad(params[f'B.vertex_positions'])
    dr.enable_grad(params[f'C.vertex_positions'])

    errors = []
    volumes = []
    rng = mi.PCG32(size=nuetron_number)
    dr.eval(rng)
    # with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
    for it in range(iteration_count):
        if it % 1 == 0:
            shapes = scene.shapes()
            shapes[1].write_ply(TEMP_DIR + f"B_iter{it}_csg.ply")
            shapes[2].write_ply(TEMP_DIR + f"C_iter{it}_csg.ply")

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
        range_loss = value_range_loss(opt)

        # tensorxf_0 = TensorXfD(opt["BitmapTextureImpl.data"])
        # tensorxf_1 = TensorXfD(opt["BitmapTextureImpl_1.data"])
        # heights_map_0 = mi.Texture2f(tensorxf_0, wrap_mode=dr.WrapMode.Repeat)
        # heights_map_1 = mi.Texture2f(tensorxf_1, wrap_mode=dr.WrapMode.Repeat)
        # dr.make_opaque(heights_map_0)
        # dr.make_opaque(heights_map_1)
        # mi.util.write_bitmap(TEMP_DIR + f"opt_height_0_{it}.exr",  mi.Bitmap(heights_map_0.tensor()))
        # mi.util.write_bitmap(TEMP_DIR + f"opt_height_1_{it}.exr",  mi.Bitmap(heights_map_1.tensor()))


        # offsets_0 = heights_map_0.eval_cubic(aUV)[0]
        # offsets_1 = heights_map_1.eval_cubic(aUV)[0]
        offsets_0 = FloatD(opt['offsetB'])
        # offsets_1 = FloatD(opt['offsetC'])
        EPS = 0.00001
        offsets_0 = dr.select(offsets_0 < 0.0, EPS, offsets_0)
        # offsets_1 = dr.select(offsets_1 < 0.0, EPS, offsets_1)
        offseted_0V = aV + aN * offsets_0
        offseted_1V = aV + aN * offsetC
        params['B.vertex_positions'] = dr.ravel(offseted_0V)
        params['C.vertex_positions'] = dr.ravel(offseted_1V)
        dr.make_opaque(params['B.vertex_positions'])
        dr.make_opaque(params['C.vertex_positions'])
        params.update()

        vertices_list, faces_list = get_vertices_face_list(params)

        
        # ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
        ray_vec, ray_origin = sample_dir_origin_from_ring_nosym(rng, 1.0)
        # ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
        ray_current = mi.Ray3f(ray_origin, ray_vec)

        
        # scene, rng, scm, vertices_list, faces_list, ray_current,  number_of_energy_group, reparam
        energy = render_nuetron_in_csg_shape_energy_dependent(SceneInfo(scene, rng, scm, vertices_list, faces_list), ray_current, True)
        # level_0 = dr.gather(FloatD, energy, UInt(0))
        level_0 = dr.sum(energy)
        dr.eval(level_0)
        
        # if it == 30:
        #     hist = dr.kernel_history()
        #     dump_history(hist, "level_0_history.txt")
        #     exit(0)
        # dr.sum(energy) / nuetron_number

        # if it == 0:
            # volume_init_0 = dr.detach(compute_differentiable_csg_volume(params, 'B', 'A'))
            # volume_init_1 = dr.detach(compute_differentiable_csg_volume(params, 'C', 'B'))
            # energy_init = dr.detach(energy)
        volume0 = compute_differentiable_csg_volume(params, 'B', 'A')
        volume1 = compute_differentiable_csg_volume(params, 'C', 'B')
        
        # volume_loss = dr.power(volume0 + volume1 - volume_init_1 - volume_init_0, 2.0)
        # energy_loss = dr.power(energy - energy_init, 2.0)
        energy_loss = level_0
        # dr.select(energy > 0.5, dr.power(energy - 0.5, 2.0),  0.0)
        
        loss = energy_loss + range_loss 
        # + volume_loss
        dr.backward(loss)
        errors.append(energy.numpy())
        # volumes.append([volume0.numpy(), volume1.numpy()])
        # print("\n\nvalue before step", opt["offset.data"].numpy())
        # print("\n\ngradient", dr.grad(opt["offset.data"]).numpy())
        opt.step()
        # print("\n\nvalue after step", opt["offset.data"].numpy())
        # exit(0)



        # heights_map = mi.Bitmap(heights_map.tensor())
        # print(height_bitmap)
        
        # volume loss = {volume_loss.numpy()[0]:6f},
        print(f"Iteration {it:02d}: energy = {energy_loss.numpy()[0]:6f},  volume  0 = {volume0.numpy()[0]:6f} volume  1 = {volume1.numpy()[0]:6f}")  #end='\r'
        
        del energy, loss, range_loss, energy_loss
        # volume_loss, 

        # dr.kernel_history_clear()
        # dr.flush_malloc_cache()
        # dr.malloc_clear_statistics()
        
        # dr.flush_kernel_cache()
        # if it % 10 == 0:
            # dr.flush_malloc_cache()

    np.save(TEMP_DIR+"enery_csg_ce_ed.npy", np.array(errors))
    np.save(TEMP_DIR+"volume_csg_ce_ed.npy", np.array(volumes))
    print('\nOptimization complete.')


def point_light(num_neutrons, rng):
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    ray_origin.x -= 4.0
    random_cos = sample_float_32(rng) * 2.0 - 1.0
    random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    phi = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec.x += random_sin * dr.cos(phi)
    ray_vec.y += random_sin * dr.sin(phi)
    ray_vec.z += random_cos
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    return ray_current



# @dr.wrap(source='torch', target='drjit')
def opt_mesh_volume_constraints(iteration_count, name, key, nuetron_number, param_dict, remesh=5000):
    material_params = set_parameter(param_dict)
    scene, scm, sensor = load_reactor_scene(material_params)
    rng = mi.PCG32(size=nuetron_number, initstate=1994)
    dr.make_opaque(rng)

    params = mi.traverse(scene)
    lambda_ = 35
    ls = mi.ad.LargeSteps(params[f'{key}.vertex_positions'], params[f'{key}.faces'], lambda_)

    opt = mi.ad.Adam(lr=0.005)
    opt['u'] = ls.to_differential(params[f'{key}.vertex_positions'])
    
    dr.enable_grad(params[f'{key}.vertex_positions'])
    v_np, f_np, target_length = get_vf(params, key)
    
    

    params.update(opt)
    errors = []
    volumes = []
    for it in range(iteration_count):
        if it % 10 == 0:
            shapes = scene.shapes()
            shapes[0].write_ply(TEMP_DIR + f"{name}_{key}_iter{it}.ply")
            np.save(TEMP_DIR+f"{name}_energy.npy", np.array(errors))
            np.save(TEMP_DIR+f"{name}_volumes.npy", np.array(volumes))

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
    
        vertices_list, faces_list = get_vertices_face_list(params)
        ray_current = point_light(nuetron_number, rng)
        sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list, sensor)

        volume = compute_volume(vertices_list[0], faces_list[0])
        volume_loss = dr.abs(1.51 - volume)

        Etot = render_neutron_in_csg_shape_energy_dependent_with_sensor(sceneinfo, ray_current, True)
        energy_loss = 1.0 - dr.gather(type(Etot), Etot, UInt(1))

        loss = energy_loss + 0.1 * volume_loss
        dr.eval()
        dr.backward(loss)
        opt.step()

        print(f"Iteration {it:02d}: energy = {energy_loss.numpy()[0]:6f}, volume = {volume.numpy()[0]:6f}")  #end='\r'
        errors.append(energy_loss.numpy())
        volumes.append(volume.numpy())
        del Etot, loss, volume_loss, energy_loss

    np.save(TEMP_DIR+f"{name}_energy.npy", np.array(errors))
    np.save(TEMP_DIR+f"{name}_volumes.npy", np.array(volumes))
    print('\nOptimization complete.')


# @dr.wrap(source='torch', target='drjit')
def opt_mesh(iteration_count, name, key, nuetron_number, param_dict, remesh=5000):
    material_params = set_parameter(param_dict)
    scene, scm, sensor = load_reactor_scene(material_params)
    rng = mi.PCG32(size=nuetron_number, initstate=1994)
    dr.make_opaque(rng)

    params = mi.traverse(scene)
    lambda_ = 35
    ls = mi.ad.LargeSteps(params[f'{key}.vertex_positions'], params[f'{key}.faces'], lambda_)

    opt = mi.ad.Adam(lr=0.005)
    opt['u'] = ls.to_differential(params[f'{key}.vertex_positions'])
    
    dr.enable_grad(params[f'{key}.vertex_positions'])
    v_np, f_np, target_length = get_vf(params, key)
    
    params.update(opt)
    errors = []
    volumes = []
    for it in range(iteration_count):
        if it % 10 == 0:
            shapes = scene.shapes()
            shapes[0].write_ply(TEMP_DIR + f"{name}_{key}_iter{it}.ply")
            np.save(TEMP_DIR+"energy.npy", np.array(errors))

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
    
        vertices_list, faces_list = get_vertices_face_list(params)
        ray_current = point_light(nuetron_number, rng)
        sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list, sensor)
        Etot = render_neutron_in_csg_shape_energy_dependent_with_sensor(sceneinfo, ray_current, True)
        loss = 1.0 - dr.gather(type(Etot), Etot, UInt(1))
        dr.eval()
        dr.backward(loss)
        opt.step()

        print(f"Iteration {it:02d}: energy = {loss.numpy()[0]:6f}")  #end='\r'
        errors.append(loss.numpy())
        del Etot, loss

    np.save(TEMP_DIR+"energy.npy", np.array(errors))
    print('\nOptimization complete.')



# @dr.wrap(source='torch', target='drjit')
def opt(iteration_count, key, nuetron_number, param_dict):
     # param_dict = {"sig_t": FloatD(2.0)}
    material_params = set_parameter(param_dict)
    scene, scm, sensor = load_reactor_scene(material_params)
    # dr.make_opaque(seed)
    rng = mi.PCG32(size=nuetron_number, initstate=1994)
    dr.make_opaque(rng)
    
    # create optimizer
    Radius = FloatD(param_dict["geo"])
    dr.enable_grad(Radius)
    opt = mi.ad.Adam(lr=0.02)
    opt["radius"] = Radius
    dr.make_opaque(opt["radius"])

    
    params = mi.traverse(scene)
    dV = dr.unravel(mi.Point3f, params['D.vertex_positions'])

    # radius = param_dict["geo"] 
    # dr.make_opaque(radius)
    params['D.vertex_positions'] = dr.ravel(dV)
    dr.enable_grad(params['D.vertex_positions'])
    params.update()
    
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Vc = dr.unravel(mi.Point3f, params['C.vertex_positions'])
    Vd = dr.unravel(mi.Point3f, params['D.vertex_positions'])
    
    # Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    # Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))
    Fc = dr.unravel(mi.Vector3i, mi.Int(params[f'C.faces']))
    Fd = dr.unravel(mi.Vector3i, mi.Int(params[f'D.faces']))
    
    errors = []
    rs_value = []

    for it in range(iteration_count):
        if it % 10 == 0:
            shapes = scene.shapes()
            shapes[0].write_ply(TEMP_DIR + f"{key}_D_iter{it}_csg.ply")
            shapes[1].write_ply(TEMP_DIR + f"{key}_C_iter{it}_csg.ply")
            np.save(TEMP_DIR+f"{key}_loss.npy", np.array(errors))
            np.save(TEMP_DIR+f"{key}_rs.npy", np.array(rs_value))

        rscale = FloatD(opt['radius'])
        EPS = 1.00001
        rscale = dr.select(rscale < 1.0, EPS, rscale)
        
        scaleV = dr.copy(dV)
        scaleV.x *= rscale
        scaleV.y *= rscale
        params['D.vertex_positions'] = dr.ravel(scaleV)
        dr.make_opaque(params['D.vertex_positions'])
        params.update()

        vertices_list, faces_list = get_vertices_face_list(params)
        ray_current = point_light(nuetron_number, rng)
        sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list, sensor)
        Etot = render_neutron_in_csg_shape_energy_dependent_with_sensor(sceneinfo, ray_current, True)
        level_1 = dr.gather(FloatD, Etot, UInt(1))
        dr.eval(level_1)
        energy_loss = 1 - level_1
        
        # loss = energy_loss 
        dr.backward(energy_loss)
        errors.append(energy_loss.numpy())
        rs_value.append(rscale.numpy())
        opt.step()
        
        print(f"Iteration {it:02d}: energy = {energy_loss.numpy()[0]:6f}, radius: {rscale.numpy()[0]:6f}")  #end='\r'
        
        del energy_loss, Etot, level_1
        
        # dr.flush_malloc_cache()

    np.save(TEMP_DIR+f"{key}_loss.npy", np.array(errors))
    np.save(TEMP_DIR+f"{key}_rs.npy", np.array(rs_value))
    print('\nOptimization complete.')
    

def plot_energy_and_volume(values, sticker):
    # dr.set_log_level(dr.LogLevel.Debug) 
    # mi.LogLevel = "Debug"
    mi.DEBUG = True
    length = values.shape[0]

    for i in range(length):
        print(i)
        es = values[:i]
        # /vs = volumes[:i]

        plt.plot(es, label=sticker)
        plt.xlim(0,200)
        plt.ylim(0.640,0.648)
        plt.legend()
        plt.savefig(TEMP_DIR + f"{sticker}_{i}.png")
        plt.close()

def forward_capture(num_neutrons, seed, param_dict):
    # param_dict = {"sig_t": FloatD(2.0)}
    material_params = set_parameter(param_dict)
    scene, scm, sensor = load_reactor_scene(material_params)
    # seed = 1994
    # num_neutrons = 1000000

    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    # param_dict["geo"] = 1.1 # scael sensor shell x and y by this amount

    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    ray_origin.x -= 4.0
    random_cos = sample_float_32(rng) * 2.0 - 1.0
    random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    phi = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec.x += random_sin * dr.cos(phi)
    ray_vec.y += random_sin * dr.sin(phi)
    ray_vec.z += random_cos

    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    
    params = mi.traverse(scene)
    dV = dr.unravel(mi.Point3f, params['D.vertex_positions'])

    radius = param_dict["geo"] 
    dr.make_opaque(radius)
    dV.y *= radius
    dV.x *= radius
    params['D.vertex_positions'] = dr.ravel(dV)
    dr.enable_grad(params['D.vertex_positions'])
    params.update()
    
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Vc = dr.unravel(mi.Point3f, params['C.vertex_positions'])
    Vd = dr.unravel(mi.Point3f, params['D.vertex_positions'])
    
    # Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    # Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))
    Fc = dr.unravel(mi.Vector3i, mi.Int(params[f'C.faces']))
    Fd = dr.unravel(mi.Vector3i, mi.Int(params[f'D.faces']))

    vertices_list = [Vd, Vc]
    faces_list = [Fd, Fc]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list, sensor)
    Etot = render_neutron_in_csg_shape_energy_dependent_with_sensor(sceneinfo, ray_current, True)
    dr.eval()

    return Etot

def scan_radius():
    param_range = [1.0, 7.0]
    data_points = 50
    step_size = (param_range[1] - param_range[0]) / data_points
    r = param_range[0]
    energy = []
    rs = []
    N = 40
    for i in range(data_points+1):
        print("iteration i", i)
        param_dict = {
            "geo": FloatD(r),
            "sig_t": FloatD(2.0)
        }
        cur_energy = []
        for j in range(N):
            Etot = forward_capture(4000000, 1994 + j, param_dict)
            energy_in_middle = Etot.numpy()[1]
            cur_energy.append(energy_in_middle)
            del Etot
            dr.flush_kernel_cache()
        avg_energy = np.mean(np.array(cur_energy))
        energy.append(avg_energy)
        rs.append(0.1 * 2.5 * r)
        r += step_size
        print("value: ", avg_energy)
        
    # print(energy)
    np.save("sensor_opt_value.npy", np.array(energy))
    np.save("sensor_opt_rs.npy", np.array(rs))

if __name__ == "__main__":
    param_dict = {
        "geo": 1.1,
        "sig_t": FloatD(2.0)
    }
    # scan_radius()
    id = 3
    # opt_mesh(5000, f"opt_sensor_mesh_{id}", "D", 100000, param_dict)
    opt_mesh_volume_constraints(5000, f"opt_sensor_mesh_vol_{id}", "D", 100000, param_dict)
    # opt(10000, f"opt_sensor_{id}", 2000000, param_dict)
    
    
    # opt_energy_dependent_two_layers(200, 10000)
# opt(200, "A", 10000)
#values = np.load(TEMP_DIR + "enery_csg_2c.npy")
#plot_energy_and_volume(values, "energy_with_volume_constraints")