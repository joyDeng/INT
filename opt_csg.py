from area_tally import *
# import torch
from gpytoolbox import remesh_botsch

from constant import DATA_DIR, TEMP_DIR

dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_log_level(dr.LogLevel.Info)

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
    # print(params)
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

# def differentiable_mesh_volume(mesh):
#     # Access raw vertex buffer and enable gradients
#     vertex_buf = mesh.vertex_positions_buffer()
#     dr.enable_grad(vertex_buf)

#     # Get faces and vertices as AD-tracked arrays
#     faces = mesh.faces_buffer()
#     vertices = dr.unravel(mi.Point3f, vertex_buf)

#     # Gather triangle vertices
#     face_indices = dr.unravel(mi.Vector3u, faces)
#     v0 = dr.gather(mi.Point3f, vertices, face_indices.x)
#     v1 = dr.gather(mi.Point3f, vertices, face_indices.y)
#     v2 = dr.gather(mi.Point3f, vertices, face_indices.z)

#     # Compute signed volume with AD tracking
#     return dr.sum(dr.dot(v0, dr.cross(v1, v2))) / 6.0

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
    # opt["offsetC"] = offsetC
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


# @dr.wrap(source='torch', target='drjit')
def opt(iteration_count, key, nuetron_number):
    scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)
    
    # initialize the offset to 0.1
    params["BitmapTextureImpl.data"] += 0.1
    dr.enable_grad(params["BitmapTextureImpl.data"])

    opt = mi.ad.Adam(lr=0.001)
    opt["BitmapTextureImpl.data"] = params["BitmapTextureImpl.data"]
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
        rng = mi.PCG32(size=nuetron_number, initstate=it+1, initseq=(it+1)*2)
        dr.make_opaque(rng)

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
        range_loss = compute_range_loss(opt)

        tensorxf = TensorXfD(opt["BitmapTextureImpl.data"])
        heights_map = mi.Texture2f(tensorxf, wrap_mode=dr.WrapMode.Repeat)
        mi.util.write_bitmap(TEMP_DIR + f"opt_height_{it}.exr",  mi.Bitmap(heights_map.tensor()))


        offsets = heights_map.eval_cubic(aUV)[0]
        EPS = 0.00001
        offsets = dr.select(offsets < 0.0, EPS, offsets)
        offsetedV = aV + aN * offsets
        params['A.vertex_positions'] = dr.ravel(offsetedV)
        params.update()

        vertices_list, faces_list = get_vertices_face_list(params)

        
        # ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
        ray_vec, ray_origin = sample_dir_origin_from_ring_nosym(rng, 1.0)
        # ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
        ray_current = mi.Ray3f(ray_origin, ray_vec)

        
        energy = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, True)
        if it == 0:
            volume_init = dr.detach(compute_differentiable_csg_volume(params, scm.csg_node_list[0]))
            energy_init = dr.detach(energy)
        volume = compute_differentiable_csg_volume(params, scm.csg_node_list[0])
        # volume_loss = dr.power(volume - volume_init, 2.0)
        volume_loss = dr.power(volume - volume_init, 2.0)
        # energy_loss = dr.power(energy - energy_init, 2.0)
        energy_loss = energy
        # dr.select(energy > 0.5, dr.power(energy - 0.5, 2.0),  0.0)
        
        loss = energy_loss + range_loss + volume_loss
        dr.backward(loss)
        errors.append(energy.numpy())
        volumes.append(volume.numpy())
        # print("\n\nvalue before step", opt["offset.data"].numpy())
        # print("\n\ngradient", dr.grad(opt["offset.data"]).numpy())
        opt.step()
        # print("\n\nvalue after step", opt["offset.data"].numpy())
        # exit(0)



        # heights_map = mi.Bitmap(heights_map.tensor())
        # print(height_bitmap)
        

        print(f"Iteration {it:02d}: energy = {energy_loss.numpy()[0]:6f}, volume = {volume_loss.numpy()[0]:6f}")  #end='\r'
        
        del energy, loss, range_loss, volume_loss, energy_loss

        # dr.kernel_history_clear()
        # dr.flush_malloc_cache()
        # dr.malloc_clear_statistics()
        dr.flush_kernel_cache()
        # if it % 10 == 0:
            # dr.flush_malloc_cache()

    np.save(TEMP_DIR+"enery_csg_ce.npy", np.array(errors))
    np.save(TEMP_DIR+"volume_csg_ce.npy", np.array(volumes))
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


if __name__ == "__main__":
    opt_energy_dependent_two_layers(200, 10000)
# opt(200, "A", 10000)
#values = np.load(TEMP_DIR + "enery_csg_2c.npy")
#plot_energy_and_volume(values, "energy_with_volume_constraints")