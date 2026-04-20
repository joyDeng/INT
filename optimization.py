import sys
sys.path = ["."] + sys.path[2:]

# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64
from drjit.cuda.ad import Float as FloatD
import numpy as np
import random

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR, TEMP_DIR

from area_tally import sample_dir_from_unit_sphere, sample_direction_hg, hg, recomputeIntersection, cross_section_nor, sample_distance

from gpytoolbox import remesh_botsch

NUMBER_NEUTRONS = 100000
MAX_BOUNCE = 2
TOT_CROSS_SECTION_T = 2.5
TOT_CROSS_SECTION_A = 0.15
AVERAGE_COS = mi.Float(0.0)
FOUR_PI = 4.0 * 3.141592653
INV_FOUR_PI = 1.0 / FOUR_PI
PI = 3.141592653
BOUNCE_RECORD = -1
# PARAM = "y offset k=2"

# def bbox_penalty(v, faces):
#     p0 = dr.gather(mi.Vector3f, v, faces.x, True)
#     p1 = dr.gather(mi.Vector3f, v, faces.y, True)
#     p2 = dr.gather(mi.Vector3f, v, faces.z, True)

def compute_range_loss(v,faces):
    p0 = dr.gather(mi.Vector3f, v, faces.x, True)
    p1 = dr.gather(mi.Vector3f, v, faces.y, True)
    p2 = dr.gather(mi.Vector3f, v, faces.z, True)
    l1 = dr.select(p0.x < -0.9, dr.sqr(p0.x + 0.9), dr.select(p0.x > 0.9, dr.sqr(p0.x - 0.9), 0.0))
    l2 = dr.select(p1.x < -0.9, dr.sqr(p1.x + 0.9), dr.select(p1.x > 0.9, dr.sqr(p1.x - 0.9), 0.0))
    l3 = dr.select(p2.x < -0.9, dr.sqr(p2.x + 0.9), dr.select(p2.x > 0.9, dr.sqr(p2.x - 0.9), 0.0))
    loss = dr.sum(l1) + dr.sum(l2) + dr.sum(l3)
    return loss

def simple_loss_present_self_collision(v, f, eps):
    NV = dr.width(v)
    NF = dr.width(f)
    vindex, findex = dr.meshgrid(dr.arange(dr.cuda.UInt, NV), dr.arange(dr.cuda.UInt, NF))
    # print(vindex)
    # print(findex)
   
    vertices = dr.gather(mi.Point3f, v, vindex, True)
    # print(f)
    # print(findex)
    faces = dr.gather(mi.Vector3i, f, findex, True)
    
    p0 = dr.gather(mi.Vector3f, v, faces.x, True)
    p1 = dr.gather(mi.Vector3f, v, faces.y, True)
    p2 = dr.gather(mi.Vector3f, v, faces.z, True)

    normal_v = dr.cross(p1 - p0, p2 - p0)
    normal = normal_v / dr.norm(normal_v)
    dist = dr.abs(dr.dot(normal, vertices - p0))
    exclude_adjcent_faces = (dr.neq(vindex, faces.x) & dr.neq(vindex, faces.y) & dr.neq(vindex, faces.z)) & (dist < eps)
    # print(exclude_adjcent_faces)
    penalty = dr.select(dist < eps, -dr.sqr(dist - eps) *  dr.log(dist / eps), 0.0)
    loss = dr.sum( penalty & exclude_adjcent_faces)
    return loss
    # projected_point = v1 - dist * normal

    



    return dr.sum(penalty)
    
def compute_volume(v, faces):
    # print(v.type)
    # mass_center = mi.Vector3f(v)
    # p = dr.detach(mi.Vector3f(dr.sum(v.x), dr.sum(v.y), dr.sum(v.z))) / dr.width(v)
    # print(p)
    # print()
    # exit(0)
    p0 = dr.gather(mi.Vector3f, v, faces.x, True)
    p1 = dr.gather(mi.Vector3f, v, faces.y, True)
    p2 = dr.gather(mi.Vector3f, v, faces.z, True)

    # volume = dr.sum(dr.dot(p2, dr.cross(p0, p1))) / 6.0
    delta_1 = p1 - p0
    delta_2 = p2 - p0
    corss_delta12 = dr.cross(delta_1, delta_2)
    # d = (p - p0)
    # volume = dr.sum(dr.abs(dr.dot(corss_delta12, d))) / 6.0
    volume = dr.sum((p0.y + p1.y + p2.y) * corss_delta12.y) / 6.0
    # volume_abs_x = dr.sum(dr.abs((p0.y + p1.y + p2.y) * corss_delta12.y)) / 6.0
    # volume_abs_y = dr.sum(dr.abs((p0.x + p1.x + p2.x) * corss_delta12.x)) / 6.0
    # volume_abs_z = dr.sum(dr.abs((p0.z + p1.z + p2.z) * corss_delta12.z)) / 6.0
    # print(volume_abs_x, volume_abs_y, volume_abs_z, volume)
    # exit(0)
    return volume
    # , volume_abs_x - volume, volume_abs_y - volume, volume_abs_z - volume

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

def opt(scenefile, iteration_count, key, remesh):
    # print("start optimization")
    scene = mi.load_file(DATA_DIR + scenefile)
    params = mi.traverse(scene)

    # print(params)
    # exit(0)
    
    lambda_ = 35
    ls = mi.ad.LargeSteps(params[f'{key}.vertex_positions'], params[f'{key}.faces'], lambda_)

    opt = mi.ad.Adam(lr=0.005)
    opt['u'] = ls.to_differential(params[f'{key}.vertex_positions'])
    
    F = dr.unravel(mi.Vector3i, params[f'{key}.faces'])
    V = dr.unravel(mi.Point3f, params[f'{key}.vertex_positions'])

    init_volume = dr.detach(compute_volume(V, F))
    # negative_delta_volme = init_volume_abs - init_volume
    # print(init_volume.numpy())
    # exit(0)
    dr.enable_grad(params[f'{key}.vertex_positions'])
    v_np, f_np, target_length = get_vf(params, key)
    
    params.update(opt)
    errors = []
    volumes = []
    for it in range(iteration_count):
        # print(iteration_count)
        

        if it % 1 == 0:
            shapes = scene.shapes()
            shapes[0].write_ply(TEMP_DIR + f"{key}_iter{it}.ply")

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
        # print(params)
        # exit(0)
        V = dr.unravel(mi.Point3f, params[f'{key}.vertex_positions'])
        F = dr.unravel(mi.Vector3i, params[f'{key}.faces'])

        vtemp = compute_volume(V, F)
        range_loss = compute_range_loss(V, F)
        volume_mse = mse(vtemp, init_volume)
        # negative_volume_mse = mse(dvx, negative_delta_volme)

        energy = calculate_tally_energy_light_connection(scene, V, F, TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A, it, True)
        loss = energy + range_loss * 2.0 + volume_mse * 5.0 
        dr.backward(loss)
        opt.step()

        print(f"Iteration {it:02d}: energy = {energy.numpy()[0]:6f}, volume = {vtemp.numpy()}")  #end='\r'
        errors.append(energy.numpy())
        volumes.append(vtemp.numpy())
        del energy, loss, volume_mse, range_loss, vtemp

    np.save(TEMP_DIR+"enery.npy", np.array(errors))
    np.save(TEMP_DIR+"volumes.npy", np.array(volumes))
    print('\nOptimization complete.')


def accumulateTransmittance(scene, ray, cross_section_tot_t, active):
    # state = False
    # state = dr.zeros(mi.Int, dr.width(ray))
    transmittance = dr.ones(mi.Float, dr.width(ray))
    active_zec = active
    while dr.any(active):
        its = scene.ray_intersect(ray)
        hit_emitter = (~dr.isinf(its.t)) | (~its.is_medium_transition())
        
        active &= ((~dr.isinf(its.t)) | (~its.is_medium_transition()) | (its.t > ray.maxt))
        interdist, uv, ptheta = recomputeIntersection(scene, its, V, F, ray_init, its.is_medium_transition())
        exit_medium = (dr.dot(its.n, ray.d) > 0.0)
        transmittance *= dr.select(exit_medium, dr.exp(-interdist * cross_section_tot_t), 1.0)
        its.p = ray.o + interdist * ray.d
        maxt = ray.maxt - its.t
        ray = its.spawn_ray(ray.d)
        ray.maxt = maxt

    return transmittance
        

def calculate_tally_energy_light_connection(scene, V, F, cross_section_tot_t, cross_section_tot_a, seed, reparam=True):
    rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)

    # set up tally
    E_tot = dr.zeros(FloatD)

    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1.0, 0, 0)
    
    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    ray_init = mi.Ray3f(source_origin, N_directions)
    its = scene.ray_intersect(ray_init)
    interdist, uv, ptheta = recomputeIntersection(scene, its, V, F, ray_init, its.is_medium_transition())
    
    
    intersect_medium = (~dr.isinf(its.t)) & (its.is_medium_transition())
    active = True
    
    detector = scene.emitters()[0]
    ep = detector.sample_position(0.0, mi.Point2f(rng.next_float32(), rng.next_float32()))[0]
    
    edir = (ep.p - ray_init.o) / dr.norm(ep.p - ray_init.o)
    ray_emitter = mi.Ray3f(source_origin, edir)
    

    tr, emitter_ac = accumulateTransmittance(scene, ray_emitter, V, F, cross_section_tot_t, True)
    
    eits = scene.ray_intersect(ray_emitter)


    
    # TODO: add support to non_convex
    einterdist, euv, eactive  = recomputeIntersection(scene, eits, V, F, ray_emitter, eits.is_medium_transition())
    eits.p = einterdist * ray_emitter.d + ray_emitter.o
    ray_emitter = eits.spawn_ray(ray_emitter.d)
    

    intersect_tally = (~dr.isinf(eits.t)) & (~eits.is_medium_transition())

    Jacobian =  dr.abs(dr.dot(mi.Vector3f(-1.0, 0.0, 0.0), -edir)) / (dr.sqr(dr.norm(ep.p - ray_init.o))) * INV_FOUR_PI # 
    arrive_energy = (radiance / dr.detach(ep.pdf) * Jacobian) & active & intersect_tally 
    active_emitter = eits.is_medium_transition()
    if (0 == BOUNCE_RECORD) or  (-1 == BOUNCE_RECORD):
        E_tot += dr.sum(arrive_energy)
    
    active &= intersect_medium
    its.p = interdist * N_directions + ray_init.o
    ray_current = its.spawn_ray(ray_init.d)
    p0 = its.p

    phase_end = Jacobian
    phase_continue = 1.0

    for i in range(MAX_BOUNCE):
        eits = scene.ray_intersect(ray_emitter, active_emitter)
        last_rest_dist, uv, activei = recomputeIntersection(scene, eits, V, F, ray_emitter, eits.is_medium_transition())

        transmittance = dr.exp(-cross_section_tot_t * last_rest_dist)

        # debug for separate bounce
        if (i == BOUNCE_RECORD) or  (-1 == BOUNCE_RECORD):
            e = (radiance * transmittance * phase_end / dr.detach(ep.pdf)) & active_emitter & eits.is_medium_transition()
            E_tot += dr.sum(e & ~dr.isnan(e))
            # print("E_TOT", E_tot)

        its = scene.ray_intersect(ray_current, active)
        remain_dist, uv, ptheta = recomputeIntersection(scene, its, V, F, ray_current, its.is_medium_transition())
        wo = (ptheta - p0) / dr.norm(ptheta - p0)
        
        if i > 0:
            fp_continue = hg(dr.dot(wi, wo), AVERAGE_COS)
        else:
            fp_continue = 1.0

        if reparam:
            tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_nor(cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0)
        else:
            tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0

        # remain_dist_reparam = remain_dist / sheilding.height
        dist_reparam = sample_distance(tot_cross_section_reparam_t, rng)
        optical_dist = dr.select(remain_dist_reparam <= dist_reparam, remain_dist_reparam, dist_reparam)

        # costheta = dr.dot(ray_current.direction, )
        transmittance = dr.exp(-tot_cross_section_reparam_t * optical_dist)
        dist_pdf = dr.detach(dr.select(remain_dist_reparam <= dist_reparam,  dr.exp(-tot_cross_section_reparam_t * optical_dist), tot_cross_section_reparam_t * dr.exp(-tot_cross_section_reparam_t * optical_dist)))

        # update current position of the neutron
        radiance *= (transmittance / dist_pdf) * phase_continue * fp_continue / dr.detach(fp_continue) # TODO:find the correct jacobian for the last bounce
        dist = dist_reparam * jacobian_reparam

        p0 = dist * ray_current.d + ray_current.o
        #np.savetxt("debug_ray_o_1.txt", ray_current.o.numpy())
        wi = (p0 - ray_current.o) / dr.norm(p0 - ray_current.o)

        escape = (dist > remain_dist)
        active &= (~escape)
        active_emitter = active

        ep = detector.sample_position(0.0, mi.Point2f(rng.next_float32(), rng.next_float32()))[0]
        
        edir = (ep.p - p0) / dr.norm(ep.p - p0)
        ray_emitter = mi.Ray3f(p0, edir)

        ray_current.o = p0
        wo = sample_direction_hg(rng, ray_current.d, AVERAGE_COS)
        
        dist_sqr = dr.sqr(dr.norm(ep.p - p0))
        fp_continue = hg(dr.dot(wo, wi), AVERAGE_COS)

        Gal_val = dr.abs(dr.dot(edir, mi.Vector3f(-1.0, 0.0, 0.0))) / dist_sqr
        phase_continue = (tot_cross_section_reparam_t - tot_cross_section_reparam_a)
        phase_end = (tot_cross_section_reparam_t - tot_cross_section_reparam_a) * hg(dr.dot(edir, wi), AVERAGE_COS) * Gal_val

        ray_current.d = wo

    return E_tot / NUMBER_NEUTRONS

def render_result(scenefile, iters):
    scene = mi.load_file(DATA_DIR + scenefile)
    param = mi.traverse(scene)
    #for i in range(iters)
    for i in range(iters):
        print(i)
        id = i * 1
        objs = mi.load_dict({
            'type': 'ply',
            'filename':TEMP_DIR + f"shield_iter{id}.ply",
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
        mi.util.write_bitmap(TEMP_DIR + f"opt_{i}.png", image)
        del objs


def render_geo(scenefile, iters):
    CURRENT_TEMP_DIR = "E:\Research\data\psdr_jit\camera_ready_geo\\tempginko\\"
    scene = mi.load_file(DATA_DIR + scenefile)
    param = mi.traverse(scene)
    #for i in range(iters)
    for i in range(iters):
        
        id = i * 50
        print(id)
        if (id >= 500) and (id < 600):
            continue
        objs = mi.load_dict({
            'type': 'obj',
            'filename':CURRENT_TEMP_DIR + f"displacement_iter{id}.obj",
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
        mi.util.write_bitmap(CURRENT_TEMP_DIR + "geo_iter{:02d}.png".format(i), image)
        del objs

opt("hetero.xml", 1250, "shield", remesh=100)
# render_geo("result.xml", 20)