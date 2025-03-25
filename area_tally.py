
# used only on my windows machine
import sys
sys.path = ["."] + sys.path[2:]
import matplotlib.pyplot as plt
# import cv2
# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64
from drjit.cuda.ad import Float as FloatD
from drjit.cuda.ad import UInt32 as UIntD, TensorXf as TensorXfD
import numpy as np
import random

from csg import CSGLeaf, CSGNode, SceneMaterial, MaterialParameter, scene_material_intersect, ith_hit_from_current

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR


NUMBER_NEUTRONS = 10000
MAX_BOUNCE = 1
TOT_CROSS_SECTION_T = 1.5
TOT_CROSS_SECTION_A = 0.15
AVERAGE_COS = mi.Float(0.0)
FOUR_PI = 4.0 * 3.141592653
INV_FOUR_PI = 1.0 / FOUR_PI
PI = 3.141592653
BOUNCE_RECORD = 1
PARAM = "y offset k=2"

def torus(precision, c, a):
    u = np.linspace(0, 2*np.pi, precision)
    v = np.linspace(0, 2*np.pi, precision)
    u, v = np.meshgrid(u, v)
    x = (c+a*np.cos(v))*np.cos(u)
    z = (c+a*np.cos(v))*np.sin(u)
    y = a*np.sin(v)
    return x, y, z

def load_scene_node(cross_tots, cross_as):
    
    v  = np.zeros((64, 64, 1), dtype=np.float32)
    image = mi.Bitmap(v)
    mi.util.write_bitmap("offset.exr", image)

    
    scene_dict = {
        'type': 'scene',
            # 'integrator': {
            #     'type': 'path',
            #     # Indirect visibility effects aren't that important here
            #     # let's turn them off and save some computation time
            #     # 'spp': 1,
            # },
        # 'emitter': {
        #     'type': 'envmap',
        #     'filename': "../scenes/textures/envmap2.exr",
        # },
        # 'bsdf':{
        #     'type':"diffuse",
            'offset': {
                'type':'bitmap',
                'filename':"offset.exr",
            },
        # },
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/torusC.obj",
            'bsdf': {'type': 'diffuse'}
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/torusC.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }
    scene = mi.load_dict(scene_dict)
    # A = scene.shapes()[0]
    # B = scene.shapes()[1]
    
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    node = CSGNode("difference", shape0, shape1)
    scm = SceneMaterial([node], cross_tots, cross_as, 2)
    
    return scene, scm

def hg(costheta, g):
    demon = 1.0 + g * g + 2.0 * g * costheta
    return 1.0 / (4.0 * dr.pi) * (1.0 - g * g ) / (demon * dr.sqrt(demon)); 

def sample_direction_hg(rng, wi, g):
    sample1, sample2 = rng.next_float32(), rng.next_float32()
    sqrTerm = (1.0 - g * g ) / (  1.0 - g + 2.0 * g * sample1)
    cosTheta  =  dr.select(g < 1e-3,  1.0 - 2.0 * sample1, (1.0 + g * g - sqrTerm * sqrTerm) / (2.0 * g) )

    phi = 2.0 * dr.pi * sample2
    sinThetaSqr = 1.0 - cosTheta * cosTheta
    sinThetaSqr[sinThetaSqr <= 0.0] = 0.0
    sinTheta = dr.sqrt(sinThetaSqr)
    wo = mi.Frame3f(wi).to_world(mi.Vector3f(sinTheta * dr.cos(phi), sinTheta * dr.sin(phi), cosTheta))
    return wo

def sample_dir_from_unit_ring(rng, radius, cos_theta=0.0):
    #sample position on the ring
    sample1 = rng.next_float32()
    number_neutrons = dr.width(sample1)
    angle = 2.0 * dr.pi * sample1
    
    o = dr.zeros(mi.Vector3f, number_neutrons)
    o.x = radius * dr.sin(angle)
    o.z = radius * dr.cos(angle)

    sin_theta = dr.sqrt(1.0 - dr.power(cos_theta, 2.0))

    v = dr.zeros(mi.Vector3f, number_neutrons)
    v.x = sin_theta * dr.sin(angle)
    v.z = sin_theta * dr.cos(angle)
    v.y = cos_theta
    
    v = v / dr.norm(v)
    return v, o



# def test_ring():
#     rng = mi.PCG32(size=NUMBER_NEUTRONS,initstate=100)
#     v = sample_dir_from_unit_ring(rng, 100)
#     print(v)

# test_ring()

def sample_dir_from_unit_sphere(rng):
    v = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS)
    sample1, sample2 = rng.next_float32(), rng.next_float32()
    v.z = (0.5 - sample1) * 2.0
    sin_theta = dr.sqrt(1.0 - dr.power(v.z, 2.0))
    v.x  = sin_theta * dr.sin(dr.pi * 2.0 * sample2)
    v.y = sin_theta * dr.cos(dr.pi * 2.0 * sample2)
    return v

# RETURN a float distance that is sampled proportional to the transmittance term
def sample_distance(sig_t, rng):
    distance = - dr.log(1.0 - rng.next_float32()) / dr.detach(sig_t)
    return distance

# RETURN True if test current point is outside of the filter
# def escape(point, d):
#     return (point.x <= 0.0) | (point.x >= d)

# # RETURN True if the nuetron exit from the x=0 plane
# def goback(point, d):
#     return point.x <= 0.0

# # RETURN True if the nuetron exit from the x=d plane
# def gothrough(point, d):
#     return point.x >= d

# RETURN a float distance that is the closest hit from current point to 
# the surface alone current direction
# def compute_rest_dist(N_directions, current_point, depth):
#     x_dist = dr.select(N_directions.x > 0.0, depth - current_point.x, current_point.x)
#     cos_theta = dr.abs(N_directions.x)
#     rest_dist = x_dist / cos_theta
#     return rest_dist

#  RETURN reparameterize cross_section values 
def cross_section_nor(cross_section_tot, cross_section_tot_a, depth, constant):
    absorb_ratio = cross_section_tot_a / cross_section_tot
    ltot_n = (1.0 / cross_section_tot) / (constant * depth)
    # ltot_a = absorb_ratio * ltot_n
    cross_section_new = 1.0 / ltot_n
    cross_section_new_a = cross_section_new * absorb_ratio
    return cross_section_new, cross_section_new_a, depth / (constant * depth),  depth

# def balance(pdf1, pdf2):
#     return pdf1 / (pdf1 + pdf2)


def calculate_tally_energy(tally, sheilding, cross_section_tot_t, cross_section_tot_a, seed=0):
    rng = mi.PCG32(size=NUMBER_NEUTRONS,initstate=seed)

    # set up tally
    E_tot = dr.zeros(FloatD)

    # set up source: point source at (-1, 0, 0)
    # N_array_E = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1.0, 0, 0)
    
    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    # compute the position where the neutron enter the medium
    ray_init = Ray(source_origin, N_directions)
    # intersect_tally, tt, uvt = tally.intersect(ray_init)
    intersect, t, uv = sheilding.intersect(ray_init)
    p0 = ray_init.origin + ray_init.direction * (t + 0.000001)

    # for the ray that didn't intersect with the scene, check intersection with tally
    intersect_tally, tt = tally.intersect(ray_init)
    active = True
    # add contribution (TODO: add jacobian for perpendicular area)
    arrive_energy = radiance & active & (~intersect & intersect_tally)
    if (0==BOUNCE_RECORD) or (-1 == BOUNCE_RECORD):
        E_tot += dr.sum(arrive_energy)
    active &= intersect

    ray_current = Ray(p0, ray_init.direction)

    for i in range(MAX_BOUNCE):
        
        # sample collision event
        dist = dr.detach(sample_distance(cross_section_tot_t, rng))
        intersect, r1, uv = sheilding.intersect(ray_current)

        optical_dist = dr.select(r1 <= dist, r1, dist)

        transmittance = dr.exp(-cross_section_tot_t * optical_dist) 
        dist_pdf = dr.detach(dr.select(r1 <= dist, dr.exp(-cross_section_tot_t * optical_dist), cross_section_tot_t * dr.exp(-cross_section_tot_t * optical_dist)))

        # update current position of the neutron
        radiance *= (transmittance / dist_pdf)
        p0 = dist * ray_current.direction + p0

        # whether the escape neturon hit the tally (TODO: intersect with tally)
        escape = (dist > r1)
        ray_continue = Ray(p0, ray_current.direction)
        hittally, ttally = tally.intersect(ray_continue)
        cos_theta = dr.abs(dr.dot(ray_continue.direction, tally.normal))
        pdf_tally = (1.0 / tally.area) / cos_theta / (ttally * ttally)
        # weight1 = balance(1.0 / (4.0 * dr.pi), pdf_tally)
        # accumulate to the tally
        arrive_energy = (radiance) & active & escape & hittally 
        if (i==BOUNCE_RECORD) or (BOUNCE_RECORD == -1):
            E_tot += dr.sum(arrive_energy)

        active &= ~escape

        # connect to the tally (TODO: Multiple Important sampling)
        pt, pdf = tally.samplePoint(rng)
        connect_direction = (pt - ray_current.origin)
        ray_connect = Ray(ray_current.origin, connect_direction)
        exit_point, travel_dist, uvs = sheilding.intersect(ray_connect)

        connect_trans = dr.exp(-travel_dist * cross_section_tot_t)
        t = dr.norm(connect_direction)
        cos_theta = dr.abs(dr.dot(connect_direction / t, tally.normal))
        jacobian = cos_theta / (t * t) # there should be an cos
        pdf_wo = pdf / jacobian
        weight = balance(pdf_wo, 1.0 / (4 * dr.pi))
 
        # arrive_energy = connect_trans * radiance * (cross_section_tot_t - cross_section_tot_a) * (4.0 * dr.pi) / (pdf / jacobian) * weight

        # arrive_energy &= active 
        # E_tot += dr.sum(arrive_energy)

        # continue scattering and sample direction 
        radiance *= (cross_section_tot_t - cross_section_tot_a) 
        # * 4.0 * dr.pi

        ray_current.origin = p0
        ray_current.direction = sample_dir_from_unit_sphere(rng)

        # sample energy
        # N_array_E = rng.next_float32() * N_array_E

    return E_tot / NUMBER_NEUTRONS


def recomputeIntersection(scene, its, v, f, ray, active, debug=False, id=0):
    intersect = True & active
    faces = dr.gather(mi.Vector3i, f, its.prim_index, active)

    p0 = dr.gather(mi.Point3f, v, faces.x, active)
    p1 = dr.gather(mi.Point3f, v, faces.y, active)
    p2 = dr.gather(mi.Point3f, v, faces.z, active)

    # dits.p - p1
    e1 = (p1 - p0)
    e2 = (p2 - p0)

    pvec = dr.cross(ray.d, e2)
    inv_det = dr.rcp(dr.dot(e1, pvec))

    tvec = ray.o - p0
    u = dr.dot(tvec, pvec) * inv_det

    intersect &= ((u >= 0.0) & (u <= 1.0))

    qvec = dr.cross(tvec, e1)
    v = dr.dot(ray.d, qvec) * inv_det
    intersect &= ((v >= 0.0) & (u + v <= 1.0))

    t = dr.dot(e2, qvec) * inv_det
    intersect &= (t >= 0.0)

    p = e1 * u + e2 * v

    # if debug:
    #     print("p0, p1, p2", p0.numpy()[id], p1.numpy()[id], p2.numpy()[id])
    #     print("e1, e2:", e1.numpy()[id], e2.numpy()[id])
    #     print("pvec, tvec:", pvec.numpy()[id], tvec.numpy()[id])
    #     print("intersection point", p.numpy()[id])
    #     print("inv_det", inv_det.numpy()[id])
    #     print("distance", t.numpy()[id])
    #     print("its.p", its.p.numpy()[7505])
    #     print(dr.isinf(inv_det).numpy()[7505])

    # the p is not comptued correctly here
    return t, mi.Vector2f(u, v), p, intersect


def visualize_intersect(its, c="green"):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # ax.set_box_aspect([2.5, 2, 2])
    ax.set_box_aspect([3, 1, 3])
    for p, c in zip(its, c):
        ax.scatter(p.x, p.y, p.z, marker="o", color=c)
    
    # Make data
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = 1 * np.outer(np.cos(u), np.sin(v))
    # y = 1 * np.outer(np.sin(u), np.sin(v))
    # z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    # c = radius = 1.0
    # a = second_radius = 0.5
    # precision = 100

    x, y, z = torus(100, 1.0, 0.5)
    x1, y1, z1 = torus(100, 1.0, 0.3)

    # Plot the surface
    ax.plot_surface(x, y, z, alpha=0.2, color="g")
    ax.plot_surface(x1, y1, z1, alpha=0.2, color="g")
    # ax.plot_surface(x-0.5, y, z, alpha=0.2, color="g")

    # Set an equal aspect ratio
    # ax.set_aspect('equal')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def calculate_tally_energy_light_connection(height, cross_section_tot_t, cross_section_tot_a, seed, reparam=True):
    rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)
    #print("float: ", rng.next_float32().numpy())

    # set up tally
    E_tot = dr.zeros(FloatD)

    # set up source: point source at (-1, 0, 0)
    # N_array_E = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1.0, 0, 0)
    
    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    scene = mi.load_file("scene.xml")
    params = mi.traverse(scene)

    V = dr.unravel(mi.Point3f, params['shield.vertex_positions'])
    F = dr.unravel(mi.Vector3i, params['shield.faces'])
    V.y = V.y + height
    params['shield.vertex_positions'] = dr.ravel(V)
    dr.enable_grad(params['shield.vertex_positions'])
    params.update()

    ray_init = mi.Ray3f(source_origin, N_directions)
    its = scene.ray_intersect(ray_init)
    interdist, uv, ptheta = recomputeIntersection(scene, its, V, F, ray_init, its.is_medium_transition())
    
    
    intersect_medium = (~dr.isinf(its.t)) & (its.is_medium_transition())
    active = True
    
    detector = scene.emitters()[0]
    ep = detector.sample_position(0.0, mi.Point2f(rng.next_float32(), rng.next_float32()))[0]
    
    edir = (ep.p - ray_init.o) / dr.norm(ep.p - ray_init.o)
    ray_emitter = mi.Ray3f(source_origin, edir)
    eits = scene.ray_intersect(ray_emitter)
    
    # TODO: add support to non_convex
    einterdist, euv, eactive  = recomputeIntersection(scene, eits, V, F, ray_emitter, eits.is_medium_transition())
    eits.p = einterdist * ray_emitter.d + ray_emitter.o
    ray_emitter = eits.spawn_ray(ray_emitter.d)
    

    intersect_tally = (~dr.isinf(eits.t)) & (~eits.is_medium_transition())
    #print(intersect_tally)

    Jacobian =  dr.abs(dr.dot(mi.Vector3f(-1.0, 0.0, 0.0), -edir)) / (dr.sqr(dr.norm(ep.p - ray_init.o))) * INV_FOUR_PI
    arrive_energy = (radiance / dr.detach(ep.pdf) * Jacobian) & active & intersect_tally 
    active_emitter = eits.is_medium_transition()
    if (0 == BOUNCE_RECORD) or  (-1 == BOUNCE_RECORD):
        E_tot += dr.sum(arrive_energy)
    
    active &= intersect_medium
    its.p = interdist * N_directions + ray_init.o
    #np.savetxt("debug_ray_init_0.txt", ray_init.o.numpy())
   # np.savetxt("debug_its.p.txt", its.p.numpy())
    ray_current = its.spawn_ray(ray_init.d)
    #np.savetxt("debug_ray_o_0.txt", ray_current.o.numpy())
    #np.savetxt("debug_active.txt", active.numpy())
    #np.savetxt("debug_interdist.txt", interdist.numpy())
    p0 = its.p
    #print(ray_current.o)
    #exit(0)

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

        #np.savetxt("debug_wi.txt", wi.numpy())
        #np.savetxt("debug_ray_d.txt", ray_current.d.numpy())
        #np.savetxt("debug_ray_o_2.txt", ray_current.o.numpy())
        #np.savetxt("debug_dist_recompute", dr.norm(p0 - ray_current.o).numpy())
        #np.savetxt("debug_dist", dist.numpy())
        #exit(0)

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

        # sample energy
        # N_array_E = rng.next_float32() * N_array_E

    return E_tot / NUMBER_NEUTRONS



def calculate_tally_energy_reparam(height, cross_section_tot_t, cross_section_tot_a, seed, reparam=True):
    rng = mi.PCG32(size=NUMBER_NEUTRONS,initstate=seed)

    # set up tally
    E_tot = dr.zeros(FloatD)

    # set up source: point source at (-1, 0, 0)
    # N_array_E = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1.0, 0, 0)
    
    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    # compute the position where the neutron enter the medium
    # ray_init = Ray(source_origin, N_directions)
    # intersect_tally, tt, uvt = tally.intersect(ray_init)
    # intersect, t, uv = sheilding.intersect(ray_init)
    # trecompute = sheilding.compute_rest_dist(uv, ray_init.origin)

    scene = mi.load_file("scene.xml")
    params = mi.traverse(scene)
    # print(params)
    # exit(0)

    V = dr.unravel(mi.Point3f, params['shield.vertex_positions'])
    F = dr.unravel(mi.Vector3i, params['shield.faces'])
    # UV = dr.unrevel(mi.Vector2f, params['shield.vertex_texcoords'])
    V.y = V.y * height
    params['shield.vertex_positions'] = dr.ravel(V)
    dr.enable_grad(params['shield.vertex_positions'])
    params.update()
    
    ray_init = mi.Ray3f(source_origin, N_directions)
    its = scene.ray_intersect(ray_init)
    interdist, uv, activei = recomputeIntersection(scene, its, V, F, ray_init, its.is_medium_transition())

    # for the ray that didn't intersect with the scene, check intersection with tally
    intersect_tally = (~dr.isinf(its.t)) & (~its.is_medium_transition())
    intersect_medium = (~dr.isinf(its.t)) & (its.is_medium_transition())
    active = True
    # add contribution (TODO: add jacobian for perpendicular area)
    arrive_energy = (radiance * dr.abs(dr.dot(its.n, -ray_init.d))) & active & intersect_tally
    E_tot += dr.sum(arrive_energy)
    
    active &= intersect_medium
    its.p = interdist * N_directions + ray_init.o
    ray_current = its.spawn_ray(ray_init.d)

    for i in range(MAX_BOUNCE):
        # intersect, r1, uv = sheilding.intersect(ray_current)
        its = scene.ray_intersect(ray_current, active)
        remain_dist, uv, activei = recomputeIntersection(scene, its, V, F, ray_current, its.is_medium_transition())
        # pits = scene.ray_intersect_preliminary(ray_current, active)
        # print(its)
        # exit(0)
        # _preliminary
        # tempP = recomputeIntersection(scene, pits, V, F, its.is_medium_transition())
        # remain_dist = dr.norm(tempP - ray_current.o)
        # 
        # print(remain_dist)
        # exit(0)
        # sheilding.compute_rest_dist(uv, ray_current.origin)
        if reparam:
            tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_nor(cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0)
        else:
            tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0
        #cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0
        
        # remain_dist_reparam = remain_dist / sheilding.height
        dist_reparam = sample_distance(tot_cross_section_reparam_t, rng)
        optical_dist = dr.select(remain_dist_reparam <= dist_reparam, remain_dist_reparam, dist_reparam)

        # costheta = dr.dot(ray_current.direction, )
        transmittance = dr.exp(-tot_cross_section_reparam_t * optical_dist)
        dist_pdf = dr.detach(dr.select(remain_dist_reparam <= dist_reparam,  dr.exp(-tot_cross_section_reparam_t * optical_dist), tot_cross_section_reparam_t * dr.exp(-tot_cross_section_reparam_t * optical_dist)))

        # transmittance = dr.select(remain_dist_reparam <= dist_reparam,  tot_cross_section_reparam_t * dr.exp(-tot_cross_section_reparam_t * dist_reparam), dr.exp(-tot_cross_section_reparam_t * optical_dist))
        # dist_pdf = tot_cross_section_reparam_t * dr.detach( dr.exp(-tot_cross_section_reparam_t * dist_reparam))

        # update current position of the neutron
        radiance *= (transmittance / dist_pdf)# TODO:find the correct jacobian for the last bounce
        # / dist_pdf
        # * jacobian_reparam
        dist = dist_reparam * jacobian_reparam
        # print(dist, r1)
        p0 = dist * ray_current.d + ray_current.o

        # whether the escape neturon hit the tally (intersect with tally)
        escape = (dist > remain_dist)
        ray_continue = its.spawn_ray(ray_current.d)
        
        sensor_its = scene.ray_intersect(ray_continue, active & escape)
        # print(sensor_its)
        # exit(0)
        hitally = (~sensor_its.is_medium_transition()) & (~dr.isinf(sensor_its.t))

        #cos_theta = dr.abs(dr.dot(ray_continue.direction, tally.normal))
        #pdf_tally = (1.0 / tally.area) / cos_theta / (ttally * ttally)
        # weight1 = balance(1.0 / (4.0 * dr.pi), pdf_tally)
        # accumulate to the tally
        arrive_energy = (radiance) & active & escape & hitally
        E_tot += dr.sum(arrive_energy)

        active &= ~escape

        # connect to the tally (TODO: Multiple Important sampling)
        # pt, pdf = tally.samplePoint(rng)
        # connect_direction = (pt - ray_current.origin)
        # ray_connect = Ray(ray_current.origin, connect_direction)
        # exit_point, travel_dist, uv = sheilding.intersect(ray_connect)


        # connect_trans = dr.exp(-travel_dist * cross_section_tot_t)
        # t = dr.norm(connect_direction)

        # cos_theta = dr.abs(dr.dot(connect_direction / t, tally.normal))
        # jacobian = cos_theta / (t * t) # there should be an cos
        # pdf_wo = pdf / jacobian
        # weight = balance(pdf_wo, 1.0 / (4 * dr.pi))
 
        # arrive_energy = connect_trans * radiance * (cross_section_tot_t - cross_section_tot_a) * (4.0 * dr.pi) / (pdf / jacobian) * weight

        # arrive_energy &= active 
        # E_tot += dr.sum(arrive_energy)

        # continue scattering and sample direction 
        radiance *= (tot_cross_section_reparam_t - tot_cross_section_reparam_a)
        # * 4.0 * dr.pi

        ray_current.o = p0
        wo = sample_direction_hg(rng, ray_current.d, AVERAGE_COS)

        fp = hg(dr.dot(wo, ray_current.d), AVERAGE_COS)
        radiance *= fp / dr.detach(fp)

        ray_current.d = wo

        # sample energy
        # N_array_E = rng.next_float32() * N_array_E

    return E_tot / NUMBER_NEUTRONS

def energy_tally_height(height, seed=0):
    # my_shield = RectShield(FloatD(height), FloatD(0.5), FloatD(2.0), mi.Vector3f(0.0, 0.0, 0.0))
    energy = calculate_tally_energy_light_connection(FloatD(height), TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A, seed, False)
    return energy

def energy_tally_height_reparam(height, reparam, seed):
    # my_shield = RectShield(FloatD(height), FloatD(0.5), FloatD(2.0), mi.Vector3f(0.0, 0.0, 0.0))
    energy = calculate_tally_energy_light_connection(FloatD(height), TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A, seed, reparam)
    return energy

def energy_finite_different(height, delta, seed):
    e1 = energy_tally_height(height+delta, seed)
    e2 = energy_tally_height(height-delta, seed)
    dr.detach(e1)
    dr.detach(e2)
    return (e1 - e2) / (2.0 * delta)

def test_increase_height(smallest, largest, stepsize):
    height = smallest
    list_energy = []
    list_gradient = []
    heights = []
    N=10
    while height < largest:
        print("height", height)
        Energy = 0.0
        FD_gradient = 0.0
        start = random.randint(0,1000)
        for i in range(start, start+N):
            #print("seed: ", i)
            energy = energy_tally_height(height, i)
            Energy += energy.numpy() / N
            del energy
            difference = energy_finite_different(height, 0.002, i) / N
            dr.eval(difference)
            FD_gradient += difference.numpy()
            del difference
            
        list_gradient.append(FD_gradient)
        list_energy.append(Energy)
        heights.append(height)
        height += stepsize
    energies = np.array(list_energy)
    print(heights)
    heights = np.array(heights)
    gradients_fd = np.array(list_gradient)
    return energies, heights, gradients_fd


def compute_auto_def_gradient(smallest, largest, stepsize, reparam):
    height = smallest
    list_energy = []
    list_gradient = []
    heights = []
    N = 30
    while height < largest:
        print("ad height", height)

        start = random.randint(0,1000)
        g = 0.0
        energy = 0.0
        for i in range(start, start+N):
            H = FloatD(height)
            dr.enable_grad(H)
            Energy = energy_tally_height_reparam(H, reparam, i)
            dr.backward(Energy)
            dEdR = dr.grad(H)
            g += dEdR.numpy() / N
            energy += Energy.numpy() / N
            del dEdR, Energy
        
        list_gradient.append(g)
        # print(dEdR)
        # list_gradient.append(1.0)
        list_energy.append(energy)
        heights.append(height)
        height += stepsize
    energies = np.array(list_energy)
    heights = np.array(heights)
    gradients_fd = np.array(list_gradient)
    return energies, heights, gradients_fd

def test_fd_ad():
    hl = 0.1
    hh = 1.2
    step = 0.03

    # energy_variation_ad, heights_ad, gfd_ad = compute_auto_def_gradient(hl, hh, step, reparam=False)
    # np.save(DATA_DIR + f"{PARAM}_energy_variation_ad.npy", energy_variation_ad)
    # np.save(DATA_DIR + f"{PARAM}_shield_height_ad.npy", heights_ad)
    # np.save(DATA_DIR + f"{PARAM}_gradients_fd_ad.npy", gfd_ad)

    energy_variation_ad, heights_ad, gfd_ad = compute_auto_def_gradient(hl, hh, step, reparam=True)
    np.save(DATA_DIR + f"{PARAM}_energy_variation_ad_reparam.npy", energy_variation_ad)
    np.save(DATA_DIR + f"{PARAM}_shield_height_ad_reparam.npy", heights_ad)
    np.save(DATA_DIR + f"{PARAM}_gradients_fd_ad_reparam.npy", gfd_ad)

    # energy_variation, heights, gfd = test_increase_height(hl, hh, step)
    # np.save(DATA_DIR + f"{PARAM}_energy_variation.npy", energy_variation)
    # np.save(DATA_DIR + f"{PARAM}_shield_height.npy", heights)
    # np.save(DATA_DIR + f"{PARAM}_gradients_fd.npy", gfd)


def test_light_connection():
    elight = calculate_tally_energy_light_connection(FloatD(2.5), FloatD(0.5), FloatD(0.0), 0)
    ephase = calculate_tally_energy_reparam(FloatD(2.5), FloatD(0.5), FloatD(0.0), 0)

    print("elight: ", elight, "ephase: ", ephase)


def recompute_intersect_csg(scene, its, Vs, Fs, ray, shape_id, active):
    """
    Return: distance from ray origin to the intersection, position of the intersection point
    """
    interdist = dr.zeros(FloatD, dr.width(its))
    pos = dr.zeros(mi.Vector3f, dr.width(its)) + ray.o
    # print("\n\n ")
    valid_intersect = False
    for i in range(len(Vs)):
        intersect_dist_i, uv, p, intersect = recomputeIntersection(scene, its, Vs[i], Fs[i], ray, its.is_valid() & dr.eq(UIntD(shape_id), i) & active)
        valid_intersect = dr.select(dr.eq(UIntD(shape_id), i), intersect, valid_intersect)
        interdist = dr.select(dr.eq(UIntD(shape_id), i) & active & intersect, intersect_dist_i, interdist)
        pos = dr.select(dr.eq(UIntD(shape_id), i) & active & intersect, ray.o + intersect_dist_i * ray.d, pos)
        # print("\n")
        # print(intersect.numpy()[7505], its.is_valid().numpy()[7505])
    #     print("shape_id", i, intersect_dist_i[7505], interdist.numpy()[7505], p.numpy()[7505], its.p.numpy()[7505])
    # print("shape_id", interdist.numpy()[7505], pos.numpy()[7505])
    # print("\n\n ")

        # print("Is distance smaller than zero: ", dr.any(interdist < 0.0))
        # print("ray origin", ray.o.numpy()[7505])
        # print("ray vec", ray.d.numpy()[7505])
        # print("intersection poinit", its.p.numpy()[7505], its.t.numpy()[7505], interdist.numpy()[7505], shape_id.numpy()[7505], active.numpy()[7505])
        # if dr.any(interdist < 0.0):
        #     exit(0)

    return interdist, pos, valid_intersect

def sample_attenuation_along_ray(material_spaces, ray, scm, rng):
    """
    Return: sample the an attenuation along the ray, and it's pdf
    Parameters:
        material_space: list of material idx along ray
        ray,
        scm: material information
        rng: random number generator
    """
    num_rays = dr.width(ray)
    sample_ext_list = FloatD(scm.cross_tot_list)
    sample_alb_list = FloatD(scm.alb_list)
    cross_section_tots = dr.zeros(FloatD, num_rays)
    albedos = dr.zeros(FloatD, num_rays)

    medium_count = dr.zeros(UIntD, num_rays)
    medium_idx = dr.full(UIntD, scm.num_material, num_rays)
    # print(scm.num_material, scm.num_geo)
    # exit(0)
    # active_sample = True
    # print("\n\n sample attenuation")
    for cur_space in material_spaces[:-1]:
        in_medium = ~dr.eq(UIntD(cur_space), scm.num_material)
        # in_medium_idx =  np.where(in_medium.numpy() > 0)
        # print("in medium", cur_space.numpy()[in_medium_idx])
        medium_count += dr.select(in_medium, 1, 0)
        medium_idx = dr.select(dr.eq(medium_count, 1) & in_medium, cur_space, medium_idx)
    # print("\n")

    through_vaccum = dr.eq(medium_idx, scm.num_material)
    # print("number of vaccum ray", dr.sum(through_vaccum), medium_idx)
    # exit(0)
    medium_idx = dr.select(through_vaccum, 0, medium_idx)
    
    cross_section_tots = dr.gather(FloatD, sample_ext_list, medium_idx)
    albedos = dr.gather(FloatD, sample_alb_list, medium_idx)
    
    distance_sample = sample_distance(cross_section_tots, rng)
    distance_sample = dr.select(through_vaccum, 0.0, distance_sample)

    attenuation_sample = dr.exp(-distance_sample * cross_section_tots)

    pdf = dr.detach(cross_section_tots * attenuation_sample)
    features = MaterialParameter(cross_section_tots, albedos)
    return attenuation_sample, pdf, features, through_vaccum

def sample_and_compute_attenuation_along_ray(scene, all_its, material_spaces, ray_pass, scm, vertices_list, faces_list, rng):
    """
    Return: the attenuation of the energy along the ray, and tot distance in the medium along the ray
    Parameters:
        all_its: all intersection list, each element is structured as [mi.surfaceintersect, active, shape_id]
        material_spaces: list of material along the ray
        scm: material properties along the ray
    """
    ray = mi.Ray3f(ray_pass)
    num_rays = dr.width(ray)
    cross_section_tots = FloatD(scm.cross_tot_list)
    attenuation = dr.ones(FloatD, num_rays)
    distance_tot = dr.zeros(FloatD, num_rays)
    cur_cross_section_tots = dr.zeros(FloatD, num_rays)

    attenuation_sample, pdf, features, tv = sample_attenuation_along_ray(material_spaces, ray, scm, rng)
    scatter_pos = ray_pass.o
    end_pos = ray_pass.o
    nerest_hit_from_scatter_pos = dr.zeros(mi.Point3f, num_rays)
    sample_active = ~tv
    numerical_mask = False

    attenuation_final = dr.ones(FloatD, num_rays)

    # i = 0
    # print("\n\n\nnew scattering")
    for intersect, current_material in zip(all_its, material_spaces[:-1]):
        cur_is_valid = intersect[1]
        distance, p, valid_intersect = recompute_intersect_csg(scene, intersect[0], vertices_list, faces_list, ray, intersect[2], cur_is_valid)
        # cur_is_valid &= valid_intersect

        numerical_mask |= dr.select(~dr.eq(valid_intersect, intersect[0].is_valid()) & cur_is_valid, True, False)
        # active_ray &= (cur_is_valid & valid_intersect)
        # sample_active &= valid_intersect
        material_idx = UIntD(current_material)
        # no attenuation in vaccum
        in_medium = ~dr.eq(material_idx, scm.num_material)
        material_idx = dr.select(in_medium, material_idx, 0)

        # get cross_section_value of materials
        cur_cross_section_tots = dr.gather(FloatD, cross_section_tots, material_idx)
        update_attenuation = dr.select(in_medium, dr.exp(-distance * cur_cross_section_tots), 1.0)
        stop_sample_in_current_space = (attenuation * update_attenuation < attenuation_sample) & sample_active
        residual_attenuation = dr.select(stop_sample_in_current_space, attenuation_sample / attenuation, 1.0)


        update_dist = - dr.log(residual_attenuation) / cur_cross_section_tots
        # update pdf with current attenuation
        pdf =  dr.select(stop_sample_in_current_space, cur_cross_section_tots * attenuation_sample, pdf)
        attenuation_final = dr.select(stop_sample_in_current_space, attenuation * dr.exp(-cur_cross_section_tots * update_dist), attenuation_final)

        scatter_pos = dr.select(stop_sample_in_current_space, ray.o + update_dist * ray.d, scatter_pos)
        # print("scattering pose", i, stop_sample_in_current_space.numpy()[118], "ray origin", ray.o.numpy()[118], "update_distance", update_dist.numpy()[118], ray.d.numpy()[118], distance.numpy()[118], "valid_intersect", valid_intersect[118])
        nerest_hit_from_scatter_pos = dr.select(stop_sample_in_current_space, ray.o, nerest_hit_from_scatter_pos)
        
        sample_active &= ~(stop_sample_in_current_space)
        # print("scatter_pos", scatter_pos)
        # print("nerest_hit_from_scatter_pos", nerest)
        
        attenuation *= update_attenuation
        # distance_tot += dr.select(in_medium, distance, 0.0)
        distance_tot = dr.select(stop_sample_in_current_space, distance, distance_tot)

        # update the ray origin
        # print("end pos", i, distance.numpy()[118], scatter_pos.numpy()[618], ray.o.numpy()[618], attenuation_sample.numpy()[618], update_dist.numpy()[618], attenuation.numpy()[618], in_medium.numpy()[618])
        ray.o += distance * ray.d
        # end_pos = dr.select(~sample_active, ray.o, end_pos)
        # idx = np.where(sample_active.numpy() > 0)
        # print(i, (dr.norm(scatter_pos - ray.o)).numpy()[idx])
        # print("divide by zero", np.where(dr.isnan(1.0 / dr.norm(scatter_pos - ray.o)).numpy() == True))
        
        end_pos = dr.select(cur_is_valid & valid_intersect, ray.o, end_pos)
        
        # i += 1
        # print(dr.norm(end_pos - ray_pass.o).numpy()[2531], distance.numpy()[2531], cur_is_valid.numpy()[2531], intersect[0].t.numpy()[2531])
    # exit(0)
    # terminate ray if the sample point go beyond the range. exclude the ray travel through vaccuum
    # print(sample_active.numpy()[118])
    exit_ray = sample_active
    return attenuation, distance_tot, attenuation_final, pdf, scatter_pos, nerest_hit_from_scatter_pos, end_pos, features, exit_ray, numerical_mask


def sample_direction_from_linear_source(num_ray):
    v = dr.zeros(mi.Vector3f, num_ray)
    v.x = -1.0
    o = dr.zeros(mi.Vector3f, num_ray)
    o.x = 5.0
    o.z = dr.linspace(Float, -1.0, 1.0, num_ray)
    return v, o


def render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, reparam):
    Etot = dr.zeros(FloatD, dr.width(ray_current))
    radiance = dr.zeros(FloatD, dr.width(ray_current)) + 1.0
    active = True

    
    for i in range(MAX_BOUNCE):
        
        # get all intersection along current ray
        all_its, material_spaces = scene_material_intersect(scene, ray_current, scm)
        
        attenuation, distance_tot, attenuation_sample, pdf, scatter_pos, start_pos, end_pos, feature, exit_ray, mask_invalid = sample_and_compute_attenuation_along_ray(scene, 
                                                                  all_its, material_spaces, 
                                                                  ray_current, scm, 
                                                                  vertices_list, faces_list, rng)
        
        
        terminate_ray = mask_invalid | exit_ray
        
        if i == 0:
        # accumulate to the tally
            # print(np.where(dr.isinf(attenuation * radiance & active).numpy() == 1))
            # print(attenuation.numpy()[9143], radiance.numpy()[9143])
            Etot += (attenuation * radiance & active)
            # visualize_intersect(scatter_pos & ~exit_ray & ~mask_invalid, "red")
            # print(exit_ray.numpy()[118])
            # print(np.where(((scatter_pos.z > 0.5) & mask_invalid & ~exit_ray).numpy() == True))
            # exit(0)
            # print(np.where(dr.eq(scatter_pos.x, 5.0) & active & ~terminate_ray).numpy() > 0.0)
            # pass
        else:
            wo_theta = (end_pos - ray_current.o) / dr.norm(end_pos - ray_current.o)
            # visualize_intersect([ray_current.o & active, end_pos & active], ["red", "blue"])
            # print("is nan", np.where(dr.isnan(wo_theta & active).numpy() == 1))
            fp = hg(dr.dot(wo_theta, wi_theta), AVERAGE_COS)
            Etot += (attenuation * radiance * fp / dr.detach(fp) & active)

        # reparameterize at the scattering position
        if reparam:
            cross_section_reparam_t = feature.ext * distance_tot
            pdf = pdf * distance_tot
        else:
            cross_section_reparam_t = feature.ext
        
        active &= (~terminate_ray)
        
   
        # phase function, sample a direction
        wo = sample_direction_hg(rng, ray_current.d, AVERAGE_COS)
        # TODO: ray_current.o should be the point that enter the current medium
        wi_theta = (start_pos - scatter_pos) / dr.norm(start_pos - scatter_pos)
        
        # print(scatter_pos)
        # print(start_pos)
        # print(terminate_ray)
        # fp = hg(dr.dot(wo, wi_theta), AVERAGE_COS)
        # print(np.where(dr.isnan(fp & active).numpy() == 1))
        radiance *= (attenuation_sample / dr.detach(pdf)) * (cross_section_reparam_t * feature.alb) 

        # update ray
        ray_current.o = scatter_pos
        ray_current.d = wo

    number = dr.width(ray_current)
    return dr.sum(Etot) / number

def simulate_neutron_in_csg_shape(scene, scm, seed, Va, reparam=False, AD=False):
    rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)
    # set up tally that exit the shape
  
    # generate rays
    # print("ray generation")
    # ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
    ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    
    # load scene
    # temp example, a scene with torus
    # print("scene loading")
    # scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)
    # get scene parameter for optimization
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Fa = dr.unravel(mi.Vector3i, params['A.faces'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Fb = dr.unravel(mi.Vector3i, params['B.faces'])
    # Va.y = Va.y + height

    
    params['A.vertex_positions'] = dr.ravel(Va)
    params['B.vertex_positions'] = dr.ravel(Vb)
    if AD:
        dr.enable_grad(params['A.vertex_positions'])
        dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    
    vertices_list = [Va, Vb]
    faces_list = [Fa, Fb]

    
    Etot = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, reparam) 
    return Etot
    
    
   
    


# test and visualize
# value = simulate_neutron_in_csg_shape(5, 0.01 + 0.002, False)
# print(value)
# value = simulate_neutron_in_csg_shape(5, 0.01 - 0.002, False)
# value = simulate_neutron_in_csg_shape(2, 0.01, False)
# print(value)

# TODO validate the gradient computation 
def test_fd(scene, scm, k, height):
    N = 20
    g = FloatD(0.0)
    grad_list = []

    params = mi.traverse(scene)
    # get scene parameter for optimization
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    
    for i in range(N):
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y = va_param.y + height + 0.001
        v1 = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, False)
        
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y = va_param.y + height - 0.001
        v2 = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, False)

        # print(Va)
        gradient = (v1 - v2) / 0.002
        dr.eval(gradient)
        del v2, v1

        g += gradient / N
        grad_list.append(gradient)
        del gradient

        gradients_array = np.array(grad_list)
        np.save( f"gradient_fd_{k}_{h}.npy", gradients_array)
        print("finite difference gradient: ith ", i, g * N / (i+1))

    print("finite difference gradient avg across", N, g)

    # dvdh = FloatD(0.0)
    # for i in range(N):
    #     height = FloatD(h)
    #     dr.enable_grad(height)
    #     v = simulate_neutron_in_csg_shape(i+k*N, height, True)
    #     dr.backward(v)
    #     dvdh_g = dr.grad(height)
    #     grad_list.append(dvdh_g.numpy())
    #     dvdh += dr.grad(height) / N
    #     del v
    #     gradients_array = np.array(grad_list)
    #     np.save( f"gradient_ad_{k}_{h}.npy", gradients_array)
    #     print("auto dif grandients avg across", i, dvdh * N / (i+1))
    # print("auto dif grandients avg across", N, dvdh)
    
# dr.set_flag(dr.JitFlag.ReuseIndices, False)

# scene, scm = load_scene_node([1.5], [0.9])
# exit(0)
# mi.set_log_level(mi.LogLevel.Debug)
# for h in range(2, 4):
#     for i in range(2, 4):
#         print(f"h {h}, i {i}")
#         test_fd(scene, scm, i, h * 0.005)


def test_pdf_1d(seed, number_of_neutrons):
    r_a, r_b, r_c = 1.0, 1.0, 1.0
    cs_a, cs_b, cs_c = 0.5, 0.75, 1.0
    rng = mi.PCG32(size=number_of_neutrons, initstate=seed, initseq=seed*2)
    # sample distance
    t = sample_distance(cs_a, rng)
    stop_in_a = (t < r_a)
    t_b = ( t * cs_a - r_a * cs_a ) / cs_b
    stop_in_b = (~stop_in_a) & (t_b < r_b)
    t_c = (t * cs_a - r_a * cs_a - r_b * cs_b) / cs_c
    stop_in_c = (~stop_in_a) & (~stop_in_b) & (t_c < r_c)

    t_real = dr.select(stop_in_a, t, dr.select(stop_in_b, r_a + t_b, dr.select(stop_in_c, r_a + r_b + t_c, r_a + r_b + t_c)))
    distances = t_real.numpy()
    t_real = distances[distances > 0.0]

    x = dr.linspace(Float, 0, 3.5, 1000)
    y = dr.exp(-x * cs_a)
    value = dr.select(x < r_a, cs_a * y, dr.select(x < (r_b + r_a), cs_b * dr.exp( - ((x-r_a) * cs_b + cs_a * r_a)) , dr.select(x < (r_b + r_a + r_c), cs_c * dr.exp(-((x-r_b-r_a) * cs_c + cs_a * r_a + cs_b * r_b)), dr.exp(- cs_a * r_a - cs_b * r_b - cs_c * r_c))))
    # value = dr.select(x < r_a, cs_a * dr.exp(-x * cs_a), dr.select(x < (r_b + r_a), cs_b *  dr.exp(-x * cs_b), dr.select(x < (r_b + r_a + r_c), cs_c * dr.exp(-x * cs_b), 0.0)))

    colors = plt.cm.Dark2(np.linspace(0.0, 1.0, 8))
    plt.hist(t_real, bins=2000, density=True, color = colors[0], alpha=0.5, label="sample histogram")
    plt.plot(x, value, color = colors[1], label="pdf")
    plt.ylim(0.0, 0.6)
    plt.xlim(0.0, 3.0)
    plt.legend()
    plt.show()
    # draw a histogram here
    
def test_gradient_multi_1d(seed, number_of_neutrons):
    r_a, r_b, r_c = FloatD(1.0), FloatD(1.0), FloatD(1.0)
    dr.enable_grad(r_a), dr.enable_grad(r_b), dr.enable_grad(r_c)
    cs_a, cs_b, cs_c = 0.5, 0.75, 1.0
    rng = mi.PCG32(size=number_of_neutrons, initstate=seed, initseq=seed*2)
    t = sample_distance(cs_a, rng)

    attenuation = dr.exp(-t * cs_a)

    stop_in_a = (t < r_a)
    t_b = ( t * cs_a - r_a * cs_a ) / cs_b
    stop_in_b = (~stop_in_a) & (t_b < r_b)
    t_c = (t * cs_a - r_a * cs_a - r_b * cs_b) / cs_c
    stop_in_c = (~stop_in_a) & (~stop_in_b) & (t_c < r_c)

    value = dr.select( (~stop_in_a) & (~stop_in_b) & (~stop_in_c),  dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c) / dr.detach(dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)) , 0.0)
    sum = dr.sum(value) / number_of_neutrons
    dr.backward(sum)
    dEdra = dr.grad(r_a)
    dEdrb = dr.grad(r_b)
    dEdrc = dr.grad(r_c)
    print("                       dEda,                     dEdb,                       dEdc")
    print("Auto diff, monte carlo", dEdra, dEdrb, dEdrc)
    AdEdra = -cs_a * dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)
    AdEdrb = -cs_b * dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)
    AdEdrc = -cs_c * dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)
    print("Analytic              ", AdEdra, AdEdrb, AdEdrc)
    
    
# test_gradient_multi_1d(0, 200000)