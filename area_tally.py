
# used only on my windows machine
import sys
sys.path = ["."] + sys.path[2:]
import matplotlib.pyplot as plt
# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64
from drjit.cuda.ad import Float as FloatD
from drjit.cuda.ad import UInt32 as UIntD
import numpy as np
import random

from csg import CSGLeaf, CSGNode, csg_intersect

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR


NUMBER_NEUTRONS = 10000
MAX_BOUNCE = 2
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

def load_scene_node():
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
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/torusA.obj",
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
    A = scene.shapes()[0]
    B = scene.shapes()[1]
    
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    node = CSGNode("difference", shape0, shape1)
    return scene, node

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


def recomputeIntersection(scene, its, v, f, ray, active):
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

    return t, mi.Vector2f(u, v), p


def visualize_intersect(its, c="green"):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # ax.set_box_aspect([2.5, 2, 2])
    ax.set_box_aspect([3, 1, 3])
    ax.scatter(its.p.x, its.p.y, its.p.z, marker="o", color=c)
    
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
    x1, y1, z1 = torus(100, 1.0, 0.4)

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

# def render_csg(height, cross_section_tot_t, cross_section_tot_a, seed, reparam=True):
#     rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)
#     E_tot = dr.zeros(FloatD)
#     radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
#     source_origin = mi.Point3f(0.0, 0, 0)
#     N_directions = sample_dir_from_unit_sphere(rng)

#     scene, material_node = load_scene_node()

#     params = mi.traverse(scene)

#     V = dr.unravel(mi.Point3f, params['A.vertex_positions'])
#     F = dr.unravel(mi.Vector3i, params['A.faces'])
#     V.y = V.y + height
#     params['A.vertex_positions'] = dr.ravel(V)
#     dr.enable_grad(params['A.vertex_positions'])
#     params.update()

#     ray_init = mi.Ray3f(source_origin, N_directions)
#     its = csg_intersect(scene, ray_init, material_node)
    

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
    interdist = dr.zeros(FloatD, dr.width(its))
    pos = dr.zeros(mi.Vector3f, dr.width(its))
    for i in range(len(Vs)):
        intersect_dist_i, uv, p = recomputeIntersection(scene, its, Vs[i], Fs[i], ray, its.is_valid() & dr.eq(UIntD(shape_id), i) & active)
        interdist = dr.select(dr.eq(UIntD(shape_id), i), intersect_dist_i, interdist)
        pos = dr.select(dr.eq(UIntD(shape_id), i), p, pos)
        # print("p before interesct", its.p.numpy()[1966])
        # print("p", p.numpy()[1966])
        # print("intersect_dist_i", intersect_dist_i.numpy()[1966])
        # print("shape_id", shape_id.numpy()[1966])
        # print("i", i)
        # print(pos.numpy()[1966])
        # print(interdist.numpy()[1966])
        # print("\n")
    return interdist, pos

def simulate_neutron_in_csg_shape(cross_section_tot_t, cross_section_tot_a, seed, height, reparam=False):
    rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)
    # set up tally that exit the shape
    E_tot = dr.zeros(FloatD)

    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    
    # generate rays
    ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
    ray_init = mi.Ray3f(ray_origin, ray_vec)
    
    # load scene
    # temp example, a scene with torus
    scene, material_node = load_scene_node()
    params = mi.traverse(scene)
    # get scene parameter for optimization
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Fa = dr.unravel(mi.Vector3i, params['A.faces'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Fb = dr.unravel(mi.Vector3i, params['B.faces'])
    Va.y = Va.y + height

    params['A.vertex_positions'] = dr.ravel(Va)
    params['B.vertex_positions'] = dr.ravel(Vb)
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()

    # print("gradients")
    its, io, shape_id = csg_intersect(scene, ray_init, material_node)
    # TODO refine this to filter validation of va and vb


    interdist, p0 = recompute_intersect_csg(scene, its, [Va, Vb], [Fa, Fb], ray_init, shape_id, its.is_valid())
    
    #interdistA, uv, activei = recomputeIntersection(scene, its, Va, Fa, ray_init,  its.is_valid() & dr.eq(UIntD(shape_id), 0))
    #interdistB, uv, activei = recomputeIntersection(scene, its, Vb, Fb, ray_init,  its.is_valid() & dr.eq(UIntD(shape_id), 1))
    #interdist = dr.select(dr.eq(UIntD(shape_id), 1), interdistB, interdistA)

    active = True
    active &= its.is_valid()
    its.p = interdist * ray_vec + ray_init.o
    ray_current = its.spawn_ray(ray_init.d)
    # color = ['red', 'green', 'blue']

    for i in range(MAX_BOUNCE):
        #print("loop id ", i)
        its, io, shape_id = csg_intersect(scene, ray_current, material_node)
        #remain_dist_a, uv, ptheta_a = recomputeIntersection(scene, its, Va, Fa, ray_current, its.is_valid() & active & dr.eq(UIntD(shape_id), 0))
        #print("radiance", radiance)
        #remain_dist_b, uv, ptheta_b = recomputeIntersection(scene, its, Vb, Fb, ray_current, its.is_valid() & active & dr.eq(UIntD(shape_id), 1))
        #print("remain_dist", remain_dist_a)
        #remain_dist = dr.select(dr.eq(UIntD(shape_id), 1) & active & its.is_valid(), remain_dist_b, dr.select(dr.eq(UIntD(shape_id), 0) & active & its.is_valid(), remain_dist_a, 0.0))
        #print("renmain dist", remain_dist)

        
        #arrive_energy = (radiance) & active & escape 
        # print("arraive")
        # E_tot +
        if i > 0:
            # sample an outgoing direction by sampling unit sphere
            connect_out_dir = sample_dir_from_unit_sphere(rng)
            
            ray_connect = mi.Ray3f(ray_current.o, connect_out_dir)
            connect_its, connect_io, connect_shape_id = csg_intersect(scene, ray_connect, material_node)
            connect_dist, ptheta = recompute_intersect_csg(scene, connect_its, [Va, Vb], [Fa, Fb], ray_connect, connect_shape_id, active & connect_its.is_valid())
            # print(active)
            # ptheta = dr.select(dr.eq(UIntD(shape_id), 1) & active & its.is_valid(), ptheta_b, dr.select(dr.eq(UIntD(shape_id), 0) & active & its.is_valid(), ptheta_a, ptheta_a))
            wo_connect = (ptheta - ray_current.o) / dr.norm(ptheta - ray_current.o)

            #if reparam:
                #tot_cross_section_reparam_t, tot_cross_section_reparam_a, connect_dist_reparam, jacobian_reparam = cross_section_nor(cross_section_tot_t, cross_section_tot_a, connect_dist, 1.0)
            #ßelse:
                #tot_cross_section_reparam_t, tot_cross_section_reparam_a, connect_dist_reparam, jacobian_reparam = cross_section_tot_t, cross_section_tot_a, connect_dist, 1.0
            #  transmittance
            # if i > 0:
            # 
            # print(cross_section_tot_t)
            # print(connect_dist)
            # print(active)
            # print(dr.max(connect_dist * cross_section_tot_t & active & connect_its.is_valid()))
            # print(dr.max(connect_dist & active & connect_its.is_valid()))
            # vdist = connect_dist.numpy()
            # print(vdist)
            # vdist[np.isnan(vdist)] = 0.0
            # idx = np.argmax(vdist)
            # print("idx", idx)
            
            # print("intersection points", ptheta.numpy()[idx])
            # exit(0)
            fp_connect = hg(dr.dot(wi, wo_connect), AVERAGE_COS)
            e = radiance * fp_connect * dr.exp(-cross_section_tot_t * connect_dist)  * np.pi * 4.0 * scatter_function & active & connect_its.is_valid()
            E_tot += dr.sum(e & ~dr.isnan(e))


        remain_dist, ptheta = recompute_intersect_csg(scene, its, [Va, Vb], [Fa, Fb], ray_current, shape_id, its.is_valid() & active)

        if reparam:
            tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_nor(cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0)
        else:
            tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0
        
        dist_reparam = sample_distance(tot_cross_section_reparam_t, rng)
        optical_dist = dr.select(remain_dist_reparam <= dist_reparam, remain_dist_reparam, dist_reparam)
        transmittance = dr.exp(-tot_cross_section_reparam_t * optical_dist)
        dist_pdf = dr.detach(dr.select(remain_dist_reparam <= dist_reparam,  transmittance, tot_cross_section_reparam_t * transmittance))
        # update current position of the neutron
        if i == 0:
            E_tot += dr.sum(radiance * transmittance & active)
        else:
            radiance *= hg_continue
        
        dist = dist_reparam * jacobian_reparam
        escape = (dist > remain_dist)

        active &= ~escape
        # dr.eval(E_tot)
        #print(E_tot)
        radiance *= (transmittance / dist_pdf)
        
        # continue scattering and sample direction 
        #radiance *= 

        p0 = dist * ray_current.d + ray_current.o
        wi = (ray_current.o - p0) / dr.norm(ray_current.o - p0)
        ray_current.o = p0
        wo = sample_direction_hg(rng, ray_current.d, AVERAGE_COS)
        fp = hg(dr.dot(wo, ray_current.d), AVERAGE_COS)
        # radiance *= fp / dr.detach(fp)
        # print('new direction')
        hg_continue = fp / dr.detach(fp)
        scatter_function = (tot_cross_section_reparam_t - tot_cross_section_reparam_a)
        ray_current.d = wo
    
    return E_tot / NUMBER_NEUTRONS
    


# test and visualize
# simulate_neutron_in_csg_shape(1.5, 0.2, 10, 0.01, False)

def test_fd(k):
    N = 20
    g = FloatD(0.0)
    # test gradient computation in ring and torus
    grad_list = []
    for i in range(N):
        v1 = simulate_neutron_in_csg_shape(1.5, 0.1, i + k * N, 0.01 + 0.005, False)
        #print("Etot(h+delta) id", i, v1)
        v2 = simulate_neutron_in_csg_shape(1.5, 0.1, i + k * N, 0.01 - 0.005, False)
        #print("Etot(h-delta)", v2)
        gradient = (v1 - v2) / 0.01
        dr.eval(gradient)
        del v2, v1
        g += gradient / N
        grad_list.append(gradient)
        del gradient
        gradients_array = np.array(grad_list)
        np.save( f"gradient_fd_{k}.npy", gradients_array)
        print("finite difference gradient", g * N / (i+1))
    print("finite difference gradient avg across", N, g)

    # dvdh = FloatD(0.0)
    # for i in range(N):
    #     height = FloatD(0.01)
    #     dr.enable_grad(height)
    #     v = simulate_neutron_in_csg_shape(1.5, 0.1, i, height, True)
    #     dr.backward(v)
    #     dvdh += dr.grad(height) / N
    #     del v
    #     print("auto dif grandients avg across", i, dvdh * N / (i+1))
    # print("auto dif grandients avg across", N, dvdh)
    

test_fd(3)