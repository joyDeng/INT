
# used only on my windows machine
import sys
# print(sys.path)
# exit(0)
# sys.path = ["."] + sys.path[2:]
import matplotlib.pyplot as plt


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


NUMBER_NEUTRONS = 100000
MAX_BOUNCE = 3
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

def load_test_scene_2cubes(cross_tots, cross_as):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/cube.obj",
            'bsdf': {'type': 'diffuse'}
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/cube.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }
    scene = mi.load_dict(scene_dict)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    node = CSGNode("union", shape0, shape1)
    scm = SceneMaterial([node], cross_tots, cross_as, 2)
    
    return scene, scm

def load_test_scene_hemisphere(cross_tots, cross_as):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/sphere2.obj",
            'bsdf': {'type': 'diffuse'}
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, -3.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/large_cube.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }
    scene = mi.load_dict(scene_dict)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    node = CSGNode("difference", shape0, shape1)
    scm = SceneMaterial([node], cross_tots, cross_as, 2)
    
    return scene, scm

def load_scene_node(cross_tots, cross_as):
    
    v  = np.zeros((64, 64, 1), dtype=np.float32) + 0.1
    image = mi.Bitmap(v)
    mi.util.write_bitmap("offset.exr", image)

    
    scene_dict = {
        'type': 'scene',
            'offset': {
                'type':'bitmap',
                'filename':"offset.exr",
            },
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

def sample_dir_origin_from_ring_nosym(rng, radius):
    sample1 = rng.next_float32()
    number_neutrons = dr.width(sample1)
    angle = 2.0 * dr.pi * sample1

    o = dr.zeros(mi.Vector3f, number_neutrons)
    o.x = radius * dr.sin(angle)
    o.z = radius * dr.cos(angle)
    o.y = 0.2 * dr.cos(5 * angle)

    costheta = (0.5 - rng.next_float32()) * 2.0
    phi = 2.0 * dr.pi * rng.next_float32()
    sintheta = dr.sqrt(1.0 - dr.power(costheta, 2.0))

    v = dr.zeros(mi.Vector3f, number_neutrons)
    v.x = sintheta * dr.cos(phi)
    v.z = sintheta * dr.sin(phi)
    v.y = costheta

    v = v / dr.norm(v)
    return v, o

def sample_dir_origin_from_ring(rng, radius):
    sample1 = rng.next_float32()
    number_neutrons = dr.width(sample1)
    angle = 2.0 * dr.pi * sample1

    o = dr.zeros(mi.Vector3f, number_neutrons)
    o.x = radius * dr.sin(angle)
    o.z = radius * dr.cos(angle)

    sample2 = dr.cos(rng.next_float32() * 2 * dr.pi)
    sin_theta = dr.sqrt(1.0 - dr.power(sample2, 2.0))

    v = dr.zeros(mi.Vector3f, number_neutrons)
    v.x = sin_theta * dr.cos(angle)
    v.z = -sin_theta * dr.sin(angle)
    v.y = sample2

    v = v / dr.norm(v)
    return v, o

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
        # help(dr)
        intersect_dist_i, uv, p, intersect = recomputeIntersection(scene, its, Vs[i], Fs[i], ray, its.is_valid() & (UIntD(shape_id) == i) & active)
        valid_intersect = dr.select((UIntD(shape_id) == i), intersect, valid_intersect)
        interdist = dr.select((UIntD(shape_id) == i) & active & intersect, intersect_dist_i, interdist)
        pos = dr.select((UIntD(shape_id) == i) & active & intersect, ray.o + intersect_dist_i * ray.d, pos)
        # print("its.p, p, intersect", its.p.numpy()[1], p.numpy()[1], intersect.numpy()[1])
    #print("is valid intersect", valid_intersect)
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
    
    for cur_space in material_spaces[:-1]:
        in_medium = ~ (UIntD(cur_space) == (scm.num_material))
        # in_medium = ~dr.eq(UIntD(cur_space), scm.num_material)
        medium_count += dr.select(in_medium, 1, 0)
        medium_idx = dr.select((medium_count == 1) & in_medium, cur_space, medium_idx)

    through_vaccum = (medium_idx == scm.num_material)

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
    Return: 
        the attenuation of the energy along the ray
        reparam_factor: reparameterization factor for the scattering event alone the ray
        attenuation_till_scatter_point:
        pdf: probability of sampling the attenuation
        scatter_pos:  point for scattering event
        features: material propertes at the scattering point
        exit_ray: ray go through the medium
        numerical_mask: mask out the invalid intersection due to the numerical issue
    Parameters:
        scene: mitsuba scene
        all_its: all intersection list, each element is structured as [mi.surfaceintersect, active, shape_id]
        material_spaces: list of material along the ray
        ray_pass: the queried ray
        scm: material properties along the ray
        vertices_list: list of vertices in scene
        face_list: list of faces in scene
        rng: random number generator
    """
    ray = mi.Ray3f(ray_pass)
    num_rays = dr.width(ray)
    cross_section_tots = FloatD(scm.cross_tot_list)
    attenuation = dr.ones(FloatD, num_rays)
    reparam_factor = dr.ones(FloatD, num_rays)
    cur_cross_section_tots = dr.zeros(FloatD, num_rays)

    attenuation_sample, pdf, features, tv = sample_attenuation_along_ray(material_spaces, ray, scm, rng)
    scatter_pos = ray_pass.o
    end_pos = ray_pass.o
    nerest_hit_from_scatter_pos = dr.zeros(mi.Point3f, num_rays)
    sample_active = ~tv
    numerical_mask = False

    attenuation_till_scatter_point = dr.ones(FloatD, num_rays)
    # current_distance = dr.zeros(FloatD, num_rays)
    
    for intersect, current_material in zip(all_its, material_spaces[:-1]):

        # recompute the intersection with the gradient attached
        cur_is_valid = intersect[1]
        distance, p, valid_intersect = recompute_intersect_csg(scene, intersect[0], vertices_list, faces_list, ray, intersect[2], cur_is_valid)
        # distance = next_distance - current_distance
        # current_distance = next_distance
        
        numerical_mask |= dr.select(~(valid_intersect == intersect[0].is_valid()) & cur_is_valid & sample_active, True, False)
        material_idx = UIntD(current_material)

        # no attenuation in vaccum
        in_medium = ~ (material_idx == scm.num_material)
        material_idx = dr.select(in_medium, material_idx, 0)


        # get cross_section_value of materials
        cur_cross_section_tots = dr.gather(FloatD, cross_section_tots, material_idx)
        update_attenuation = dr.select(in_medium, dr.exp(-distance * cur_cross_section_tots), 1.0)

        # shall I check sample ray in medium or not
        stop_sample_in_current_space = (attenuation * update_attenuation < attenuation_sample) & sample_active 
        residual_attenuation = dr.select(stop_sample_in_current_space, attenuation_sample / attenuation, 1.0)


        update_dist = - dr.log(residual_attenuation) / cur_cross_section_tots
        # update pdf with current attenuation
        pdf =  dr.select(stop_sample_in_current_space, cur_cross_section_tots * attenuation_sample, pdf)
        

        # compute the point that scatters in the medium
        scatter_pos = dr.select(stop_sample_in_current_space, ray.o + dr.detach(update_dist) * ray.d, scatter_pos)
        dist_reparamed = dr.detach(update_dist / distance)
        attenuation_till_scatter_point = dr.select(stop_sample_in_current_space, attenuation * dr.exp(-cur_cross_section_tots * dist_reparamed * distance), attenuation_till_scatter_point)
        nerest_hit_from_scatter_pos = dr.select(stop_sample_in_current_space, ray.o, nerest_hit_from_scatter_pos)
        
        sample_active &= ~(stop_sample_in_current_space)
        
        attenuation *= update_attenuation
        reparam_factor = dr.select(stop_sample_in_current_space, distance, reparam_factor)

        # update the ray origin
        ray.o += distance * ray.d
        
        end_pos = dr.select(cur_is_valid & valid_intersect, ray.o, end_pos)
        
    exit_ray = sample_active
    return attenuation, reparam_factor, attenuation_till_scatter_point, pdf, scatter_pos, features, exit_ray, numerical_mask


def sample_direction_from_linear_source(num_ray):
    v = dr.zeros(mi.Vector3f, num_ray)
    v.x = -1.0
    o = dr.zeros(mi.Vector3f, num_ray)
    o.x = 5.0
    o.z = dr.linspace(Float, -1.0, 1.0, num_ray)
    return v, o

def release_intersect(its_list):
    for it in its_list:
        del it

def render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, reparam):
    Etot = dr.zeros(FloatD, dr.width(ray_current))
    radiance = dr.zeros(FloatD, dr.width(ray_current)) + 1.0
    active = True

    bounceIdx = 0
    while bounceIdx < MAX_BOUNCE:
        # get a list of intersection alone the ray
        all_its, material_spaces = dr.detach(scene_material_intersect(scene, ray_current, scm, active))
        dr.eval()
        attenuation, reparam_factor, attenuation_till_scatter, pdf, scatter_pos, feature, exit_ray, mask_invalid = sample_and_compute_attenuation_along_ray(scene, 
                                                                  all_its, material_spaces, 
                                                                  ray_current, scm, 
                                                                  vertices_list, faces_list, rng)
        # dr.detach(all_its)
        terminate_ray = mask_invalid | exit_ray

        if bounceIdx == 0:
            Etot += (attenuation * radiance & active)
            dr.eval(Etot)
        else:
            fp = dr.detach(hg(dr.dot(ray_current.d, wi_theta), AVERAGE_COS))
            Etot += (attenuation * radiance * fp / dr.detach(fp) & active)
            dr.eval(Etot)

        if reparam:
            cross_section_reparam_t = feature.ext * reparam_factor
            pdf = pdf * reparam_factor
        else:
            cross_section_reparam_t = feature.ext
        
        
        active &= (~terminate_ray)
        
   
        # phase function, sample a direction
        # better sample from the source instead of sample from the phase function
        wo = sample_direction_hg(rng, ray_current.d, AVERAGE_COS)
        wi_theta = -ray_current.d
        radiance *= (attenuation_till_scatter / dr.detach(pdf)) * (cross_section_reparam_t * feature.alb) 

        # update ray
        ray_current.o = scatter_pos
        ray_current.d = wo

        bounceIdx += 1

    number = dr.width(ray_current)
    return dr.sum(Etot) / number

def simulate_neutron_in_csg_shape(scene, scm, seed, Va, AD=False):
    # print("random seed", seed)
    rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)
    # set up tally that exit the shape
    
    # value = rng.next_float32()
    # print(value)
    # generate rays
    # ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
    # ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS), dr.zeros(mi.Point3f, NUMBER_NEUTRONS)
    ray_origin.x = 5.0
    ray_vec.x = -1.0
    
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

    
    Etot = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, AD) 
    return Etot


# test and visualize
# value = simulate_neutron_in_csg_shape(5, 0.01 + 0.002, False)
# print(value)
# value = simulate_neutron_in_csg_shape(5, 0.01 - 0.002, False)
# value = simulate_neutron_in_csg_shape(2, 0.01, False)
# print(value)

# TODO validate the gradient computation 
def test_fd(scene, Va, scm, k, height):
    N = 20
    g = FloatD(0.0)
    grad_list = []
    delta = 0.001
    
    for i in range(N):
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y += (height + delta)
        v1 = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, False)
        #print(v1.numpy())
        
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y += (height - delta)
        v2 = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, False)
        #print(v2.numpy())
        #exit(0)

        # print(Va)
        gradient = (v1 - v2) / (2.0 * delta)
        dr.eval(gradient)
        del v2, v1

        g += gradient / N
        grad_list.append(gradient)
        del gradient

        gradients_array = np.array(grad_list)
        np.save( f"delta-gradient_fd_{k}_{height:.3f}.npy", gradients_array)
        print("finite difference gradient: ith ", i, g * N / (i+1))

    # print("finite difference gradient avg across", N, g)
    # print(h)
    
    dvdh = FloatD(0.0)
    for i in range(N):
        update_height = FloatD(height)
        dr.enable_grad(update_height)
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y +=  update_height
        v_r = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, True)
        # dvdh_b = dr.grad(update_height)
        # print("before", dvdh_b.numpy())
        dr.backward(v_r)
        dvdh_g = dr.grad(update_height)
        # dvdh_g = dr.select(dr.isnan(dvdh_g), 0.0, dvdh_g)
        grad_list.append(dvdh_g.numpy())
        dvdh += dvdh_g / N
        # print("reparam, ", v_r, "no reparam", v_p)
        del v_r
        gradients_array = np.array(grad_list)
        np.save( f"delta-gradient_ad_{k}_{height:.3f}.npy", gradients_array)
        print("auto dif grandients avg across", i, dvdh * N / (i+1))
    print("auto dif grandients avg across", N, dvdh)
    
# dr.set_flag(dr.JitFlag.ReuseIndices, False)

def test_torus():
    scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)

    # has offsets
    # aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['A.vertex_normals'])
    # aUV = dr.unravel(mi.Vector2f, params['A.vertex_texcoords'])
    # tensorxf = TensorXfD(params["offset.data"])
    # heights_map = mi.Texture2f(tensorxf, wrap_mode=dr.WrapMode.Repeat)
    # offsets = heights_map.eval_cubic(aUV)[0]
    # Va = aV + aN * offsets
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    init_height = -0.08

    for h in range(32):
        for i in range(3):
            print(f"h {h}, i {i}")
            test_fd(scene,  Va, scm, i, init_height + h * 0.0045)



def delta_emission(scene, scm, num_neutrons, seed, Va, AD):
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    # set up tally that exit the shape
  
    # generate rays
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    ray_origin.x = -2.0
    ray_origin.y = 1.0
    ray_vec.x = 1.0
    
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    
    params = mi.traverse(scene)
    Fa = dr.unravel(mi.Vector3i, mi.Int(params['A.faces']))
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    # Vb.x += 2.0
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Fb = dr.unravel(mi.Vector3i, mi.Int(params['B.faces']))
    # Vb.y = Vb.y + height

    
    params['A.vertex_positions'] = dr.ravel(Va)
    params['B.vertex_positions'] = dr.ravel(Vb)
    if AD:
        dr.enable_grad(params['A.vertex_positions'])
        dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    vertices_list = [Va, Vb]
    faces_list = [Fa, Fb]

    
    Etot = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, AD) 
    return Etot

def test_hemisphere(num_neutrons, variable):
    scene, scm = load_test_scene_hemisphere(1.0, 1.0)
    params = mi.traverse(scene)
    delta = 0.01

    theta = variable
    # height = 1.0 - theta

    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Va_ = dr.zeros(mi.Point3f, dr.width(Va)) + Va 
    Va_.y -= ( theta + delta )
    energy_plus = delta_emission(scene, scm, num_neutrons, 0, Va_, False)

    Va_ = dr.zeros(mi.Point3f, dr.width(Va)) + Va 
    Va_.y -= ( theta - delta )
    energy_minus = delta_emission(scene, scm, num_neutrons, 0, Va_, False)

    gradient = ( energy_plus -  energy_minus ) / (2.0 * delta)
    print("finite difference f(x+delta), f(x-delta) and gradient are", energy_plus.numpy(), energy_minus.numpy(), gradient.numpy())
    

    # auto diff
    theta = FloatD(variable)
    dr.enable_grad(theta)

    Va_ = dr.zeros(mi.Point3f, dr.width(Va)) + Va 
    Va_.y -= theta
    energyauto = delta_emission(scene, scm, num_neutrons, 0, Va_, True)
    dr.backward(energyauto)
    auto_grad = dr.grad(theta)
    print("\nauto diff value is ", energyauto)
    print("auto diff gradients: ", auto_grad.numpy())

    # validation
    theta = FloatD(variable)
    dr.enable_grad(theta)

    dtheta = dr.sqrt(2.0 * 2.0 - (theta + 1.0) * (theta + 1.0))
    energy_ana = dr.exp(-1.0) - dr.exp(-2.0 * dtheta - 1.0)
    dr.backward(energy_ana)
    energy_grdient_wrt_theta = dr.grad(theta)
    print("\nanalytic value is", energy_ana)
    # energy_grdient_wrt_theta = - dr.exp( - 2.0 * dtheta - 1.0) * (2.0 * (theta + 1.0) / (dtheta))
    print("analytic gradient is", energy_grdient_wrt_theta)

    return gradient.numpy(), auto_grad.numpy(), energy_grdient_wrt_theta.numpy()
    
def test_hemisphere_range():
    init_value = 0.0
    fd = []
    ad = []
    vd = []
    for i in range(33):
        f, a, v = test_hemisphere(4000000, init_value + i * 0.03)
        fd.append(f)
        ad.append(a)
        vd.append(v)
        # print(init_value + i * 0.05)

    np.save("hemisphere_test_fd.npy", np.array(fd))
    np.save("hemisphere_test_ad.npy", np.array(ad))
    np.save("hemisphere_test_vd.npy", np.array(vd))



def test_2cubes(num_neutrons):
    scene, scm = load_test_scene_2cubes(1.0, 1.0)
    params = mi.traverse(scene)
    delta = 0.01

    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    print('validation: value', dr.exp(-0.5) * (1.0 - dr.exp(-1)) * (1 + dr.exp(-1)))
    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *=(2.0+delta)
    vb_.x += 2.0

    energy = delta_emission(scene, scm, num_neutrons, 0, vb_, False)
    print("mc", energy.numpy())

    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *= ( 1.5 + delta )
    vb_.x += 2.0
    
    print("validate: ", dr.exp(-0.5) - dr.exp(-3.5))
    energy1 = delta_emission(scene, scm, num_neutrons, 0, vb_, False)
    print(energy1.numpy())
    
    
    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *= ( 1.5 - delta )
    vb_.x += 2.0
    energy2 = delta_emission(scene, scm, num_neutrons, 0, vb_, False)
    print(energy2.numpy())

    gradients = (energy1 - energy2) / (2.0 * delta)
    print("finite difference gradients ", gradients.numpy())
    print("gradient validation analytic results: ", dr.exp(-3.5))
    del energy1, energy2

    offset = FloatD(1.5)
    dr.enable_grad(offset)
    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *= offset
    vb_.x += 2.0
    energyA = delta_emission(scene, scm, num_neutrons, 0, vb_, True)
    print(energyA.numpy())

    dr.backward(energyA)
    gradientAD = dr.grad(offset)
    print("auto diff gradients", gradientAD)

    del gradientAD, energyA

   



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
    

if __name__ == "__main__":
    # test_gradient_multi_1d(0, 200000)
    # test_2cubes(400000)
    test_hemisphere_range()
    # pass