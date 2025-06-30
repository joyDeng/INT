
# used only on my windows machine
import sys
# print(sys.path)
# exit(0)
# sys.path = ["."] + sys.path[2:]
import matplotlib.pyplot as plt


# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64, UInt, TensorXf
from drjit.cuda.ad import Float as FloatD
from drjit.cuda.ad import UInt32 as UIntD, TensorXf as TensorXfD
import numpy as np
import random
import torch

from csg import CSGLeaf, CSGNode, SceneMaterial, MaterialParameter, MultiGroupTally, scene_material_intersect, Beams

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR

# dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_log_level(dr.LogLevel.Info)


NUMBER_NEUTRONS = 100000
MAX_BOUNCE = 5
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
    scm = SceneMaterial([node], cross_tots, cross_as, 2, 4)
    
    return scene, scm

def load_scene_node_track_length_test():
    v  = np.zeros((128, 128, 1), dtype=np.float32) + 0.1
    image = mi.Bitmap(v)
    mi.util.write_bitmap("offset.exr", image)

    scene_dict = {
        'type': 'scene',
        'offset0': {
                'type':'bitmap',
                'filename':"offset.exr",
            },
        'offset1': {
                'type':'bitmap',
                'filename':"offset.exr",
            },
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
            'filename': "E:/Research/NeutronInv/INT/scene/torusA.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }

    scene = mi.load_dict(scene_dict)
    
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    node1 = CSGNode("difference", shape1, shape0)
    # node1 = CSGNode("difference", CSGNode("union", shape2, shape1), shape0)

    scm = SceneMaterial([node1], [[5.0]], [[0.95]], 2, 1)
    scm.set_material_sigma(TensorXf([[5.0]]))
    scm.set_material_ald(TensorXf([[0.95]]))
    phase_function = TensorXf([
        [[1.0]],
    ])
    
    scm.set_phase_function(phase_function)

    return scene, scm


def load_scene_node_energy_dependent():
    v  = np.zeros((128, 128, 1), dtype=np.float32) + 0.1
    image = mi.Bitmap(v)
    mi.util.write_bitmap("offset.exr", image)

    
    scene_dict = {
        'type': 'scene',
        'offset0': {
                'type':'bitmap',
                'filename':"offset.exr",
            },
        'offset1': {
                'type':'bitmap',
                'filename':"offset.exr",
            },
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
            'filename': "E:/Research/NeutronInv/INT/scene/torusA.obj",
            'bsdf': {'type': 'diffuse'}
        },
        'C': {
            'id': 'C',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/torusA.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }
    scene = mi.load_dict(scene_dict)
    
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)
    shape2 = CSGLeaf(2)

    node2 = CSGNode("difference", CSGNode("difference", shape2, shape0), shape1)
    node1 = CSGNode("difference", CSGNode("intersection", shape2, shape1), shape0)
    # node1 = CSGNode("difference", CSGNode("union", shape2, shape1), shape0)

    scm = SceneMaterial([node1, node2], [[1.0, 1.0], [1.0, 1.0]], [[0.95, 0.95], [0.95, 0.95]], 3, 2)
    scm.set_material_sigma(TensorXf([[1.0, 5.0], [1.0, 2.0]]))
    scm.set_material_ald(TensorXf([[0.95, 0.95], [0.95, 0.95]]))
    phase_function = TensorXf([
        [[0.1, 0.9], [0.1, 0.9]],
        [[0.5, 0.5], [0.5, 0.5]]
    ])
    
    scm.set_phase_function(phase_function)

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

def sample_float_32(rng):
    x = rng.next_float32()
    dr.eval(x, rng)
    return x

# def energy_phase(energy_group_ids, scm, material_id):
def sample_next_energy_group(rng, cdfs, num_energy):
    rand1 = sample_float_32(rng)
    # dr.eval(rand1)
    sample1 = Float(rand1)

    # group_idx = torch.zeros(cdfs.shape[0], device="cuda:0", dtype=torch.long)
    # phase_energy_pdf = torch.zeros(cdfs.shape[0], device="cuda:0", dtype=torch.float32)
    # count = torch.zeros(cdfs.shape[0], device="cuda:0", dtype=torch.long)
    # for i in range(cdfs.shape[1]):
    #     count[(sample1 < cdfs[:, i])] += 1
    #     group_idx[(sample1 < cdfs[:, i]) & (count == 1)] = i
    #     if i == 0:
    #         phase_energy_pdf = cdfs[:, i]
    #     else:
    #         phase_energy_pdf = cdfs[:, i] - cdfs[:, i-1]
    num_ray_idx = dr.arange(UInt, dr.width(rng))
    
    pos = dr.binary_search(0, num_energy, lambda index: dr.gather(Float, cdfs.array, num_ray_idx * num_energy + index) < sample1)
    dr.eval(pos)
    value = dr.gather(Float, cdfs.array, num_ray_idx * num_energy + pos)
    pre_value = dr.gather(Float, cdfs.array, num_ray_idx * num_energy + pos-1)
    
    group_idx = UInt(pos)
    dr.make_opaque(group_idx)
    return group_idx, value - pre_value
# , pre_value - value


def test_sample_next_energy_group(num_groups, num_neutron):
    rng = mi.PCG32(size=num_neutron, initstate=100)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    nodeA = CSGNode("intersection", shape0, shape1)
    nodeB = CSGNode("difference", shape0, shape1)

    # two material here
    media = SceneMaterial([nodeA, nodeB], [1.5, 1.0], [0.1, 0.1], 2, num_groups)

    group_idx = UInt(torch.zeros(num_neutron, device="cuda:0", dtype=torch.long))
    material_idx = UInt(torch.zeros(num_neutron, device="cuda:0", dtype=torch.long))
    sigt, alb, pf = media.get_optical_properties(group_idx, material_idx)
    next_group_idx, pdf = sample_next_energy_group(rng, pf, media.energy_groups)
    # print("next group idx", next_group_idx)
    # print("next group idx", next_group_idx)
    # print("next pdf idx", phase_energy_pdf)

# test_sample_next_energy_group(2, 10)
# exit(0)
     

def hg(costheta, g):
    demon = 1.0 + g * g + 2.0 * g * costheta
    return 1.0 / (4.0 * dr.pi) * (1.0 - g * g ) / (demon * dr.sqrt(demon)); 

def sample_direction_hg(rng, wi, g):
    sample1, sample2 = sample_float_32(rng), sample_float_32(rng)
    # dr.eval(sample1, rng)
    # sample2 =  rng.next_float32()
    # dr.eval(sample2, rng)
    sqrTerm = (1.0 - g * g ) / (  1.0 - g + 2.0 * g * sample1)
    cosTheta  =  dr.select(g < 1e-3,  1.0 - 2.0 * sample1, (1.0 + g * g - sqrTerm * sqrTerm) / (2.0 * g) )

    phi = 2.0 * dr.pi * sample2
    sinThetaSqr = 1.0 - cosTheta * cosTheta
    sinThetaSqr[sinThetaSqr <= 0.0] = 0.0
    sinTheta = dr.sqrt(sinThetaSqr)
    wo = mi.Frame3f(wi).to_world(mi.Vector3f(sinTheta * dr.cos(phi), sinTheta * dr.sin(phi), cosTheta))
    return wo

def sample_dir_origin_from_ring_nosym(rng, radius):
    sample1 = sample_float_32(rng)
    # dr.eval(sample1, rng)
    number_neutrons = dr.width(sample1)
    angle = 2.0 * dr.pi * sample1

    o = dr.zeros(mi.Vector3f, number_neutrons)
    o.x = radius * dr.sin(angle)
    # o.x = 5.0
    o.z = radius * dr.cos(angle)
    # o.y = 0.2 * dr.cos(5 * angle)
    
    costheta = (0.5 - sample_float_32(rng)) * 2.0
    phi = 2.0 * dr.pi * sample_float_32(rng)
    sintheta = dr.sqrt(1.0 - dr.power(costheta, 2.0))

    v = dr.zeros(mi.Vector3f, number_neutrons)
    # v.x = -1
    v.x = sintheta * dr.cos(phi)
    v.z = sintheta * dr.sin(phi)
    v.y = costheta

    v = v / dr.norm(v)
    return v, o

def sample_dir_origin_from_ring(rng, radius):
    sample1 = sample_float_32(rng)
    # dr.eval(sample1, rng)
    number_neutrons = dr.width(sample1)
    angle = 2.0 * dr.pi * sample1

    o = dr.zeros(mi.Vector3f, number_neutrons)
    o.x = radius * dr.sin(angle)
    o.z = radius * dr.cos(angle)

    sample2 = dr.cos(sample_float_32(rng) * 2 * dr.pi)
    sin_theta = dr.sqrt(1.0 - dr.power(sample2, 2.0))

    v = dr.zeros(mi.Vector3f, number_neutrons)
    v.x = sin_theta * dr.cos(angle)
    v.z = -sin_theta * dr.sin(angle)
    v.y = sample2

    v = v / dr.norm(v)
    return v, o

def sample_dir_from_unit_ring(rng, radius, cos_theta=0.0):
    #sample position on the ring
    sample1 = sample_float_32(rng)
    number_neutrons = dr.width(sample1)
    angle = 2.0 * dr.pi * sample1
    
    o = dr.zeros(mi.Vector3f, number_neutrons)
    o.x = radius * dr.sin(angle)
    o.z = radius * dr.cos(angle)

    

    sample2 = sample_float_32(rng)
    phi = 2.0 * dr.pi * sample2
    cosTheta = (sample_float_32(rng) - 0.5) * 2.0
    # print(cosTheta)
    # exit(0)
    sin_theta = dr.sqrt(1.0 - dr.power(cosTheta, 2.0))
    v = dr.zeros(mi.Vector3f, number_neutrons)
    v.x = sin_theta * dr.sin(phi)
    v.z = sin_theta * dr.cos(phi)
    v.y = cosTheta
    
    v = v / dr.norm(v)
    return v, o



# def test_ring():
#     rng = mi.PCG32(size=NUMBER_NEUTRONS,initstate=100)
#     v = sample_dir_from_unit_ring(rng, 100)
#     print(v)

# test_ring()

def sample_dir_from_unit_sphere(rng):
    v = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS)
    sample1, sample2 = sample_float_32(rng), sample_float_32(rng)
    v.z = (0.5 - sample1) * 2.0
    sin_theta = dr.sqrt(1.0 - dr.power(v.z, 2.0))
    v.x  = sin_theta * dr.sin(dr.pi * 2.0 * sample2)
    v.y = sin_theta * dr.cos(dr.pi * 2.0 * sample2)
    return v

# RETURN a float distance that is sampled proportional to the transmittance term
def sample_distance(sig_t, rng):
    rnd1 = sample_float_32(rng)
    dr.eval(rnd1, rng)
    distance = - dr.log(1.0 - rnd1) / dr.detach(sig_t)
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
    ep = detector.sample_position(0.0, mi.Point2f(sample_float_32(rng), sample_float_32(rng)))[0]
    
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

        ep = detector.sample_position(0.0, mi.Point2f(sample_float_32(rng), sample_float_32(rng)))[0]
        
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
        
        Energy = 0.0
        FD_gradient = 0.0
        start = random.randint(0,1000)
        for i in range(start, start+N):
            
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
    # print(heights)
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
        # print("ad height", height)

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
        intersect_dist_i, uv, p, intersect = recomputeIntersection(scene, its, Vs[i], Fs[i], ray, its.is_valid() & (UInt(shape_id) == i) & active)
        valid_intersect = dr.select((UInt(shape_id) == i), intersect, valid_intersect)
        interdist = dr.select((UInt(shape_id) == i) & active & intersect, intersect_dist_i, interdist)
        pos = dr.select((UInt(shape_id) == i) & active & intersect, ray.o + intersect_dist_i * ray.d, pos)
        # print("its.p, p, intersect", its.p.numpy()[1], p.numpy()[1], intersect.numpy()[1])
    #print("is valid intersect", valid_intersect)
    return interdist, pos, valid_intersect

def sample_attenuation_along_ray(material_spaces, ray, scm, rng, energy_group):
    """
    Return: sample the an attenuation along the ray, and it's pdf
    Parameters:
        material_space: list of material idx along ray
        ray,
        scm: material information
        rng: random number generator
    """
    num_rays = dr.width(ray)
    # sample_ext_list = FloatD(scm.cross_tot_list)
    # sample_alb_list = FloatD(scm.alb_list)

    # return dr.FloatD, dr.FloatD, and Tensor
    # sample_ext_list, sample_alb_list, sample_phase_cdf = scm.get_optical_properties_by_energy(energy_group)
    # print(sample_ext_list.shape)
    # print(sample_alb_list.shape)

    cross_section_tots = dr.zeros(FloatD, num_rays)
    albedos = dr.zeros(FloatD, num_rays)

    medium_count = dr.zeros(UInt, num_rays)
    medium_idx = dr.full(UInt, scm.num_material, num_rays)
    
    for cur_space in material_spaces[:-1]:
        in_medium = ~ (UInt(cur_space) == (scm.num_material))
        medium_count += dr.select(in_medium, 1, 0)
        medium_idx = dr.select((medium_count == 1) & in_medium, cur_space, medium_idx)

    through_vaccum = (medium_idx == scm.num_material)

    medium_idx = dr.select(through_vaccum, 0, medium_idx)
    
    # old version no energy dependency
    # cross_section_tots = dr.gather(FloatD, sample_ext_list, medium_idx)
    # albedos = dr.gather(FloatD, sample_alb_list, medium_idx)

    # get material by medium idx
    
    # print(sample_ext_list[medium_idx.torch().long(), :])
    # exit(0)
    sg = scm.get_sig(medium_idx, energy_group)

    cross_section_tots = sg
    
    distance_sample = sample_distance(cross_section_tots, rng)
    distance_sample = dr.select(through_vaccum, 0.0, distance_sample)

    attenuation_sample = dr.exp(-distance_sample * cross_section_tots)

    pdf = dr.detach(cross_section_tots * attenuation_sample)
    return attenuation_sample, pdf, through_vaccum

def sample_and_compute_attenuation_along_ray(scene, all_its, material_spaces, ray_pass, scm, vertices_list, faces_list, rng, energy_group_idx, beams=[], bounceIdx=-1):
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

    # cross_section_tots = FloatD(scm.cross_tot_list[:, energy_group_idx_tensor].reshape(energy_group_idx_tensor.shape[0]))

    attenuation = dr.ones(FloatD, num_rays)
    reparam_factor = dr.ones(FloatD, num_rays)

    cur_cross_section_tots = dr.zeros(FloatD, num_rays)
    # cur_alb = dr.zeros(FloatD, num_rays)

    attenuation_sample, pdf, tv = sample_attenuation_along_ray(material_spaces, ray, scm, rng, energy_group_idx)
    scatter_pos = ray_pass.o
    end_pos = ray_pass.o
    nerest_hit_from_scatter_pos = dr.zeros(mi.Point3f, num_rays)
    sample_active = ~tv
    numerical_mask = False

    attenuation_till_scatter_point = dr.ones(FloatD, num_rays)
    scatter_material_idx = dr.zeros(UInt, num_rays)

    energy_per_ray = dr.ones(FloatD, num_rays) * 1.0 / num_rays

    

    for intersect, current_material in zip(all_its, material_spaces[:-1]):
        # recompute the intersection with the gradient attached
        cur_is_valid = intersect[1]
        distance, p, valid_intersect = recompute_intersect_csg(scene, intersect[0], vertices_list, faces_list, ray, intersect[2], cur_is_valid)
        dr.eval()

        
        
        numerical_mask |= dr.select(~(valid_intersect == intersect[0].is_valid()) & cur_is_valid & sample_active, True, False)
        material_idx = UInt(current_material)

        # no attenuation in vaccum
        in_medium = ~ (material_idx == scm.num_material)
        material_idx = dr.select(in_medium, material_idx, 0)

        # TODO: replace this 
        cur_cross_section_tots = scm.get_sig(material_idx, energy_group_idx)
        update_attenuation = dr.select(in_medium, dr.exp(-distance * cur_cross_section_tots), 1.0)

        # shall I check sample ray in medium or not
        stop_sample_in_current_space = (attenuation * update_attenuation < attenuation_sample) & sample_active 
        residual_attenuation = dr.select(stop_sample_in_current_space, attenuation_sample / attenuation, 1.0)
        update_dist = - dr.log(residual_attenuation) / cur_cross_section_tots

        # update pdf with current attenuation
        pdf =  dr.select(stop_sample_in_current_space, cur_cross_section_tots * attenuation_sample, pdf)
        scatter_material_idx = dr.select(stop_sample_in_current_space, material_idx, scatter_material_idx)
        
        # compute the point that scatters in the medium
        scatter_pos = dr.select(stop_sample_in_current_space, ray.o + dr.detach(update_dist) * ray.d, scatter_pos)
        dist_reparamed = dr.detach(update_dist / distance)
        attenuation_till_scatter_point = dr.select(stop_sample_in_current_space, attenuation * dr.exp(-cur_cross_section_tots * dist_reparamed * distance), attenuation_till_scatter_point)
        nerest_hit_from_scatter_pos = dr.select(stop_sample_in_current_space, ray.o, nerest_hit_from_scatter_pos)
        
        # add sample to get spatial value TODO: add reparameterized value to it
        if bounceIdx > -1:
            beam_distance = dr.select(sample_active, dr.select(stop_sample_in_current_space, update_dist, distance), 0)
            cur_beams = Beams(dr.detach(ray.o), dr.detach(ray.o + ray.d * beam_distance), sample_active, energy_per_ray, bounceIdx)
            beams.append(cur_beams)

        sample_active &= ~(stop_sample_in_current_space)
        
        attenuation *= update_attenuation
        reparam_factor = dr.select(stop_sample_in_current_space, distance, reparam_factor)

        # update the ray origin
        ray.o += distance * ray.d
        end_pos = dr.select(cur_is_valid & valid_intersect, ray.o, end_pos)

    sig_t, alb, phase = scm.get_optical_properties(UInt(scatter_material_idx), energy_group_idx)
    features = MaterialParameter(sig_t, alb, phase)

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

def get_voxel(startpoint, lbb, steps, resolution):
    xyz_id = dr.floor((startpoint - lbb) / steps)
    lb = (xyz_id) * steps + lbb
    rt = (xyz_id + 1) * steps + lbb
    # print("startpoint", startpoint)
    # print("steps", steps)
    # print("xyz_id", xyz_id)
    vid = xyz_id.z * resolution.x * resolution.y + xyz_id.y * resolution.x + xyz_id.x
    # vid = xyz_id.x * resolution.z * resolution.y + xyz_id.y * resolution.y + xyz_id.z
    outside = (xyz_id.x >= (resolution.x-1)) | (xyz_id.y >= (resolution.y-1)) | (xyz_id.z >= (resolution.z-1))
    return vid, lb, rt, xyz_id, outside


def accumulate_photon_beams_faster(beam_list, resolution, boundingbox):
    lbb = boundingbox[0]
    rtf = boundingbox[1]
    
    stepsizes = (rtf - lbb) / resolution
    all_voxel = (resolution.x * resolution.y * resolution.z)[0]

    voxels = dr.zeros(FloatD, all_voxel)
    voxel_volume = (stepsizes.x * stepsizes.y * stepsizes.z)[0]

    m = resolution.x * resolution.x + resolution.y * resolution.y + resolution.z * resolution.z
    
    max_grid = mi.Int(dr.ceil(dr.sqrt(mi.Float(m)))).numpy()[0]
    # dr.make_opaque(max_grid)

    for beams in beam_list:
        total_length = beams.length
        start_point = beams.start
        direction = (beams.end - beams.start) / dr.norm(beams.end - beams.start)
        # print("beams")
        for v in range(max_grid):
            # print("v", v)
            valid_beams = (total_length > 0.0) & beams.active
            vid, lb, rt, xyz_id, outside = get_voxel(start_point, lbb, stepsizes, resolution)
            # print(xyz_id)
            # print("xyz_id", xyz_id)
            # print("outside", outside)
            # print(beams.active & (~outside))
            # print(vid)
            # print("lower bound", lb)
            # print("top bound", rt)
            # exit(0)

            dist_in_voxel = beams.intersect3D(lb, rt)
            # dist_in_voxel / voxel_volume
            
            contribution = dr.select((~outside) & valid_beams, dist_in_voxel / voxel_volume, 0.0)
            # temp_voxels = dr.zeros(FloatD, all_voxel)
            dr.scatter_add(voxels, contribution, vid)
            # voxels += temp_voxels

            start_point += (dist_in_voxel + 0.00001) * direction
            total_length -= dist_in_voxel
        # exit(0)
        # print(voxels)
    return voxels


def accumulate_photon_beams(beams, resolution, lbb, rtf):
    """
    Return tensor of average energy in a voxel, this is the second pass of track-length estimator 
        resolution: Vector3i, resolution along x, y, z axis
        lbb: left bottom  back (x, y, z)
        rtf: right top  front (x, y, z)
    """
    stepsizes = (rtf - lbb) / resolution
    all_voxel = (resolution.x * resolution.y * resolution.z)[0]
    
    voxels = dr.zeros(FloatD, all_voxel)
    voxel_volume = stepsizes.x * stepsizes.y * stepsizes.z

    for vid in range(all_voxel):
        print(vid)
        vid = UInt(vid)
        dr.make_opaque(vid)
        xid = vid // (resolution.y * resolution.z)
        yz = vid % (resolution.y * resolution.z)
        yid = yz // resolution.z
        zid = yz % resolution.z

        vlbb = lbb + mi.Vector3f(xid, yid, zid) * stepsizes
        vrtf = vlbb + stepsizes

        # return this constant with right value
        constant = 1.0
        for beam in beams:
            dist = beam.intersect3D(vlbb, vrtf)
            dr.scatter_add(voxels, dist * constant / voxel_volume, vid)
            dr.eval()

    
    return voxels


def render_nuetron_in_csg_shape_energy_dependent(scene, rng, scm, vertices_list, faces_list, ray_current, reparam, beams=[]):
    """
    Energy dependent version of 'render_nuetron_in_csg_shape'
    """
    # initialize energy tallies with width equals to the number of energy groups
    num_neutron = dr.width(ray_current)
    number_of_energy_group = scm.energy_groups
    Etot = dr.zeros(FloatD, number_of_energy_group)
    radiance = dr.zeros(FloatD, num_neutron) + 1.0 
    
    energy_group_idx = dr.zeros(UInt, num_neutron) # from energy group idx 0 to n, the energy goes from high to low

    # group_pdf = dr.ones(FloatD, dr.width(ray_current))
    active = True

    bounceIdx = 0
    while bounceIdx < MAX_BOUNCE:
        all_its, material_spaces = dr.detach(scene_material_intersect(scene, ray_current, scm, active))
        dr.eval()

        attenuation, reparam_factor, attenuation_till_scatter, pdf, scatter_pos, feature, exit_ray, mask_invalid = sample_and_compute_attenuation_along_ray(scene, 
                                                                  all_its, material_spaces, 
                                                                  ray_current, scm, 
                                                                  vertices_list, faces_list, rng, energy_group_idx, beams, bounceIdx)
        # dr.detach(all_its)
        terminate_ray = mask_invalid | exit_ray

        if bounceIdx == 0:
            incremental = (attenuation * radiance & active)
        else:
            fp = dr.detach(hg(dr.dot(ray_current.d, wi_theta), AVERAGE_COS))
            incremental = ((attenuation * radiance * fp / dr.detach(fp) & active)) 

        # add the weight to corresponding energy group
        Etot_incremental = dr.zeros(FloatD,  number_of_energy_group)
        dr.scatter_add(Etot_incremental, incremental, energy_group_idx)
        Etot += Etot_incremental

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

        #use energy dependent phase function to decide the change of energy group of each particle
        energy_group_idx, group_pdf = sample_next_energy_group(rng, feature.phase_cdfs, scm.energy_groups)
        # update ray
        ray_current.o = scatter_pos
        ray_current.d = wo

        bounceIdx += 1

    return Etot / num_neutron


def render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, reparam):
    Etot = dr.zeros(FloatD, dr.width(ray_current))
    radiance = dr.zeros(FloatD, dr.width(ray_current)) + 1.0
    active = True
    energy_group_idx = dr.zeros(UInt64, dr.width(ray_current)) # from energy group idx 0 to n, the energy goes from high to low
    dr.make_opaque(energy_group_idx)
    bounceIdx = 0
    
    while bounceIdx < MAX_BOUNCE:
        # get a list of intersection alone the ray
        all_its, material_spaces = dr.detach(scene_material_intersect(scene, ray_current, scm, active))
        dr.eval()
        attenuation, reparam_factor, attenuation_till_scatter, pdf, scatter_pos, feature, exit_ray, mask_invalid = sample_and_compute_attenuation_along_ray(scene, 
                                                                  all_its, material_spaces, 
                                                                  ray_current, scm, 
                                                                  vertices_list, faces_list, rng, energy_group_idx)
        # dr.detach(all_its)
        terminate_ray = mask_invalid | exit_ray

        if bounceIdx == 0:
            # Etot += (attenuation * radiance & active)
            # dr.eval(Etot)
            pass
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
        # wo = sample_direction_hg(rng, ray_current.d, AVERAGE_COS)
        wo = dr.zeros(mi.Vector3f,  dr.width(ray_current))
        wo.y = -1.0

        wi_theta = -ray_current.d
        radiance *= (attenuation_till_scatter / dr.detach(pdf)) * (cross_section_reparam_t * feature.alb) 

        # update ray
        ray_current.o = scatter_pos
        ray_current.d = wo

        bounceIdx += 1

    number = dr.width(ray_current)
    return dr.sum(Etot) / number

