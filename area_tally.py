
# used only on my windows machine
import sys
# print(sys.path)
# exit(0)
# sys.path = ["."] + sys.path[2:]
import matplotlib.pyplot as plt



# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64, UInt, TensorXf, Float64 as Float64
from drjit.cuda.ad import Float as FloatD, Float64 as Float64D
from drjit.cuda.ad import UInt32 as UIntD, TensorXf as TensorXfD
import numpy as np
import random
import torch

from csg import CSGLeaf, CSGNode, SceneMaterial, MaterialParameter, MultiGroupTally, scene_material_intersect, Beams, SceneInfo, AttenSample, IterProp

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
    # print("cdf", cdfs)
    # print(sample1)
    # print(pos)
    
    # exit(0)
    value = dr.gather(Float, cdfs.array, num_ray_idx * num_energy + pos)
    cdfs_array_idx = num_ray_idx * num_energy + pos - 1
    cdfs_array_idx = dr.select(pos == 0, 0, cdfs_array_idx)

    # cdfs_array_idx = num_ray_idx * num_energy + pos - 1
    # cdfs_array_idx = dr.select(cdfs_array_idx < 0, 0, cdfs_array_idx)
    pre_value = dr.gather(Float, cdfs.array, cdfs_array_idx)
    pre_value = dr.select(pos == 0, 0.0, pre_value)
    
    
    group_idx = UInt(pos)
    dr.make_opaque(group_idx)
    # print(group_idx.shape, value.shape, pre_value.shape)
    # exit(0)
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

def sample_space(x, y, z, bbox):
    p_u = mi.Vector3f(x, y, z)
    p = (bbox.max - bbox.min) * p_u + bbox.min
    return p

def phase_2d(next_float):
    angle = next_float * 2.0 *  dr.pi
    x = dr.cos(angle)
    y = dr.sin(angle)
    return mi.Vector3f(x, y, dr.zeros(mi.Float, dr.width(x)))
     

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


def recomputeIntersection(scene, its, vertex, f, ray, active, debug=False, id=0):
    intersect = True & active
    
    faces = dr.gather(mi.Vector3i, f, its.prim_index, active)

    p0 = dr.gather(mi.Point3f, vertex, faces.x, active)
    p1 = dr.gather(mi.Point3f, vertex, faces.y, active)
    p2 = dr.gather(mi.Point3f, vertex, faces.z, active)

    #  Vector3T e1 = p1 - p0, e2 = p2 - p0;
    e1 = (p1 - p0)
    e2 = (p2 - p0)

    #     Vector3T pvec = dr::cross(ray.d, e2);
    #     T inv_det = dr::rcp(dr::dot(e1, pvec));
    pvec = dr.cross(ray.d, e2)
    inv_det = dr.rcp(dr.dot(e1, pvec))

    #     Vector3T tvec = ray.o - p0;
    #     T u = dr::dot(tvec, pvec) * inv_det;
    tvec = ray.o - p0
    u = dr.dot(tvec, pvec) * inv_det
    #     active &= u >= 0.f && u <= 1.f;
    intersect &= ((u >= 0.0) & (u <= 1.0))
    #     Vector3T qvec = dr::cross(tvec, e1);
    #     T v = dr::dot(ray.d, qvec) * inv_det;
    #     active &= v >= 0.f && u + v <= 1.f;
    qvec = dr.cross(tvec, e1)
    v = dr.dot(ray.d, qvec) * inv_det
    intersect &= ((v >= 0.0) & (u + v <= 1.0))
    #     T t = dr::dot(e2, qvec) * inv_det;
    #     active &= t >= 0.f && t <= ray.maxt;

    #     return { t, { u, v }, active };
    t = dr.dot(e2, qvec) * inv_det
    intersect &= (t >= 0.0)
    
    p = e1 * u + e2 * v + p0
    
    # t = FloatD(t_64)
    # p = mi.Point3f(p_64)
    intersect |= its.is_valid()

    
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
    # cos_score = dr.ones(FloatD, dr.width(its))
    valid_intersect = False

    for i in range(len(Vs)):
        intersect_dist_i, uv, p, intersect = recomputeIntersection(scene, its, Vs[i], Fs[i], ray, its.is_valid() & (UInt(shape_id) == i) & active)
        intersect_dist_i_detach_ray, uv, p, intersect = recomputeIntersection(scene, its, Vs[i], Fs[i], dr.detach(ray), its.is_valid() & (UInt(shape_id) == i) & active)
        valid_intersect = dr.select((UInt(shape_id) == i), intersect, valid_intersect)

        interdist = dr.select((UInt(shape_id) == i) & active & intersect, intersect_dist_i, interdist)
        pos = dr.select((UInt(shape_id) == i) & active & intersect, ray.o + intersect_dist_i * ray.d, pos)
        # cos_score = dr.select((UInt(shape_id) == i) & active & intersect, dr.abs(dr.dot(its.sh_frame.n, ray.d)), cos_score)
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

    sg = scm.get_sig(medium_idx, energy_group)
    cross_section_tots = sg
    
    distance_sample = sample_distance(cross_section_tots, rng)
    distance_sample = dr.select(through_vaccum, 0.0, distance_sample)
    attenuation_sample = dr.exp(-distance_sample * cross_section_tots)

    pdf = dr.detach(cross_section_tots * attenuation_sample)
    return attenuation_sample, pdf, through_vaccum

# def sample_and_compute_attenuation_along_ray(scene, all_its, material_spaces, ray_pass, scm, vertices_list, faces_list, rng, energy_group_idx, beams=[], bounceIdx=-1, beam_weight=1.0, active=True, travel_trough_material_boundary = False, save_beam=False):
def sample_and_compute_attenuation_along_ray(sceneinfo, all_its, material_spaces, itinfo, beams=[], bounceIdx=-1, save_beam=False):
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
    ray = mi.Ray3f(itinfo.ray_current)
    num_rays = dr.width(ray)

    attenuation = dr.ones(FloatD, num_rays)
    reparam_factor = dr.ones(FloatD, num_rays)

    cur_cross_section_tots = dr.zeros(FloatD, num_rays)

    attenuation_sample, pdf, tv = sample_attenuation_along_ray(material_spaces, ray, sceneinfo.scm, sceneinfo.rng, itinfo.energy_group_idx)
    scatter_pos = mi.Point3f(itinfo.ray_current.o)
    end_pos = mi.Point3f(itinfo.ray_current.o)
    # nerest_hit_from_scatter_pos = dr.zeros(mi.Point3f, num_rays)
    
    sample_active = ~tv & itinfo.active
    sample_in_matter = ~tv & itinfo.active

    numerical_mask = False

    attenuation_till_scatter_point = dr.ones(FloatD, num_rays)
    scatter_material_idx = dr.zeros(UInt, num_rays)

    energy_per_ray = dr.ones(FloatD, num_rays) * itinfo.radiance / num_rays
    # un_reparam_enery_per_ray = itinfo.last_reparam / num_rays

    reparam_constant = dr.zeros(FloatD, num_rays)
    distance = dr.zeros(FloatD, num_rays)
    
    last_distance = dr.zeros(FloatD, num_rays)
    cross_boundary_constant = dr.ones(FloatD, num_rays)
    same_material_mask = True

    # score_current = dr.ones(FloatD, num_rays)
    

    for intersect, current_material in zip(all_its, material_spaces[:-1]):
        # recompute the intersection with the gradient attached
        cur_is_valid = intersect[1] 
        distance, p, valid_intersect = recompute_intersect_csg(sceneinfo.scene, intersect[0], sceneinfo.vertices_list, sceneinfo.faces_list, ray, intersect[2], cur_is_valid)
        dr.eval()
        numerical_mask |= dr.select(~(valid_intersect == intersect[0].is_valid()) & cur_is_valid & sample_active, True, False)
        
        material_idx = UInt(current_material)
        in_medium = ~ (material_idx == sceneinfo.scm.num_material)
        material_idx = dr.select(in_medium, material_idx, 0)
        
        # quantities for integral over voxels
        cross_section_query = dr.select(in_medium, sceneinfo.scm.get_sig(material_idx, itinfo.energy_group_idx), 0.0)
        cur_alebdo = dr.select(in_medium, sceneinfo.scm.get_alb(material_idx, itinfo.energy_group_idx), 0.0)
        cross_section_difference_across_boundary = cross_section_query - cur_cross_section_tots
        cur_cross_section_tots = cross_section_query
        same_material_mask = dr.select((cross_section_difference_across_boundary == 0.0) & in_medium, True, False)
        # travel_trough_material_boundary |= ((~same_material_mask) & sample_active)

        # handle piece wise materials
        update_attenuation = dr.select(in_medium, dr.exp(-distance * cur_cross_section_tots), 1.0)
        stop_sample_in_current_space = (attenuation * update_attenuation < attenuation_sample) & sample_active 
        residual_attenuation = dr.select(stop_sample_in_current_space, attenuation_sample / attenuation, 1.0)
        update_dist = - dr.log(residual_attenuation) / cur_cross_section_tots

        # print(" current mateiral idx: ", material_idx[407], in_medium[407], " attenuation sample: ", attenuation_sample[407], " attenuation: ", attenuation[407], " attenudation * update: ", (attenuation * update_attenuation)[407])
        # update pdf with current attenuation
        pdf =  dr.select(stop_sample_in_current_space, cur_cross_section_tots * attenuation_sample, pdf)
        scatter_material_idx = dr.select(stop_sample_in_current_space, material_idx, scatter_material_idx)
        
        # compute the point that scatters in the medium
        dist_reparamed = dr.detach(update_dist / distance)
        
        # scatter_pos = dr.select(stop_sample_in_current_space, ray.o + update_dist * ray.d, scatter_pos)
        scatter_pos = dr.select(stop_sample_in_current_space, ray.o + dist_reparamed * distance * ray.d, scatter_pos)
        attenuation_till_scatter_point = dr.select(stop_sample_in_current_space, attenuation * dr.exp(-cur_cross_section_tots * dist_reparamed * distance), attenuation_till_scatter_point)
        # attenuation_till_scatter_point = dr.select(stop_sample_in_current_space, attenuation * dr.exp(-cur_cross_section_tots * update_dist), attenuation_till_scatter_point)
        # nerest_hit_from_scatter_pos = dr.select(stop_sample_in_current_space, ray.o, nerest_hit_from_scatter_pos)
        
        
        # add sample to get spatial value TODO: add reparameterized value to it
        if save_beam and bounceIdx > -1:
            beam_distance = dr.select(sample_active, dr.select(stop_sample_in_current_space, update_dist, distance), 0)
            cross_boundary_constant = dr.select(same_material_mask, cross_boundary_constant, cross_section_difference_across_boundary * last_distance)
            reparam_constant += dr.select(same_material_mask, 0.0, cross_boundary_constant)
            ray_end = ray.o + beam_distance * ray.d
            cur_beams = Beams(mi.Point3f(ray.o), ray_end, ray.d, beam_distance, dr.detach(sample_active & (~numerical_mask)), reparam_constant, energy_per_ray, bounceIdx, dr.detach(stop_sample_in_current_space))
            cur_beams.set_material(cur_cross_section_tots, cur_alebdo, False)
            beams.append(cur_beams) 

        # print("stop_sample_in_current_space: ", stop_sample_in_current_space[407], " distance: ", update_dist[407], " attenutaion: ", residual_attenuation[407], " attenuation sample: ", attenuation_sample[407])
        sample_active &= ~(stop_sample_in_current_space)
        attenuation *= update_attenuation
        reparam_factor = dr.select(stop_sample_in_current_space, distance, reparam_factor)
        last_distance = last_distance + distance
        
        # update the ray origin
        ray.o += distance * ray.d
        # ray.o = p
        end_pos = dr.select(cur_is_valid & valid_intersect, ray.o, end_pos)
        # score_current = dr.select(cur_is_valid & valid_intersect, cos_score, score_current)

    sig_t, alb, phase = sceneinfo.scm.get_optical_properties(UInt(scatter_material_idx), itinfo.energy_group_idx)
    features = MaterialParameter(sig_t, alb, phase)

    cross_boundary_constant = -cur_cross_section_tots * last_distance
    reparam_constant += dr.select(same_material_mask, 0.0, cross_boundary_constant)

    if save_beam:
        if bounceIdx == 0:
            vaccum_beams = Beams(mi.Point3f(itinfo.ray_current.o), mi.Point3f(dr.detach(itinfo.ray_current.o + dr.inf * ray.d)), ray.d, dr.inf, dr.detach(tv & itinfo.active), dr.zeros(Float, num_rays), energy_per_ray, bounceIdx)
            beams.append(vaccum_beams)
        exit_beams = Beams(mi.Point3f(ray.o), mi.Point3f(ray.o + dr.inf * ray.d), ray.d, dr.inf, dr.detach(sample_active & itinfo.active & (~numerical_mask)), reparam_constant, energy_per_ray, bounceIdx)
        beams.append(exit_beams)
    exit_ray = sample_active | tv

    # ref the definiation of attenuation sample:
    # class AttenSample:
    # def __init__(self, attenuation, reparam_factor, attenuation_till_scatter, pdf, scatter_pos, feature, exit_ray, mask_invalid, scatter_material_idx):
    ats = AttenSample(attenuation, reparam_factor, attenuation_till_scatter_point, pdf, scatter_pos, features, exit_ray, numerical_mask, scatter_material_idx)
    return  ats


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
    vid = xyz_id.z * resolution.x * resolution.y + xyz_id.y * resolution.x + xyz_id.x
    outside = (xyz_id.x > (resolution.x-1)) | (xyz_id.y > (resolution.y-1)) | (xyz_id.z > (resolution.z-1)) | (xyz_id.x < 0) | (xyz_id.y < 0) | (xyz_id.z < 0) 
    return vid, lb, rt, xyz_id, outside

def get_sol_id(xyzid, reso):
    vid = xyzid.z * reso.x * reso.y + xyzid.y * reso.x + xyzid.x
    outside = (xyzid.x > (reso.x-mi.Int32(1))) | (xyzid.y > (reso.y-mi.Int32(1))) | (xyzid.z > (reso.z-mi.Int32(1))) | (xyzid.x < mi.Int32(0)) | (xyzid.y < mi.Int32(0)) | (xyzid.z < mi.Int32(0)) 
    return vid, outside

def get_voxel_id(xyzid, lbb, steps, resolution):
    dr.make_opaque(steps)
    # vid = xyzid.z * resolution.x * resolution.y + xyzid.y * resolution.x + xyzid.x
    # outside = (xyzid.x > (resolution.x-mi.Int32(1))) | (xyzid.y > (resolution.y-mi.Int32(1))) | (xyzid.z > (resolution.z-mi.Int32(1))) | (xyzid.x < mi.Int32(0)) | (xyzid.y < mi.Int32(0)) | (xyzid.z < mi.Int32(0)) 
    vid, outside = get_sol_id(xyzid, resolution)
    # print(" lbb: ", lbb, type(xyzid))
    fxyzid = mi.Vector3f(xyzid)
    lb = (fxyzid) * steps + lbb
    rt = (fxyzid + 1.0) * steps + lbb
    # vid = dr.select(outside, mi.UInt32(0), vid)
    return vid, lb, rt, outside

def get_yz(xid, start, direction, steps, lbb):
    xstart1 = xid * steps.x
    xstart2 = (xid + 1) * steps.x
    t1 = (xstart1 - start.x) / direction.x
    t2 = (xstart2 - start.x) / direction.x
    valid_overlap = dr.select( (t1 > 0.0) | (t2 > 0.0), True, False)
    yid = dr.floor((( start.y + direction.y * t1 ) - lbb.y) / steps.y)
    zid = dr.floor((( start.z + direction.z * t1 ) - lbb.z) / steps.z)
    return valid_overlap, yid, zid
    

def net_voxel(cur_xyz_id, exit_step, resolution):
    net_voxel_id = cur_xyz_id + exit_step
    # dr.select(exit_face == 0, mi.Vector3i(-1, 0, 0), dr.select(exit_face == 1, mi.Vector3i(1, 0, 0), dr.select(exit_face == 2, mi.Vector3i(0, -1, 0), dr.select(exit_face == 3, mi.Vector3i(0, 1, 0), dr.select(exit_face == 4, mi.Vector3i(0, 0, -1), dr.select(exit_face == 5, mi.Vector3i(0, 0, 1), mi.Vector3i(0, 0, 0)))))))
    stop_xyz = (net_voxel_id == cur_xyz_id)
    in_range = (net_voxel_id.x < resolution.x) & (net_voxel_id.y < resolution.y ) & (net_voxel_id.z < resolution.z)
    stop_march = stop_xyz.x & stop_xyz.y & stop_xyz.z | ~in_range
    return net_voxel_id, stop_march

def accumulate_photon_point(beam_list, resolution, boundingbox, bounceid=-1):
    dr.make_opaque(boundingbox)
    dr.make_opaque(resolution)
    sol_vox_reso = resolution + mi.Vector3i(1)
    lbb = boundingbox[0]
    rtf = boundingbox[1]
    stepsizes = (rtf - lbb) / resolution
    lb = mi.Vector3f(lbb)
    rt = mi.Vector3f(rtf)
    dr.make_opaque(stepsizes)
    dr.make_opaque(bounceid)
    
    all_voxel = mi.UInt32((resolution.x * resolution.y * resolution.z))
    solution_voxel_number = mi.UInt32(((sol_vox_reso.x)* (sol_vox_reso.y) * (sol_vox_reso.z)))

    # voxels = dr.zeros(FloatD, all_voxel)
    sols = dr.zeros(FloatD, all_voxel)
    voxel_volume = (stepsizes.x * stepsizes.y * stepsizes.z)

    max_grid = resolution.x + resolution.y + resolution.z 
    len_beam_list = len(beam_list)
    beam_list[0].compress()
    concat_beams = beam_list[0]
    for i in range(1, len_beam_list):
        beam_list[i].compress()
        concat_beams.concat(beam_list[i])

    cur_xyz_id = mi.Vector3i(dr.floor((concat_beams.end - lbb) / stepsizes))

    # for bid in range(len(beam_list)):
    #     beams = beam_list[bid]

    #active_bid = True
#
    active_bid = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    if bounceid > -1:
        active_bid = (concat_beams.bounceIdx == bounceid)

    #     cur_xyz_id = mi.Vector3i(dr.floor((beams.end - lbb) / stepsizes))
    valid_ray = (concat_beams.length != dr.inf) & concat_beams.active & concat_beams.collision
        
    vid, lb, rt, outside = get_voxel_id(cur_xyz_id, lbb, stepsizes, resolution)

    contribution =  concat_beams.color * concat_beams.constant / voxel_volume / concat_beams.cross_section
    valid_mask = (~outside) & valid_ray & active_bid
            
    dr.scatter_add(sols, contribution, vid, valid_mask)

    return sols

def dot_axis(c):
    dot_x = dr.dot(c, mi.Vector3f(1, 0, 0))
    dot_y = dr.dot(c, mi.Vector3f(0, 1, 0))
    dot_z = dr.dot(c, mi.Vector3f(0, 0, 1))

    return (dot_x == 0) | (dot_y == 0) | (dot_z == 0)

def vector_dot_axis(c, omb):
    dot_x = dr.dot(c, mi.Vector3f(1, 0, 0))
    dot_y = dr.dot(c, mi.Vector3f(0, 1, 0))
    dot_z = dr.dot(c, mi.Vector3f(0, 0, 1))

    dot_x_ = dr.dot(c, mi.Vector3f(-1, 0, 0))
    dot_y_ = dr.dot(c, mi.Vector3f(0, -1, 0))
    dot_z_ = dr.dot(c, mi.Vector3f(0, 0, -1))
    
    cos_x = dr.select(c.x < 0.0, dot_x_, dot_x)
    cos_y = dr.select(c.y < 0.0, dot_y_, dot_y)
    cos_z = dr.select(c.z < 0.0, dot_z_, dot_z)
    return cos_x, cos_y, cos_z

    
def contribution_to_point(b, origin, end, ray_dir, distance):
    c_1 = dr.norm(origin - b)
    c_2 = dr.norm(end - b)
    
    # c1_closer = dot_axis(c_1)
    # c2_closer = dot_axis(c_2)
    c1_closer = (c_1 < c_2)

    o = dr.select(c1_closer, origin, end)
    d = dr.select(c1_closer, ray_dir, -ray_dir)
    

    cx, cy, cz = vector_dot_axis(d, o - b)
    # print(" cx, cy, cz", cx, cy, cz)
    

    v = dr.abs(o-b)
    # print(" ox, oy, oz", v, " o: ", o," b: ",  b, " closer: ", c1_closer)

    dist_sqr = distance * distance
    contribution = v.x * v.y * v.z * distance + [v.x * v.y * cz + v.x * v.z * cy + cx * v.y * v.z] * dist_sqr / FloatD(2.0) + [v.x * cy * cz + cx * v.y * cz + cx * cy * v.z] *  dist_sqr * distance / FloatD(3.0) + cx * cy * cz * dist_sqr * dist_sqr / FloatD(4.0)
    # contribution = c_constant / 4.0 * dist_sqr * dist_sqr
    # print( " contribution: ", contribution)
    # print(" distance ", distance, dist_sqr)
    return contribution

def add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, step, stepsizes, start_o, end_o, concat_beams, distance, volume, lb):
    vid, outside = get_sol_id(cur_xyz_id + step, sol_vox_reso)
    contri = contribution_to_point(lb + step * stepsizes, start_o, end_o, concat_beams.dir, distance) / volume
    # print(" contribution to ", step, " is ", contri & contrimask)
    dr.scatter_add(sols, contri * concat_beams.color / volume, vid, contrimask & (~outside))


def accumulate_photon_beams_hat(beam_list, resolution, boundingbox, bounceid):
    dr.make_opaque(boundingbox)
    dr.make_opaque(resolution)
    sol_vox_reso = resolution + mi.Vector3i(1)
    lbb = boundingbox[0]
    rtf = boundingbox[1]
    stepsizes = (rtf - lbb) / resolution
    lb = mi.Vector3f(lbb)
    rt = mi.Vector3f(rtf)
    dr.make_opaque(stepsizes)
    dr.make_opaque(bounceid)
    
    all_voxel = mi.UInt32((resolution.x * resolution.y * resolution.z))
    solution_voxel_number = mi.UInt32(((sol_vox_reso.x)* (sol_vox_reso.y) * (sol_vox_reso.z)))

    # voxels = dr.zeros(FloatD, all_voxel)
    sols = dr.zeros(FloatD, solution_voxel_number)
    voxel_volume = (stepsizes.x * stepsizes.y * stepsizes.z)

    max_grid = resolution.x + resolution.y + resolution.z 
    len_beam_list = len(beam_list)
    beam_list[0].compress()
    concat_beams = beam_list[0]
    for i in range(1, len_beam_list):
        beam_list[i].compress()
        concat_beams.concat(beam_list[i])

    cur_xyz_id = mi.Vector3i(dr.floor((concat_beams.start - lbb) / stepsizes))
    # print(" id ", dr.floor((concat_beams.start - lbb) / stepsizes))
    # print(" start ", concat_beams.start, stepsizes * cur_xyz_id + lbb, lbb, stepsizes * cur_xyz_id, dr.epsilon(FloatD))

    active_march = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    active_bid = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    outside = dr.full(mi.Bool, False, dr.width(concat_beams.start))
    exit_step = dr.zeros(mi.Vector3i, dr.width(concat_beams.start))
    dist_in_voxel = dr.zeros(FloatD, dr.width(concat_beams.start)) 
    vid = dr.zeros(mi.UInt32, dr.width(concat_beams.start))
    contribution = dr.zeros(FloatD, dr.width(concat_beams.start))
    
    if bounceid > -1:
        active_bid = (concat_beams.bounceIdx == mi.UInt32(bounceid))
    
    it = mi.UInt32(0)
    dr.make_opaque(it)
    
    while it < max_grid:
        # print("it-----------------------------------------------------------------------", it)
        dr.make_opaque(cur_xyz_id)
        vid, lb, rt, outside = get_voxel_id(cur_xyz_id, lbb, stepsizes, resolution)
        
        dist_in_voxel, exit_step, start_o, end_o = concat_beams.intersect3D_hat(lb, rt)
        # print(" distance in voxel: ", dist_in_voxel)
        # exit(0)
        contrimask = (~outside) & concat_beams.active & active_march & active_bid
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([0, 0, 0]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([1, 0, 0]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([0, 1, 0]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([0, 0, 1]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([1, 1, 0]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([1, 0, 1]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([0, 1, 1]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        add_to_point(cur_xyz_id, sol_vox_reso, sols, contrimask, mi.Vector3i([1, 1, 1]), stepsizes, start_o, end_o, concat_beams, dist_in_voxel, voxel_volume, lb)
        # exit(0)
        # if it == UInt32(0):
            # exit(0)
        # contribution =  dist_in_voxel * concat_beams.color / voxel_volume
        # dr.scatter_add(voxels, contribution, vid, valid_mask)
        cur_xyz_id, stop_march = net_voxel(cur_xyz_id, exit_step, resolution)
        active_march = active_march & (~stop_march)
        it += 1
    # print(voxels)
    return sols

def accumulate_photon_beam_with_sensor_beam(beam_list, resolution, boundingbox, bounceid=-1):
    dr.make_opaque(boundingbox)
    dr.make_opaque(resolution)
    lbb = boundingbox[0]
    rtf = boundingbox[1]
    stepsizes = (rtf - lbb) / resolution
    dr.make_opaque(stepsizes)
    dr.make_opaque(bounceid)
    
    all_voxel = mi.UInt32((resolution.x * resolution.y * resolution.z))

    voxels = dr.zeros(FloatD, all_voxel)
    empty = dr.zeros(FloatD, all_voxel)
    voxel_volume = (stepsizes.x * stepsizes.y * stepsizes.z)
    max_grid = resolution.x + resolution.y + resolution.z 
    
    # max_grid = mi.UInt32(m.numpy()[0])
    len_beam_list = len(beam_list)
    
    beam_list[0].compress()
    concat_beams = beam_list[0]

    # save before compress
    for i in range(1, len_beam_list):
        beam_list[i].compress()
        concat_beams.concat(beam_list[i])

    cur_xyz_id = mi.Vector3i(dr.floor((concat_beams.start - lbb) / stepsizes))
    active_march = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    active_bid = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    outside = dr.full(mi.Bool, False, dr.width(concat_beams.start))
    exit_step = dr.zeros(mi.Vector3i, dr.width(concat_beams.start))
    dist_in_voxel = dr.zeros(FloatD, dr.width(concat_beams.start)) 
    vid = dr.zeros(mi.UInt32, dr.width(concat_beams.start))
    contribution = dr.zeros(FloatD, dr.width(concat_beams.start))
    bounce_0 = concat_beams.bounceIdx == mi.UInt32(0)
    bounce_1 = concat_beams.bounceIdx > mi.UInt32(0)
    
    if bounceid > -1:
        active_bid = (concat_beams.bounceIdx == mi.UInt32(bounceid))
        
    it = mi.UInt32(0)
    dr.make_opaque(it)
    
    while it < max_grid:
        dr.make_opaque(cur_xyz_id)
        vid, lb, rt, outside = get_voxel_id(cur_xyz_id, lbb, stepsizes, resolution)
        dist_in_voxel, exit_step = concat_beams.intersect3D(lb, rt)
        valid_mask = (~outside) & concat_beams.active & active_march & active_bid
        # dist_in_voxel = dr.select(dist_in_voxel < 0.0, 0.0, dist_in_voxel)
        dist_detach = dr.detach(dist_in_voxel)
        

        # correction
        # this is working
        # contribution_a = dist_in_voxel * concat_beams.color / voxel_volume

        contribution_val = dist_in_voxel * concat_beams.color 
        # contribution_val = dist_detach * concat_beams.color * concat_beams.constant

        
        # contribution_val_gradient = dr.detach(dist_in_voxel) * concat_beams.color * concat_beams.constant
        # dist_in_voxel * concat_beams.color  * concat_beams.constant - dr.detach(dist_in_voxel) * concat_beams.color  * concat_beams.constant + dr.detach(dist_in_voxel) * concat_beams.color
        # dr.replace_grad(contribution_val, contribution_val_gradient)
        contribution_a = contribution_val
        # dr.detach(contribution_val) - dr.detach(contribution_val_gradient) + contribution_val_gradient
        #  + (-dist_in_voxel + dist_detach) * dr.detach(concat_beams.color))
        # (dist_in_voxel * concat_beams.constant) * dr.detach(concat_beams.color)  / voxel_volume +  dr.detach(dist_in_voxel * concat_beams.constant) * (concat_beams.color)  / voxel_volume
        #  - dist_in_voxel + dist_detach


        contribution_d = dist_detach * concat_beams.constant * concat_beams.color / voxel_volume
        dr.scatter_add(voxels, contribution_a / voxel_volume, vid, valid_mask & bounce_1)
        dr.scatter_add(voxels, contribution_d, vid, valid_mask & bounce_0)
        cur_xyz_id, stop_march = net_voxel(cur_xyz_id, exit_step, resolution)
        active_march = active_march & (~stop_march)
        it += 1

    # active_bid = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    # if bounceid > -1:
    #     active_bid = (concat_beams.bounceIdx == bounceid)

    #     #     cur_xyz_id = mi.Vector3i(dr.floor((beams.end - lbb) / stepsizes))
    # valid_c_ray = (concat_beams.length != dr.inf) & concat_beams.active & concat_beams.collision
            
    # vid, lb, rt, outside = get_voxel_id(cur_xyz_id, lbb, stepsizes, resolution)

    # contribution =  concat_beams.color * concat_beams.constant / voxel_volume / concat_beams.cross_section
    # valid_c_mask = (~outside) & valid_c_ray & active_bid
                
    # dr.scatter_add(voxels, -contribution + dr.detach(contribution), vid, valid_c_mask)
    
    return voxels


def accumulate_photon_beams_faster(beam_list, resolution, boundingbox, bounceid=-1):
    dr.make_opaque(boundingbox)
    dr.make_opaque(resolution)
    lbb = boundingbox[0]
    rtf = boundingbox[1]
    stepsizes = (rtf - lbb) / resolution
    dr.make_opaque(stepsizes)
    dr.make_opaque(bounceid)
    
    all_voxel = mi.UInt32((resolution.x * resolution.y * resolution.z))

    voxels = dr.zeros(FloatD, all_voxel)
    empty = dr.zeros(FloatD, all_voxel)
    voxel_volume = (stepsizes.x * stepsizes.y * stepsizes.z)
    max_grid = resolution.x + resolution.y + resolution.z 
    
    # max_grid = mi.UInt32(m.numpy()[0])
    len_beam_list = len(beam_list)
    
    beam_list[0].compress()
    concat_beams = beam_list[0]

    # save before compress
    for i in range(1, len_beam_list):
        beam_list[i].compress()
        concat_beams.concat(beam_list[i])

    cur_xyz_id = mi.Vector3i(dr.floor((concat_beams.start - lbb) / stepsizes))
    active_march = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    active_bid = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    outside = dr.full(mi.Bool, False, dr.width(concat_beams.start))
    exit_step = dr.zeros(mi.Vector3i, dr.width(concat_beams.start))
    dist_in_voxel = dr.zeros(FloatD, dr.width(concat_beams.start)) 
    vid = dr.zeros(mi.UInt32, dr.width(concat_beams.start))
    contribution = dr.zeros(FloatD, dr.width(concat_beams.start))
    bounce_0 = concat_beams.bounceIdx == mi.UInt32(0)
    bounce_1 = concat_beams.bounceIdx > mi.UInt32(0)
    
    if bounceid > -1:
        active_bid = (concat_beams.bounceIdx == mi.UInt32(bounceid))
        
    it = mi.UInt32(0)
    dr.make_opaque(it)
    
    while it < max_grid:
        dr.make_opaque(cur_xyz_id)
        vid, lb, rt, outside = get_voxel_id(cur_xyz_id, lbb, stepsizes, resolution)
        dist_in_voxel, exit_step = concat_beams.intersect3D(lb, rt)
        valid_mask = (~outside) & concat_beams.active & active_march & active_bid
        # dist_in_voxel = dr.select(dist_in_voxel < 0.0, 0.0, dist_in_voxel)
        dist_detach = dr.detach(dist_in_voxel)
        

        # correction
        # this is working
        # contribution_a = dist_in_voxel * concat_beams.color / voxel_volume

        contribution_val = dist_in_voxel * concat_beams.color 
        # contribution_val = dist_detach * concat_beams.color * concat_beams.constant

        
        # contribution_val_gradient = dr.detach(dist_in_voxel) * concat_beams.color * concat_beams.constant
        # dist_in_voxel * concat_beams.color  * concat_beams.constant - dr.detach(dist_in_voxel) * concat_beams.color  * concat_beams.constant + dr.detach(dist_in_voxel) * concat_beams.color
        # dr.replace_grad(contribution_val, contribution_val_gradient)
        contribution_a = contribution_val
        # dr.detach(contribution_val) - dr.detach(contribution_val_gradient) + contribution_val_gradient
        #  + (-dist_in_voxel + dist_detach) * dr.detach(concat_beams.color))
        # (dist_in_voxel * concat_beams.constant) * dr.detach(concat_beams.color)  / voxel_volume +  dr.detach(dist_in_voxel * concat_beams.constant) * (concat_beams.color)  / voxel_volume
        #  - dist_in_voxel + dist_detach


        contribution_d = dist_detach * concat_beams.constant * concat_beams.color / voxel_volume
        dr.scatter_add(voxels, contribution_a / voxel_volume, vid, valid_mask & bounce_1)
        dr.scatter_add(voxels, contribution_d, vid, valid_mask & bounce_0)
        cur_xyz_id, stop_march = net_voxel(cur_xyz_id, exit_step, resolution)
        active_march = active_march & (~stop_march)
        it += 1

    # active_bid = dr.full(mi.Bool, True, dr.width(concat_beams.start))
    # if bounceid > -1:
    #     active_bid = (concat_beams.bounceIdx == bounceid)

    #     #     cur_xyz_id = mi.Vector3i(dr.floor((beams.end - lbb) / stepsizes))
    # valid_c_ray = (concat_beams.length != dr.inf) & concat_beams.active & concat_beams.collision
            
    # vid, lb, rt, outside = get_voxel_id(cur_xyz_id, lbb, stepsizes, resolution)

    # contribution =  concat_beams.color * concat_beams.constant / voxel_volume / concat_beams.cross_section
    # valid_c_mask = (~outside) & valid_c_ray & active_bid
                
    # dr.scatter_add(voxels, -contribution + dr.detach(contribution), vid, valid_c_mask)
    
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
        vid = UInt(vid)
        dr.make_opaque(vid)
        xid = vid // (resolution.y * resolution.z)
        yz = vid % (resolution.y * resolution.z)
        yid = yz // resolution.z
        zid = yz % resolution.z

        vlbb = lbb + mi.Vector3f(xid, yid, zid) * stepsizes
        vrtf = vlbb + stepsizes

        # return this constant with right value
        # constant = 1.0
        for beam in beams:
            dist = beam.intersect3D(vlbb, vrtf)
            dr.scatter_add(voxels, dist * beam.color / voxel_volume, vid)
            dr.eval()

    
    return voxels


# def render_nuetron_in_csg_shape_energy_dependent(scene, rng, scm, vertices_list, faces_list, ray_current, reparam, beams=[], save_beam=False):
def render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, reparam, beams=[], save_beam=False):
    """
    Energy dependent version of 'render_nuetron_in_csg_shape'
    """
    # initialize energy tallies with width equals to the number of energy groups
    num_neutron = dr.width(ray_current)
    number_of_energy_group = sceneinfo.scm.energy_groups
    Etot = dr.zeros(FloatD, number_of_energy_group)
    radiance = dr.zeros(FloatD, num_neutron) + 1.0 
    sample_weight = dr.zeros(FloatD, num_neutron) + 1.0
    energy_group_idx = dr.zeros(UInt, num_neutron) # from energy group idx 0 to n, the energy goes from high to low
    last_reparam = dr.ones(FloatD, num_neutron)
    active = True
    # travel_trough_material_boundary = False

    bounceIdx = 0
    # 
    # class IterProp:
    # def __init__(self, radiance, energy_group_idx, ray_current, active)
    itinfo = IterProp(radiance, energy_group_idx, ray_current, active, last_reparam)

    while bounceIdx < MAX_BOUNCE:
        all_its, material_spaces = dr.detach(scene_material_intersect(sceneinfo.scene, itinfo.ray_current, sceneinfo.scm, itinfo.active))
        dr.eval()

        ats = sample_and_compute_attenuation_along_ray(sceneinfo, all_its, material_spaces, 
                                                                  itinfo, beams, bounceIdx, save_beam)
        terminate_ray = ats.mask_invalid | ats.exit_ray

        if bounceIdx == 0:
            incremental = (ats.attenuation * radiance & itinfo.active)
        else:
            # fp = 1.0 / (2.0 * dr.pi)
            # fp = dr.detach(hg(dr.dot(ray_current.d, wi_theta), AVERAGE_COS))
            fp = 1.0
            incremental = ((ats.attenuation * radiance * fp / dr.detach(fp) & itinfo.active))

        # add the weight to corresponding energy group
        Etot_incremental = dr.zeros(FloatD,  number_of_energy_group)
        dr.scatter_add(Etot_incremental, incremental, itinfo.energy_group_idx)
        Etot += Etot_incremental

        if reparam:
            cross_section_reparam_t = ats.feature.ext * ats.reparam_factor
            pdf = ats.pdf * ats.reparam_factor
            
        else:
            cross_section_reparam_t = ats.feature.ext

        itinfo.active &= (~terminate_ray)
        
        
        # phase function, sample a direction
        # better sample from the source instead of sample from the phase function
        # wo = sample_direction_hg(sceneinfo.rng, itinfo.ray_current.d, 0.0)
        # 2D version, keep z = 0 for all computation
        wo = phase_2d(sample_float_32(sceneinfo.rng))
        # wo.y = dr.zeros(FloatD, dr.width(wo)) + 1.0
        # wo.x = dr.zeros(FloatD, dr.width(wo)) + 2.0
        # wo.x = dr.zeros(FloatD, dr.width(wo)) + 0.0
        wo = wo / dr.norm(wo)


        wi_theta = -itinfo.ray_current.d
        itinfo.energy_group_idx, group_pdf = sample_next_energy_group(sceneinfo.rng, ats.feature.phase_cdfs, sceneinfo.scm.energy_groups)
        
        itinfo.radiance *= (ats.attenuation_till_scatter) * (cross_section_reparam_t * ats.feature.alb) / dr.detach(pdf)
        itinfo.last_reparam *= itinfo.radiance * ats.feature.ext * ats.feature.alb / dr.detach(ats.pdf)
        # beam_weight *= (attenuation_till_scatter) * cross_section_reparam_t * feature.alb / dr.detach(pdf)
        # sample_weight *= attenuation_till_scatter / dr.detach(pdf) * (cross_section_reparam_t * feature.alb)
        # beam_weight *= dr.detach((attenuation_till_scatter / dr.detach(pdf)) * (cross_section_reparam_t * feature.alb))
        #use energy dependent phase function to decide the change of energy group of each particle
        
        # print(energy_group_idx, group_pdf)
        # update ray
        itinfo.ray_current.o = ats.scatter_pos
        itinfo.ray_current.d = wo

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

