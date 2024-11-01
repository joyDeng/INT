
# used only on my windows machine
import sys
sys.path = ["."] + sys.path[2:]

# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32
from drjit.cuda.ad import Float as FloatD
import numpy as np
import random


mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR


# print(type(mi.Float))

NUMBER_NEUTRONS = 100000000
MAX_BOUNCE = 3
TOT_CROSS_SECTION_T = 0.5
TOT_CROSS_SECTION_A = 0.05
AVERAGE_COS = mi.Float(0.9)

def equal(x, y):
    return dr.abs(x - y) < 1e-7

def inrange(x, left, right):
    # equalleft = (dr.abs(x - left) < 1e-7)
    # equalright = (dr.abs(x - right) < 1e-7)
    return (equal(x, left) | (x > left)) & ((x < right) | equal(x, right))

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / dr.norm(direction)
        # print(self.direction)
        self.t = dr.zeros(FloatD) + dr.inf
        # print(origin.Shape)

class RectShield:
    def __init__(self, height, width, depth, origin):
        self.height = height
        self.width = width
        self.depth = depth
        self.origin = origin

        self.xplane_left = self.origin.x - width * 0.5
        self.xplane_right =  self.origin.x + width * 0.5

        self.yplane_left = self.origin.y - height * 0.5
        self.yplane_right =  self.origin.y + height * 0.5

        self.zplane_left = self.origin.z - depth * 0.5
        self.zplane_right =  self.origin.z + depth * 0.5


    def inrange(self, p, axis=0):
        xi = inrange(p.x, self.xplane_left, self.xplane_right)
        yi = inrange(p.y, self.yplane_left, self.yplane_right)
        zi = inrange(p.z, self.zplane_left, self.zplane_right)
        return xi & yi & zi
        # if axis == 0:
        #     return yi & zi
        # elif axis == 1:
        #     return xi & zi
        # elif axis == 2:
        #     return xi & yi
        # else:
        #     return False

    # ray geometry intersection
    def intersect(self, ray):
        
        intersect = True
        valid_1 = True
        valid_2 = True

        # intersect with x planes
        t_x_1 = (self.xplane_left - ray.origin.x) / ray.direction.x
        t_x_2 = (self.xplane_right - ray.origin.x) / ray.direction.x

        p1 = ray.origin + t_x_1 * ray.direction
        p2 = ray.origin + t_x_2 * ray.direction

        # yinrange_1 = 
        # yinrange_2 = 

        valid_2 &= ((t_x_2 > 0.0) & self.inrange(p2))
        valid_1 &= ((t_x_1 > 0.0) & self.inrange(p1))

        tx = dr.select(valid_1 & valid_2, dr.select(t_x_1 > t_x_2, t_x_2, t_x_1), dr.select(valid_1, t_x_1, dr.select(valid_2, t_x_2, -1.0)))

        # intersect with y planes
        valid_1 = True
        valid_2 = True

        t_y_1 = (self.yplane_left - ray.origin.y) / ray.direction.y
        t_y_2 = (self.yplane_right - ray.origin.y) / ray.direction.y

        p1 = ray.origin + t_y_1 * ray.direction
        p2 = ray.origin + t_y_2 * ray.direction

        valid_2 &= ((t_y_2 > 0.0) & self.inrange(p2))
        valid_1 &= ((t_y_1 > 0.0) & self.inrange(p1))

        ty = dr.select(valid_1 & valid_2, dr.select(t_y_1 > t_y_2, t_y_2, t_y_1), dr.select(valid_1, t_y_1, dr.select(valid_2, t_y_2, -1.0)))

        # intersect with z planes
        valid_1 = True
        valid_2 = True

        t_z_1 = (self.zplane_left - ray.origin.z) / ray.direction.z
        t_z_2 = (self.zplane_right - ray.origin.z) / ray.direction.z

        p1 = ray.origin + t_z_1 * ray.direction
        p2 = ray.origin + t_z_2 * ray.direction

        valid_2 &= ((t_z_2 > 0.0) & self.inrange(p2))
        valid_1 &= ((t_z_1 > 0.0) & self.inrange(p1))

        tz = dr.select(valid_1 & valid_2, dr.select(t_z_1 > t_z_2, t_z_2, t_z_1), dr.select(valid_1, t_z_1, dr.select(valid_2, t_z_2, -1.0)))

        t = dr.select((tx > 0.0) & (ty > 0.0), dr.select(tx > ty, ty, tx), dr.select(tx > 0.0, tx, dr.select(ty > 0.0, ty, -1.0)))
        t = dr.select((tz > 0.0) & (t > 0.0), dr.select(tz > t, t, tz), dr.select(tz > 0.0, tz, dr.select(t > 0.0, t, -1.0)))

        intersect &= (t > 0.0)

        # costheta = dr.abs(dr.dot(ray.direction, dr.norm(mi.Vector3f(0.0, self.height, 0.0))))
        # compute coordintate of intersection
        point = ray.origin + t * ray.direction 
        u = (point.x - self.xplane_left) / self.width
        v = (point.y - self.yplane_left) / self.height
        z = (point.z - self.zplane_left) / self.depth
        return intersect, t, mi.Vector3f(u,v,z)

    def compute_rest_dist(self, uv, point):
        p = mi.Vector3f(uv.x * self.width + self.xplane_left, uv.y * self.height + self.yplane_left, uv.z * self.depth + self.zplane_left)
        direction_vec = p - point
        return dr.norm(direction_vec)



class Tally:
    def __init__(self, position, width):
        self.position = position
        self.width = width
        self.y_right = self.position.y + self.width * 0.5
        self.y_left = self.position.y - self.width * 0.5

        self.z_right = self.position.z + self.width * 0.5
        self.z_left = self.position.z - self.width * 0.5

        self.area = self.width * self.width
        self.normal = mi.Vector3f(-1.0, 0.0, 0.0)

    def samplePoint(self, rng):
        rnd1, rnd2 = rng.next_float32(), rng.next_float32()
        point = self.position + mi.Vector3f(0.0, (rnd1 - 0.5) * self.width, (rnd2 - 0.5) * self.width) 
        pdf = 1.0 / (self.area)
        return point, pdf

    def intersect(self, ray):
        t = (self.position.x - ray.origin.x) / ray.direction.x
        return ray.direction.x > 0.0, t

        # p = ray.origin + t * ray.direction
        # # y = ray.direction.y * t + ray.origin.y
        # inrangey = inrange(p.y, self.y_left, self.y_right)
        

        # # try to make the tally infinite in z dimension
        # # z = ray.direction.z * t + ray.origin.z
        # inrangez = inrange(p.z, self.z_left, self.z_right)

        # intersect = (inrangey & (t > 0.0) & inrangez)

        # # uv = mi.Vector3f((y - self.y_left) / self.width, (z - self.z_left) / self.width, z)
        # return intersect, t
        
        # , uv

 
        

def test_intersect():
    shield = RectShield(1.0, 1.0, mi.Vector3f(0.0, 0.0, 0.0))
    r1y = Ray(mi.Vector3f(1.00, 0.0, 0.0), mi.Vector3f(-1.0, 0.0, 0.0))
    intersect, t, uv = shield.intersect(r1y)
    print(intersect, t, uv)


def hg(costheta, g):
    demon = 1.0 + g * g + 2.0 * g * costheta
    return 1.0 / (4.0 * dr.pi) * (1.0 - g * g ) / (demon * dr.sqrt(demon)); 

def sample_direction_hg(rng, wi, g):
    sample1, sample2 = rng.next_float32(), rng.next_float32()
    sqrTerm = (1.0 - g * g ) / (  1.0 - g + 2.0 * g * sample1)
    # print(g < 1e-3)
    cosTheta  =  dr.select(g < 1e-3,  1.0 - 2.0 * sample1, (1.0 + g * g - sqrTerm * sqrTerm) / (2.0 * g) )

    phi = 2.0 * dr.pi * sample2
    sinThetaSqr = 1.0 - cosTheta * cosTheta
    sinThetaSqr[sinThetaSqr <= 0.0] = 0.0
    sinTheta = dr.sqrt(sinThetaSqr)
    wo = mi.Frame3f(wi).to_world(mi.Vector3f(sinTheta * dr.cos(phi), sinTheta * dr.sin(phi), cosTheta))
    return wo


def sample_dir_from_unit_sphere(rng):
    v = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS)
    sample1, sample2 = rng.next_float32(), rng.next_float32()
    v.z = 1.0 - sample1
    v.x  = sample1 * dr.sin(dr.pi * 2.0 * sample2)
    v.y = sample1 * dr.cos(dr.pi * 2.0 * sample2)
    return v

# RETURN a float distance that is sampled proportional to the transmittance term
def sample_distance(sig_t, rng):
    distance = - dr.log(1.0 - rng.next_float32()) / sig_t
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
    ltot_n = (1.0 / cross_section_tot) / (constant * depth)
    ltot_a = (1.0 / cross_section_tot_a) / (constant * depth)
    cross_section_new = 1.0 / ltot_n
    cross_section_new_a = 1.0 / ltot_a
    return cross_section_new, cross_section_new_a, depth / (constant * depth),  depth

def balance(pdf1, pdf2):
    return pdf1 / (pdf1 + pdf2)


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

    return t, mi.Vector2f(u, v), intersect

    p = e1 * its.prim_uv.x + e2 * its.prim_uv.y

    return p







def calculate_tally_energy_reparam(tally, height, cross_section_tot_t, cross_section_tot_a, seed, reparam=True):
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
    V.y = V.y + height
    params['shield.vertex_positions'] = dr.ravel(V)
    dr.enable_grad(params['shield.vertex_positions'])
    params.update()
    
    # print(params)
    # exit(0)
    ray_init = mi.Ray3f(source_origin, N_directions)
    its = scene.ray_intersect(ray_init)
    interdist, uv, activei = recomputeIntersection(scene, its, V, F, ray_init, its.is_medium_transition())
    
    # exit(0)
    # p0 = dr.detach(ray_init.origin + ray_init.direction * (t + 0.000001))

    # for the ray that didn't intersect with the scene, check intersection with tally
    intersect_tally = (~dr.isinf(its.t)) & (~its.is_medium_transition())
    intersect_medium = (~dr.isinf(its.t)) & (its.is_medium_transition())
    active = True
    # add contribution (TODO: add jacobian for perpendicular area)
    arrive_energy = radiance & active & intersect_tally
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
    my_tally = Tally(mi.Vector3f(1.0, 0.0, 0.0), 2.0)
    # my_shield = RectShield(FloatD(height), FloatD(0.5), FloatD(2.0), mi.Vector3f(0.0, 0.0, 0.0))
    energy = calculate_tally_energy_reparam(my_tally, FloatD(height), TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A, seed, False)
    return energy

def energy_tally_height_reparam(height, reparam):
    my_tally = Tally(mi.Vector3f(1.0, 0.0, 0.0), 2.0)
    # my_shield = RectShield(FloatD(height), FloatD(0.5), FloatD(2.0), mi.Vector3f(0.0, 0.0, 0.0))
    energy = calculate_tally_energy_reparam(my_tally, FloatD(height), TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A, 0, reparam)
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
        Energy = energy_tally_height(height)
        FD_gradient = energy_finite_different(height, 0.002, 0) / (N+1)
        start = random.randint(0,1000)
        for i in range(start, start+N):
            difference = energy_finite_different(height, 0.002, i) / (N+1)
            dr.eval(difference)
            FD_gradient += difference
            del difference
            
        list_gradient.append(FD_gradient.numpy())
        list_energy.append(Energy.numpy())
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
    while height < largest:
        print("ad height", height)

        H = FloatD(height)
        dr.enable_grad(H)
        Energy = energy_tally_height_reparam(H, reparam)
        # dr.set_grad(H, 1.0)
        # dr.forward_to(Energy)
        dr.backward(Energy)
        dEdR = dr.grad(H)
        
        list_gradient.append(dEdR.numpy())
        # print(dEdR)
        # list_gradient.append(1.0)
        list_energy.append(Energy.numpy())
        heights.append(height)
        height += stepsize
    energies = np.array(list_energy)
    heights = np.array(heights)
    gradients_fd = np.array(list_gradient)
    return energies, heights, gradients_fd

def test_fd_ad():
    hl = 0.1
    hh = 1.2
    step = 0.023

    

    energy_variation_ad, heights_ad, gfd_ad = compute_auto_def_gradient(hl, hh, step, reparam=False)
    np.save(DATA_DIR + "energy_variation_ad.npy", energy_variation_ad)
    np.save(DATA_DIR + "shield_height_ad.npy", heights_ad)
    np.save(DATA_DIR + "gradients_fd_ad.npy", gfd_ad)

    energy_variation_ad, heights_ad, gfd_ad = compute_auto_def_gradient(hl, hh, step, reparam=True)
    np.save(DATA_DIR + "energy_variation_ad_reparam.npy", energy_variation_ad)
    np.save(DATA_DIR + "shield_height_ad_reparam.npy", heights_ad)
    np.save(DATA_DIR + "gradients_fd_ad_reparam.npy", gfd_ad)

    energy_variation, heights, gfd = test_increase_height(hl, hh, step)
    np.save(DATA_DIR + "energy_variation.npy", energy_variation)
    np.save(DATA_DIR + "shield_height.npy", heights)
    np.save(DATA_DIR + "gradients_fd.npy", gfd)

test_fd_ad()