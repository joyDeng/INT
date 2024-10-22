
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

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR


NUMBER_NEUTRONS = 100000000
MAX_BOUNCE = 5
TOT_CROSS_SECTION_T = 1.0
TOT_CROSS_SECTION_A = 0.1

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
    def __init__(self, height, width, origin):
        self.height = height
        self.width = width
        self.origin = origin

        self.xplane_left = self.origin.x - width * 0.5
        self.xplane_right =  self.origin.x + width * 0.5

        self.yplane_left = self.origin.y - height * 0.5
        self.yplane_right =  self.origin.y + height * 0.5



    # ray geometry intersection
    def intersect(self, ray):
        intersect = True
        valid_1 = True
        valid_2 = True

        # intersect with x planes
        t_x_1 = (self.xplane_left - ray.origin.x) / ray.direction.x
        t_x_2 = (self.xplane_right - ray.origin.x) / ray.direction.x

        ytx1 = ray.origin.y + t_x_1 * ray.direction.y
        ytx2 = ray.origin.y + t_x_2 * ray.direction.y

        yinrange_1 = inrange(ytx1, self.yplane_left, self.yplane_right)
        yinrange_2 = inrange(ytx2, self.yplane_left, self.yplane_right)

        valid_2 &= ((t_x_2 > 0.0) & yinrange_2)
        valid_1 &= ((t_x_1 > 0.0) & yinrange_1)

        tx = dr.select(valid_1 & valid_2, dr.select(t_x_1 > t_x_2, t_x_2, t_x_1), dr.select(valid_1, t_x_1, dr.select(valid_2, t_x_2, -1.0)))

        # intersect with y planes
        valid_1 = True
        valid_2 = True

        t_y_1 = (self.yplane_left - ray.origin.y) / ray.direction.y
        t_y_2 = (self.yplane_right - ray.origin.y) / ray.direction.y

        xty1 = ray.origin.x + t_y_1 * ray.direction.x
        xty2 = ray.origin.x + t_y_2 * ray.direction.x

        xinrange_1 = inrange(xty1, self.xplane_left, self.xplane_right)
        xinrange_2 = inrange(xty2, self.xplane_left, self.xplane_right)

        valid_2 &= ((t_y_2 > 0.0) & xinrange_2)
        valid_1 &= ((t_y_1 > 0.0) & xinrange_1)

        ty = dr.select(valid_1 & valid_2, dr.select(t_y_1 > t_y_2, t_y_2, t_y_1), dr.select(valid_1, t_y_1, dr.select(valid_2, t_y_2, -1.0)))

        t = dr.select((tx > 0.0) & (ty > 0.0), dr.select(tx > ty, ty, tx), dr.select(tx > 0.0, tx, dr.select(ty > 0.0, ty, -1.0)))
        intersect &= (t > 0.0)

        # compute coordintate of intersection
        point = ray.origin + t * ray.direction
        u = (point.x - self.xplane_left) / self.width
        v = (point.y - self.yplane_left) / self.height

        return intersect, t, mi.Vector2f(u,v)

    def compute_rest_dist(self, uv, point):
        p = mi.Vector3f(uv.x * self.width + self.xplane_left, uv.y * self.height + self.yplane_left, self.origin.z)
        direction_vec = p - point
        return dr.norm(direction_vec)



class Tally:
    def __init__(self, position, width):
        self.position = position
        self.width = width
        self.y_right = self.position.y + self.width * 0.5
        self.y_left = self.position.y - self.width * 0.5

        self.z_right = self.position.z +self.width * 0.5
        self.z_left = self.position.z -self.width * 0.5

        self.area = self.width * self.width
        self.normal = mi.Vector3f(-1.0, 0.0, 0.0)

    def samplePoint(self, rng):
        rnd1, rnd2 = rng.next_float32(), rng.next_float32()
        point = self.position + mi.Vector3f(0.0, (rnd1 - 0.5) * self.width, (rnd2 - 0.5) * self.width) 
        pdf = 1.0 / (self.area)
        return point, pdf

    def intersect(self, ray):
        t = (self.position.x - ray.origin.x) / ray.direction.x
        y = ray.direction.y * t + ray.origin.y
        inrangey = inrange(y, self.y_left, self.y_right)
        

        # try to make the tally infinite in z dimension
        z = ray.direction.z * t + ray.origin.z
        inrangez = inrange(z, self.z_left, self.z_right)

        intersect = (inrangey & (t > 0.0) & inrangez)

        uv = mi.Vector2f((y - self.y_left) / self.width, (z - self.z_left) / self.width)
        return intersect, t, uv

 
        

def test_intersect():
    shield = RectShield(1.0, 1.0, mi.Vector3f(0.0, 0.0, 0.0))
    r1y = Ray(mi.Vector3f(1.00, 0.0, 0.0), mi.Vector3f(-1.0, 0.0, 0.0))
    intersect, t, uv = shield.intersect(r1y)
    print(intersect, t, uv)


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
def escape(point, d):
    return (point.x <= 0.0) | (point.x >= d)

# RETURN True if the nuetron exit from the x=0 plane
def goback(point, d):
    return point.x <= 0.0

# RETURN True if the nuetron exit from the x=d plane
def gothrough(point, d):
    return point.x >= d

# RETURN a float distance that is the closest hit from current point to 
# the surface alone current direction
def compute_rest_dist(N_directions, current_point, depth):
    x_dist = dr.select(N_directions.x > 0.0, depth - current_point.x, current_point.x)
    cos_theta = dr.abs(N_directions.x)
    rest_dist = x_dist / cos_theta
    return rest_dist

#  RETURN reparameterize cross_section values 
def cross_section_nor(cross_section_tot, cross_section_tot_a, depth, constant):
    ltot_n = (1.0 / cross_section_tot) / (constant * depth)
    ltot_a = (1.0 / cross_section_tot_a) / (constant * depth)
    cross_section_new = 1.0 / ltot_n
    cross_section_new_a = 1.0 / ltot_a
    return cross_section_new, cross_section_new_a, depth / (constant * depth),  depth

def balance(pdf1, pdf2):
    return pdf1 / (pdf1 + pdf2)


def calculate_tally_energy(tally, sheilding, cross_section_tot_t, cross_section_tot_a):
    rng = mi.PCG32(size=NUMBER_NEUTRONS)

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
    p0 = ray_init.origin + ray_init.direction * t

    # for the ray that didn't intersect with the scene, check intersection with tally
    intersect_tally, tt, uvt = tally.intersect(ray_init)
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
        hittally, ttally, uvtally = tally.intersect(ray_continue)
        cos_theta = dr.abs(dr.dot(ray_continue.direction, tally.normal))
        pdf_tally = (1.0 / tally.area) / cos_theta / (ttally * ttally)
        # weight1 = balance(1.0 / (4.0 * dr.pi), pdf_tally)
        # accumulate to the tally
        arrive_energy = (radiance) & active & hittally & escape
        E_tot += dr.sum(arrive_energy)

        active &= ~escape

        # connect to the tally (TODO: Multiple Important sampling)
        pt, pdf = tally.samplePoint(rng)
        connect_direction = (pt - ray_current.origin)
        ray_connect = Ray(ray_current.origin, connect_direction)
        exit_point, travel_dist, uv = sheilding.intersect(ray_connect)

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


def calculate_tally_energy_reparam(tally, sheilding, cross_section_tot_t, cross_section_tot_a):
    rng = mi.PCG32(size=NUMBER_NEUTRONS)

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
    trecompute = sheilding.compute_rest_dist(uv, source_origin)
    p0 = ray_init.origin + ray_init.direction * t

    # for the ray that didn't intersect with the scene, check intersection with tally
    intersect_tally, tt, uvt = tally.intersect(ray_init)
    active = True
    # add contribution (TODO: add jacobian for perpendicular area)
    arrive_energy = radiance & active & (~intersect & intersect_tally)
    E_tot += dr.sum(arrive_energy)
    active &= intersect

    ray_current = Ray(p0, ray_init.direction)

    for i in range(MAX_BOUNCE):
        
        intersect, r1, uv = sheilding.intersect(ray_current)
        remain_dist = sheilding.compute_rest_dist(uv, ray_current.origin)
        
        tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_reparam = cross_section_nor(cross_section_tot_t, cross_section_tot_a, remain_dist, 1.0)
        # remain_dist_reparam = remain_dist / sheilding.height
        dist_reparam = dr.detach(sample_distance(tot_cross_section_reparam_t, rng))

        optical_dist = dr.select(remain_dist_reparam <= dist_reparam, remain_dist_reparam, dist_reparam)

        transmittance = dr.exp(-tot_cross_section_reparam_t * remain_dist) 
        dist_pdf = dr.detach(dr.select(remain_dist_reparam <= dist_reparam, dr.exp(-tot_cross_section_reparam_t * optical_dist), tot_cross_section_reparam_t * dr.exp(-tot_cross_section_reparam_t * optical_dist)))

        # update current position of the neutron
        radiance *= (transmittance / dist_pdf)
        # * jacobian_reparam
        dist = dist_reparam * remain_dist_reparam / remain_dist
        # print(dist, r1)
        p0 = dist * ray_current.direction + p0

        # whether the escape neturon hit the tally (intersect with tally)
        escape = (dist > remain_dist)
        ray_continue = Ray(p0, ray_current.direction)
        hittally, ttally, uvtally = tally.intersect(ray_continue)

        cos_theta = dr.abs(dr.dot(ray_continue.direction, tally.normal))
        pdf_tally = (1.0 / tally.area) / cos_theta / (ttally * ttally)
        # weight1 = balance(1.0 / (4.0 * dr.pi), pdf_tally)
        # accumulate to the tally
        arrive_energy = (radiance) & active & hittally & escape
        E_tot += dr.sum(arrive_energy)

        active &= ~escape

        # connect to the tally (TODO: Multiple Important sampling)
        pt, pdf = tally.samplePoint(rng)
        connect_direction = (pt - ray_current.origin)
        ray_connect = Ray(ray_current.origin, connect_direction)
        exit_point, travel_dist, uv = sheilding.intersect(ray_connect)


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
        radiance *= (tot_cross_section_reparam_t - tot_cross_section_reparam_a)
        # * 4.0 * dr.pi

        ray_current.origin = p0
        ray_current.direction = sample_dir_from_unit_sphere(rng)

        # sample energy
        # N_array_E = rng.next_float32() * N_array_E

    return E_tot / NUMBER_NEUTRONS

def energy_tally_height(height):
    my_tally = Tally(mi.Vector3f(1.0, 0.0, 0.0), 1.0)
    my_shield = RectShield(FloatD(height), FloatD(0.5), mi.Vector3f(0.0, 0.0, 0.0))
    energy = calculate_tally_energy(my_tally, my_shield, TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A)
    return energy

def energy_tally_height_reparam(height):
    my_tally = Tally(mi.Vector3f(1.0, 0.0, 0.0), 1.0)
    my_shield = RectShield(FloatD(height), FloatD(0.5), mi.Vector3f(0.0, 0.0, 0.0))
    energy = calculate_tally_energy_reparam(my_tally, my_shield, TOT_CROSS_SECTION_T, TOT_CROSS_SECTION_A)
    return energy

def energy_finite_different(height, delta):
    e1 = energy_tally_height(height+delta)
    e2 = energy_tally_height(height-delta)
    return (e1 - e2) / (2.0 * delta)

def test_increase_height(smallest, largest, stepsize):
    height = smallest
    list_energy = []
    list_gradient = []
    heights = []
    while height < largest:
        print("height", height)
        Energy = energy_tally_height(height)
        FD_gradient = energy_finite_different(height, 0.001)
        
        list_gradient.append(FD_gradient.numpy())
        list_energy.append(Energy.numpy())
        heights.append(height)
        height += stepsize
    energies = np.array(list_energy)
    print(heights)
    heights = np.array(heights)
    gradients_fd = np.array(list_gradient)
    return energies, heights, gradients_fd


def compute_auto_def_gradient(smallest, largest, stepsize):
    height = smallest
    list_energy = []
    list_gradient = []
    heights = []
    while height < largest:
        print("ad height", height)

        H = FloatD(height)
        dr.enable_grad(H)
        Energy = energy_tally_height_reparam(H)
        dr.set_grad(H, 1.0)
        dr.forward_to(Energy)
        dEdR = dr.grad(Energy)
        
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
    hh = 2.0
    step = 0.05

    energy_variation_ad, heights_ad, gfd_ad = compute_auto_def_gradient(hl, hh, step)
    np.save(DATA_DIR + "energy_variation_ad.npy", energy_variation_ad)
    np.save(DATA_DIR + "shield_height_ad.npy", heights_ad)
    np.save(DATA_DIR + "gradients_fd_ad.npy", gfd_ad)

    energy_variation, heights, gfd = test_increase_height(hl, hh, step)
    np.save(DATA_DIR + "energy_variation.npy", energy_variation)
    np.save(DATA_DIR + "shield_height.npy", heights)
    np.save(DATA_DIR + "gradients_fd.npy", gfd)

test_fd_ad()