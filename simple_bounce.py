import mitsuba as mi
import drjit as dr
from drjit.llvm import Float, UInt32
from drjit.llvm.ad import Float as FloatD

mi.set_variant('llvm_ad_rgb')

NUMBER_NEUTRONS = 100000000
MAX_BOUNCE = 5
TOT_CROSS_SECTION_T = 1.0
TOT_CROSS_SECTION_A = 0.3


# # log information about the source and the tally
# print("energy bins in tally", E_bins)
# print("neutron energy distribution", N_array_E)

# initialize the direction distribution of the neutrons from the source


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
    return cross_section_new, cross_section_new_a, depth / (constant * depth),  cross_section_tot / cross_section_new

def calculate_esacpe_energy(depth, cross_section_tot_t, cross_section_tot_a):
    rng = mi.PCG32(size=NUMBER_NEUTRONS)

    # set up tally
    E_tot = dr.zeros(FloatD)

    # set up source: point source at (-1, 0, 0)
    N_array_E = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1, 0, 0)

    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    # compute the position where the neutron enter the medium
    enter_point = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS)
    active = (N_directions.x > 0.0)
    enter_point.y = 1.0 / N_directions.x * N_directions.y
    enter_point.z = 1.0 / N_directions.x * N_directions.z

    current_point = enter_point
   
    for i in range(MAX_BOUNCE):
        # sample collision event
        dist = dr.detach(sample_distance(cross_section_tot_t, rng))
        remain_dist = compute_rest_dist(N_directions, current_point, depth)
        optical_dist = dr.select(remain_dist <= dist, remain_dist, dist)

        transmittance = dr.exp(-cross_section_tot_t * optical_dist) 
        dist_pdf = dr.detach(dr.select(remain_dist <= dist, dr.exp(-cross_section_tot_t * optical_dist), cross_section_tot_t * dr.exp(-cross_section_tot_t * optical_dist)))

        radiance *= (transmittance / dist_pdf)
        # check if the collision event is in the boundary
        current_point = dist * N_directions + current_point
        
        # accumulate to the tally
        escape_energy = radiance & active & gothrough(current_point, depth)
        E_tot += dr.sum(escape_energy)

        active &= ~escape(current_point, depth)

        radiance *= (cross_section_tot_t - cross_section_tot_a)

        # sample direction 
        N_directions = sample_dir_from_unit_sphere(rng)

        # sample energy
        N_array_E = rng.next_float32() * N_array_E

    return E_tot / NUMBER_NEUTRONS


def calculate_esacpe_energy_reparam_last(depth, constant, cross_section_tot_t, cross_section_tot_a):
    # set up random number generator
    rng = mi.PCG32(size=NUMBER_NEUTRONS)

    # set up tally
    E_tot = dr.zeros(FloatD)

    # set up source: point source at (-1, 0, 0)
    N_array_E = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1, 0, 0)

    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    # compute the position where the neutron enter the medium
    enter_point = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS)
    active = (N_directions.x > 0.0)
    enter_point.y = 1.0 / N_directions.x * N_directions.y
    enter_point.z = 1.0 / N_directions.x * N_directions.z

    # print("point enetery the medium", enter_point & active)
    current_point = enter_point

    for i in range(MAX_BOUNCE):
        # sample collision event
        remain_dist = compute_rest_dist(N_directions, current_point, depth)

        tot_cross_section_reparam_t, tot_cross_section_reparam_a, remain_dist_reparam, jacobian_pdf_reparam = cross_section_nor(cross_section_tot_t, cross_section_tot_a, remain_dist, constant)

        dist_reparam = dr.detach(sample_distance(tot_cross_section_reparam_t, rng))
        
        optical_dist = dr.select(remain_dist_reparam <= dist_reparam, remain_dist_reparam, dist_reparam)

        transmittance = dr.exp(-tot_cross_section_reparam_t * optical_dist) 
        dist_pdf = dr.detach(dr.select(remain_dist_reparam <= dist_reparam, dr.exp(-tot_cross_section_reparam_t * optical_dist), tot_cross_section_reparam_t * dr.exp(-tot_cross_section_reparam_t * optical_dist)))

        radiance *= (transmittance / dist_pdf)

        # check if the collision event is in the boundary
        dist = dist_reparam * remain_dist / remain_dist_reparam
        current_point = dist * N_directions + current_point
        
        # accumulate to the tally
        escape_energy = radiance & active & gothrough(current_point, depth)
        # print("escape", dr.sum(gothrough(current_point, depth) & active))
        E_tot += dr.sum(escape_energy)

        active &= ~escape(current_point, depth)
        radiance *= (tot_cross_section_reparam_t - tot_cross_section_reparam_a)

        # sample direction 
        N_directions = sample_dir_from_unit_sphere(rng)

        # sample energy
        N_array_E = rng.next_float32() * N_array_E

    return E_tot / NUMBER_NEUTRONS


def calculate_esacpe_energy_reparam(depth, constant, cross_section_tot_t, cross_section_tot_a):
    # set up random number generator
    rng = mi.PCG32(size=NUMBER_NEUTRONS)

    # set up tally
    E_tot = dr.zeros(FloatD)

    # set up source: point source at (-1, 0, 0)
    N_array_E = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    radiance = dr.zeros(FloatD, NUMBER_NEUTRONS) + 1.0
    source_origin = mi.Point3f(-1, 0, 0)

    # initialiate the neutrons
    N_directions = sample_dir_from_unit_sphere(rng)

    # compute the position where the neutron enter the medium
    enter_point = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS)
    active = (N_directions.x > 0.0)
    enter_point.y = 1.0 / N_directions.x * N_directions.y
    enter_point.z = 1.0 / N_directions.x * N_directions.z

    # print("point enetery the medium", enter_point & active)
    current_point = enter_point

    tot_cross_section, tot_cross_section_a, depth, jacobian_pdf = cross_section_nor(cross_section_tot_t, cross_section_tot_a, depth, constant)
    # print(jacobian_pdf)

    for i in range(MAX_BOUNCE):
        # sample collision event
        dist = dr.detach(sample_distance(tot_cross_section, rng))
        remain_dist = compute_rest_dist(N_directions, current_point, depth)
        optical_dist = dr.select(remain_dist <= dist, remain_dist, dist)

        transmittance = dr.exp(-tot_cross_section * optical_dist) 
        dist_pdf = dr.detach(dr.select(remain_dist <= dist, dr.exp(-tot_cross_section * optical_dist), tot_cross_section * dr.exp(-tot_cross_section * optical_dist)))

        radiance *= (transmittance / dist_pdf)
        # check if the collision event is in the boundary
        current_point = dist * N_directions + current_point
        
        # accumulate to the tally
        escape_energy = radiance & active & gothrough(current_point, depth)
        # print("escape", dr.sum(gothrough(current_point, depth) & active))
        E_tot += dr.sum(escape_energy)

        active &= ~escape(current_point, depth)

        radiance *= (tot_cross_section - tot_cross_section_a) 
        # (TOT_CROSS_SECTION_T - TOT_CROSS_SECTION_A)

        # sample direction 
        N_directions = sample_dir_from_unit_sphere(rng)

        # sample energy
        N_array_E = rng.next_float32() * N_array_E

        # radiance *= 

    return E_tot / NUMBER_NEUTRONS


st = TOT_CROSS_SECTION_T
sa = TOT_CROSS_SECTION_A

R = FloatD(0.5)
dr.enable_grad(R)
constatnt_scale = 0.1
Etot = calculate_esacpe_energy_reparam(R, constatnt_scale, st, sa)
print("flux reparam1", Etot)
dr.set_grad(R, 1.0)
dr.forward_to(Etot)
dEdR = dr.grad(Etot)
print("autodiff gradient reparam1", dEdR)
print("\n")

constatnt_scale = 1.0
Etot2 = calculate_esacpe_energy_reparam_last(R, constatnt_scale, st, sa)
print("flux reparam2", Etot2)
dr.set_grad(R, 1.0)
dr.forward_to(Etot2)
dEdR = dr.grad(Etot2)
print("autodiff gradient reparam2", dEdR)
print("\n")

delta = 0.0001
E_up = calculate_esacpe_energy(R+delta, st, sa)
E_down = calculate_esacpe_energy(R-delta, st, sa)

print("flux+", E_up)
print("flux-", E_down)
print("fd gradients", (E_up - E_down) / (2 * delta))
print("\n")
