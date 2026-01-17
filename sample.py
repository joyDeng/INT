import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64, UInt, TensorXf, Float64 as Float64
from drjit.cuda.ad import Float as FloatD, Float64 as Float64D
from drjit.cuda.ad import UInt32 as UIntD, TensorXf as TensorXfD
mi.set_variant('cuda_ad_rgb')

def sample_sensor_pos(sensor, rng, active):
    si, area = sensor.sample_position(0.0, [sample_float_32(rng), sample_float_32(rng)], active)
    
    return si

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

def sample_distance(sig_t, rng):
    rnd1 = sample_float_32(rng)
    dr.eval(rnd1, rng)
    distance = - dr.log(1.0 - rnd1) / dr.detach(sig_t)
    return distance