
# used only on my windows machine
import sys
# print(sys.path)
# exit(0)
# sys.path = ["."] + sys.path[2:]
import matplotlib.pyplot as plt


# need to change back to llvm on my mac
from sample import *
import numpy as np
import random
import torch

from csg import CSGLeaf, CSGNode, SceneMaterial, MaterialParameter, MultiGroupTally, scene_material_intersect, Beams, SceneInfo, AttenSample, IterProp


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



def recomputeIntersection(scene, its, vertex, f, ray, active, debug=False, id=0):
    intersect = True & active
    
    faces = dr.gather(mi.Vector3i, f, its.prim_index, active)

    p0 = dr.gather(mi.Point3f, vertex, faces.x, active)
    p1 = dr.gather(mi.Point3f, vertex, faces.y, active)
    p2 = dr.gather(mi.Point3f, vertex, faces.z, active)

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
    
    p = e1 * u + e2 * v + p0
    intersect |= its.is_valid()

    
    return t, mi.Vector2f(u, v), p, intersect

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

# dts = compute_attenuation_between_current_and_sensor(sceneinfo, dall_its, d_material_spaces, itinfo, ray_direct, direct_maxt)
def compute_attenuation_between_current_and_sensor(sceneinfo, dall_its, d_material_spaces, itinfo, ray_direct, max_t):
    """
     Return:
        the attenuation of the energy between the sensor and currentpoint
    """
    ray = mi.Ray3f(ray_direct)
    num_rays = dr.width(ray)
    attenuation = dr.ones(FloatD, num_rays)
    current_distance = dr.zeros(FloatD, num_rays)
    sample_active = dr.copy(itinfo.active)
    arrived = False
    

    for intersect, current_material in zip(dall_its, d_material_spaces[:-1]):
        # recompute the intersection with the gradient attached
        cur_is_valid = intersect[1] 
        distance, p, valid_intersect = recompute_intersect_csg(sceneinfo.scene, intersect[0], sceneinfo.vertices_list, sceneinfo.faces_list, ray, intersect[2], cur_is_valid)
        dr.eval()
        
        temp_dist = current_distance + distance
        arrived = dr.select(temp_dist < max_t, False, True) & sample_active
        distance_update = dr.select(temp_dist < max_t, distance, max_t - current_distance)

        material_idx = UInt(current_material)
        in_medium = ~ (material_idx == sceneinfo.scm.num_material)
        material_idx = dr.select(in_medium, material_idx, 0)
        
        # quantities for integral over voxels
        cross_section_query = dr.select(in_medium, sceneinfo.scm.get_sig(material_idx, itinfo.energy_group_idx), 0.0)
        update_attenuation = dr.select(in_medium, dr.exp(-distance_update * cross_section_query), 1.0)

        attenuation *= dr.select(sample_active, update_attenuation, 1.0)
        sample_active &= ~(arrived)
        current_distance += distance
        
        # update the ray origin
        ray.o += distance * ray.d

    return attenuation

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

    reparam_constant = dr.zeros(FloatD, num_rays)
    distance = dr.zeros(FloatD, num_rays)
    
    last_distance = dr.zeros(FloatD, num_rays)
    cross_boundary_constant = dr.ones(FloatD, num_rays)
    same_material_mask = True

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
            vaccum_beams = Beams(mi.Point3f(itinfo.ray_current.o), mi.Point3f(dr.detach(itinfo.ray_current.o + dr.inf * ray.d)), ray.d, dr.inf, dr.detach(tv & active), dr.zeros(Float, num_rays), energy_per_ray, bounceIdx)
            beams.append(vaccum_beams)
        exit_beams = Beams(mi.Point3f(ray.o), mi.Point3f(ray.o + dr.inf * ray.d), ray.d, dr.inf, dr.detach(sample_active & itinfo.active & (~numerical_mask)), reparam_constant, energy_per_ray, bounceIdx)
        beams.append(exit_beams)
    exit_ray = sample_active | tv

    # ref the definiation of attenuation sample:
    # class AttenSample:
    # def __init__(self, attenuation, reparam_factor, attenuation_till_scatter, pdf, scatter_pos, feature, exit_ray, mask_invalid, scatter_material_idx):
    ats = AttenSample(attenuation, reparam_factor, attenuation_till_scatter_point, pdf, scatter_pos, features, exit_ray, numerical_mask, scatter_material_idx)
    return  ats

def create_query_ray(sceneinfo, itinfo, si):
    origin = itinfo.ray_current.o
    direct = (si.p - origin)
    # print("si.p: ", si.p)
    # print("ray_o: ", origin)
    length = dr.norm(direct)
    direct /= length
    ray_direct = mi.Ray3f(origin, direct)
    # print("ray_origin:", origin)
    # print("point on light: ", si.p)
    # print("ray_direct", ray_direct)
    its = sceneinfo.sensor.ray_intersect(mi.Ray3f(ray_direct, length), itinfo.active)
    inter_length = dr.select(its.is_valid(), dr.norm(its.p - si.p), 0.0)
    occluded = dr.select((inter_length < 0.000001), False, True)
    # print(dr.compress(occluded))
    return ray_direct, occluded, length

def render_neutron_in_csg_shape_energy_dependent_with_sensor(sceneinfo, ray_current, reparam, beams=[], save_beam=False):
    """
    Energy dependent version of 'render_nuetron_in_csg_shape'
    """
    assert sceneinfo.sensor != None
    # initialize energy tallies with width equals to the number of energy groups
    num_neutron = dr.width(ray_current)
    number_of_energy_group = sceneinfo.scm.energy_groups
    Etot = dr.zeros(FloatD, number_of_energy_group)
    radiance = dr.ones(FloatD, num_neutron)
    sample_weight = dr.ones(FloatD, num_neutron)
    energy_group_idx = dr.zeros(UInt, num_neutron) # from energy group idx 0 to n, the energy goes from high to low
    active = True
    fp_continue = dr.ones(FloatD, num_neutron)
    # travel_trough_material_boundary = False

    bounceIdx = 0
    # 
    # class IterProp:
    # def __init__(self, radiance, energy_group_idx, ray_current, active)
    itinfo = IterProp(radiance, energy_group_idx, ray_current, active)

    while bounceIdx < MAX_BOUNCE:
        # sample direct ray towards sensor

        si =  sample_sensor_pos(sceneinfo.sensor.emitters()[0], sceneinfo.rng, itinfo.active)
        # create query ray
        # print("\n \n", si.p)
        
        ray_direct, occluded, direct_maxt = create_query_ray(sceneinfo, itinfo, si)
        # exit(0)
        cos_theta = dr.dot(si.n, -ray_direct.d)
        cos_theta = dr.select(cos_theta < 0.0, 0.0, cos_theta)
        dall_its, d_material_spaces = dr.detach(scene_material_intersect(sceneinfo.scene, ray_direct, sceneinfo.scm, itinfo.active))
        dr.eval()

        direct_attenuation = compute_attenuation_between_current_and_sensor(sceneinfo, dall_its, d_material_spaces, itinfo, ray_direct, direct_maxt)
        direct_contri = direct_attenuation * cos_theta / (direct_maxt * direct_maxt) / dr.detach(si.pdf)

        # sample continue ray
        all_its, material_spaces = dr.detach(scene_material_intersect(sceneinfo.scene, itinfo.ray_current, sceneinfo.scm, itinfo.active))
        dr.eval()

        ats = sample_and_compute_attenuation_along_ray(sceneinfo, all_its, material_spaces, 
                                                                  itinfo, beams, bounceIdx, save_beam)
        terminate_ray = ats.mask_invalid | ats.exit_ray

        if bounceIdx == 0:
            fp = INV_FOUR_PI
            # incremental = (direct_contri * itinfo.radiance) & (itinfo.active & ~occluded)
        else:
            fp = hg(dr.dot(ray_direct.d, wi_theta), AVERAGE_COS)
            
            fp_continue = hg(dr.dot(itinfo.ray_current.d, wi_theta), AVERAGE_COS)
            
        incremental = (direct_contri * itinfo.radiance * fp) & (itinfo.active & ~occluded)
        Etot_incremental = dr.zeros(FloatD,  number_of_energy_group)
        dr.scatter_add(Etot_incremental, incremental, itinfo.energy_group_idx)
        
        Etot += Etot_incremental

        if reparam:
            cross_section_reparam_t = ats.feature.ext * ats.reparam_factor
            pdf = ats.pdf * ats.reparam_factor
        else:
            cross_section_reparam_t = ats.feature.ext

        itinfo.active &= (~terminate_ray)

        wo = sample_direction_hg(sceneinfo.rng, itinfo.ray_current.d, AVERAGE_COS)
        wo = wo / dr.norm(wo)

        wi_theta = -itinfo.ray_current.d
        itinfo.energy_group_idx, group_pdf = sample_next_energy_group(sceneinfo.rng, ats.feature.phase_cdfs, sceneinfo.scm.energy_groups)
        # print("nextbounceIdx", bounceIdx, "energy_group_idx",  itinfo.energy_group_idx)
        # print("\n\n")

        itinfo.radiance *= (ats.attenuation_till_scatter) * (cross_section_reparam_t * ats.feature.alb) * fp_continue / dr.detach(pdf * fp_continue)

        itinfo.ray_current.o = ats.scatter_pos
        itinfo.ray_current.d = wo

        bounceIdx += 1
    return Etot / num_neutron

def render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, reparam, beams=[], save_beam=False):
    """
    Energy dependent version of 'render_nuetron_in_csg_shape'
    """
    # initialize energy tallies with width equals to the number of energy groups
    num_neutron = dr.width(ray_current)
    number_of_energy_group = sceneinfo.scm.energy_groups
    Etot = dr.zeros(FloatD, number_of_energy_group)
    radiance = dr.zeros(FloatD, num_neutron) + 1.0 
    sample_weight = dr.zeros(FloatD, dr.width(ray_current)) + 1.0
    energy_group_idx = dr.zeros(UInt, num_neutron) # from energy group idx 0 to n, the energy goes from high to low
    active = True
    # travel_trough_material_boundary = False

    bounceIdx = 0
    # 
    # class IterProp:
    # def __init__(self, radiance, energy_group_idx, ray_current, active)
    itinfo = IterProp(radiance, energy_group_idx, ray_current, active)

    while bounceIdx < MAX_BOUNCE:
        all_its, material_spaces = dr.detach(scene_material_intersect(sceneinfo.scene, itinfo.ray_current, sceneinfo.scm, itinfo.active))
        dr.eval()

        ats = sample_and_compute_attenuation_along_ray(sceneinfo, all_its, material_spaces, 
                                                                  itinfo, beams, bounceIdx, save_beam)
        terminate_ray = ats.mask_invalid | ats.exit_ray

        if bounceIdx == 0:
            incremental = (ats.attenuation * radiance & itinfo.active)
        else:
            fp = dr.detach(hg(dr.dot(ray_current.d, wi_theta), AVERAGE_COS))
            incremental = ((ats.attenuation * radiance * fp / dr.detach(fp) & itinfo.active))

        Etot_incremental = dr.zeros(FloatD,  number_of_energy_group)
        dr.scatter_add(Etot_incremental, incremental, itinfo.energy_group_idx)
        Etot += Etot_incremental

        if reparam:
            cross_section_reparam_t = ats.feature.ext * ats.reparam_factor
            pdf = ats.pdf * ats.reparam_factor
            
        else:
            cross_section_reparam_t = ats.feature.ext

        itinfo.active &= (~terminate_ray)
        
        wo = sample_direction_hg(sceneinfo.rng, itinfo.ray_current.d, AVERAGE_COS)
        wo = wo / dr.norm(wo)

        wi_theta = -itinfo.ray_current.d
        itinfo.energy_group_idx, group_pdf = sample_next_energy_group(sceneinfo.rng, ats.feature.phase_cdfs, sceneinfo.scm.energy_groups)
        
        itinfo.radiance *= (ats.attenuation_till_scatter) * (cross_section_reparam_t * ats.feature.alb) / dr.detach(pdf)

        itinfo.ray_current.o = ats.scatter_pos
        itinfo.ray_current.d = wo

        bounceIdx += 1
    return Etot / num_neutron