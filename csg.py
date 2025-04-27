
# used only on my windows machine
# import sys
# sys.path = ["."] + sys.path[2:]

# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64, TensorXf, ArrayXu, ArrayXf, Bool, UInt, Array2u, Int
from drjit.cuda.ad import Float as FloatD, UInt32 as UInt32D
import numpy as np
import torch
import random
from collections.abc import Collection
# dr.JitFlag.PacketOps=True

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR

class CSGLeaf:
    def __init__(self, geoid):
        self.id = geoid

    def __repr__(self):
        return repr(self.id)
    
    def is_leaf(self):
        return True

class CSGNode:
    def __init__(self, operation, left, right):
        self.op = operation
        self.left = left
        self.right = right
        

    def is_leaf(self):
        return False

    def __repr__(self):
       return repr(self.op) + " of " + repr(self.left) + " and " + repr(self.right)

    def state_list(self):
        left_order = []
        right_order = []
        state_lists = []
        left_state_lists = []
        right_state_lists = []

        if self.left.is_leaf():
            left_state_lists.append([True])
            left_order.append(self.left.id)
        else:
            left_state_lists, left_order = self.left.state_list()
            

        if self.right.is_leaf():
            right_state_lists.append([True])
            right_order.append(self.right.id)
        else:
            right_state_lists, right_order = self.right.state_list()


        if self.op == "union":
            for left_st in left_state_lists:
                for right_st in right_state_lists:
                    # print(not right_st)
                    state_lists.append(left_st+right_st)
                    state_lists.append(left_st+[not right_st])
                    state_lists.append([not left_st]+right_st)
                    # print(state_lists)
        elif self.op == "intersection":
            for left_st in left_state_lists:
                for right_st in right_state_lists:
                    state_lists.append(left_st+right_st)
        elif self.op == "difference":
            for left_st in left_state_lists:
                for right_st in right_state_lists:
                    state_lists.append(left_st + [not right_st])
        
        return state_lists, left_order + right_order
    
# add multiple energy tally
class MultiGroupTally:
    def __init__(self, number_of_energy_group):
        self.radiation_tally = dr.zeros(FloatD, number_of_energy_group)
        self.maxEnergy = Float(40000000.0)
        self.minEnergy = Float(1.0 / 40.0)
        self.energy_width = (self.maxEnergy - self.minEnergy) / number_of_energy_group

    def find_energy_index(self, values):
        slot_indx = dr.floor((values - self.minEnergy) / self.energy_width)
        return slot_indx
    
    def add_to_tally(self, values):
        indices = self.find_energy_index(values)
        dr.scatter_add(self.radiation_tally, values, indices)
    

def test_find_energy_index(number_neutron):
    rng = mi.PCG32(size=number_neutron)
    m_groups_tally = MultiGroupTally(100)
    energy_value = rng.next_float32() * (m_groups_tally.maxEnergy - m_groups_tally.minEnergy) + m_groups_tally.minEnergy
    m_groups_tally.add_to_tally(energy_value)


class SceneMaterial:
    def __init__(self, csgnodes, cross_tots, albedos, num_geo, energy_groups=1):
        """
        cross_tots: list of cross sections, if multi_group is true, this is a list of tensor (with the length that equals to the number of groups)
        albedos: list of albedo, if multi_group is true, this is a list of tensor (with the length of group)
        
        By default there is one energy group

        csg_node_list: list of material nodes
        cross_tot_list: a two dimensional tensor with row idx being material idx and col idx being energy group idx
        alb_list: a two dimensional tensor with row idx being material idx and col idx being energy group idx
        node_state_lists: list of states that is lawful for being inside the medium, first dim idx is the idx of the material
        node_shape_orders: idxs of the shape id in the mitsuba
        phase_function_cdf: a three dimensional tensor with row idx beinig material idx and dim=1 idx being incident energy group idx, dim=2 being outgoinig energy group idx

        """
        self.csg_node_list = csgnodes
        self.cross_tot_list = cross_tots
        self.alb_tot_list = albedos
        self.node_state_lists = [] 
        self.node_shape_orders = []
        self.num_material = len(csgnodes)
        # self.num_material_ad = UInt32D(len(csgnodes))
        self.num_geo = num_geo
        # which geometry are gradient enabled
        self.energy_groups = energy_groups
        # self.energy_groups_ad = UInt32D(energy_groups)

        self.init_multi_group_properties(energy_groups, len(csgnodes))

        # replace with drjit
        for node in self.csg_node_list:
            state_list, shape_order = node.state_list()
            # we have problem here when size of state list is larger than 1
            # print(state_list, shape_order)
            shape_id = UInt(shape_order)
            state_array = np.array(state_list).astype(np.int32)
            
            state_num = state_array.shape[0]
            geo_num_compress = state_array.shape[1]
            states = TensorXf(state_array)
            geo_idx = dr.zeros(TensorXf, (state_num, geo_num_compress))
            geo_idx += shape_id
  
            states_extend = dr.zeros(TensorXf, (state_num, self.num_geo))
            xidx, yidx = dr.meshgrid(dr.arange(UInt, state_num), dr.arange(UInt, state_array.shape[1]))
            # print("states array", states)
            # print(xidx)
            # print(yidx)
            # print("geo_idx", geo_idx)
            if state_num > 1:
                ridx = yidx
            else:
                ridx = xidx
            
            dr.scatter(states_extend.array, states.array, ridx * self.num_geo + geo_idx.array)
            # print("state extend", states_extend)
            # exit(0)
            # state_full = dr.zeros(UInt, self.num_geo)
            # print(states_extend)
            # exit(0)
            # dr.scatter(state_full, state_list, shape_order)
            # print("geo_idx", geo_idx.array)
            # print("xidx", xidx)
            # print("yidx", yidx)
            # print("idx", xidx * self.num_geo + geo_idx.array)
            # print("state_list", states_extend)
            # print("extend state list", states_extend)
            self.node_state_lists.append(states_extend)
            self.node_shape_orders.append(UInt(shape_order))
        # exit(0)

    def init_multi_group_properties(self, number_of_energy_group, num_of_material):
        # store the properies as pytorch tensors
        # init sigmas and albedo
        # cross sections are two dimensional tensor with dim 0: material_id, dim 1: energy group id
        # should revise this to load from some dataset
        if self.energy_groups == 1:
            self.cross_tot_list = TensorXf(torch.tensor(self.cross_tot_list, device='cuda:0', dtype=torch.float32).reshape(-1, 1))
            self.alb_tot_list = TensorXf(torch.tensor(self.alb_tot_list, device='cuda:0', dtype=torch.float32).reshape(-1, 1))
        else:
            self.cross_tot_list = TensorXf(torch.zeros(self.num_material, self.energy_groups, device='cuda:0', dtype=torch.float32) + 1.0)
            self.alb_tot_list = TensorXf(torch.zeros(self.num_material, self.energy_groups, device='cuda:0', dtype=torch.float32) + 1.0)

        # init phase function
        # phase function is a three dimensional tensor with dim 0: material_id, dim 1: incident energy group id, dim2 = exiting energy group id
        # should revise this to load from some dataset / 
        pdf = torch.zeros([num_of_material, number_of_energy_group, number_of_energy_group], device='cuda:0', dtype=torch.float32) + 1.0 / number_of_energy_group
        self.pdf = TensorXf(pdf) 
        self.cdfs = dr.cumsum(self.pdf, axis=-1)

    def set_material_sigma(self, value):
        """set material tot cross section, this should be a tensor of dim num_material x num_energy_level"""
        self.cross_tot_list = value
        dr.make_opaque(self.cross_tot_list)

    def set_material_ald(self, value):
        """set material albedo, this should be a tensor of dim num_material x num_energy_level"""
        self.alb_tot_list = value
        dr.make_opaque(self.alb_tot_list)
    
    def set_phase_function(self, value):
        """set material phase function, this should be a tensor of dim num_material x num_energy_level x num_energy_level"""
        self.pdf = value
        self.cdfs = dr.cumsum(self.pdf, axis=-1)
        dr.make_opaque(self.cdfs)

    def get_sig(self, material_idx, energy_idx):
        return dr.gather(Float, self.cross_tot_list.array, material_idx * self.energy_groups + energy_idx)

    def get_optical_properties(self, material_idx, energy_idx):
        sig = dr.gather(Float, self.cross_tot_list.array, material_idx * self.energy_groups + energy_idx)
        alb = dr.gather(Float, self.alb_tot_list.array, material_idx * self.energy_groups + energy_idx)

        xid, yid  =  dr.meshgrid(dr.arange(UInt, self.energy_groups), dr.arange(UInt, dr.width(material_idx)))
        # print(type(material_idx), type(energy_idx))
        material_idx_scattered = dr.gather(UInt, material_idx, yid)
        energy_idx_scattered = dr.gather(UInt, energy_idx, yid)

        query_idx = material_idx_scattered * self.energy_groups * self.energy_groups + self.energy_groups * energy_idx_scattered + xid
        phase_cdf = dr.gather(Float, self.cdfs.array, query_idx)
        phase_cdf = dr.reshape(TensorXf, phase_cdf, (dr.width(material_idx), self.energy_groups))

        dr.make_opaque(sig, alb, phase_cdf)
        return sig, alb, phase_cdf

    def get_state_by_id(self, node_id):
        return self.node_state_lists[node_id], self.node_shape_orders[node_id]

    def __repr__(self):
        return f"there are {len(self.csg_node_list)} materials in the scene"
    
def test_multi_energy_sigma_t(num_group):
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    nodeA = CSGNode("intersection", shape0, shape1)
    nodeB = CSGNode("difference", shape0, shape1)

    media = SceneMaterial([nodeA, nodeB], [1.5, 1.0], [0.1, 0.1], 2)

    media.init_multi_group_properties(num_group)
    sigmat, alb, phase = media.get_optical_properties(torch.tensor([1, 1], device="cuda:0", dtype=torch.int64), torch.tensor([0, 0], device="cuda:0", dtype=torch.int64))
    # print(TensorXf(phase))
 
# test_multi_energy_sigma_t(3)
# exit(0)

def test_TensorXf(num_material, num_group, num_ray=2):

    v = TensorXf([0, 1])
    print(v)

    # init_state_dr = dr.zeros(TensorXf, shape=[3, 4])
    # index_x = dr.arange(UInt, 2)
    # index_y = dr.arange(UInt, 2)
    # dr.scatter(init_state_dr.array, 2.0, index_x * 4 + index_y)
    # print(init_state_dr)

    exit(0)
    sig = torch.rand([num_material, num_group, num_group], dtype=torch.float32, device="cuda:0")
    sig_dr = TensorXf(sig)

    # print(" torch tensor", sig)
    # print(" dr tensor", sig_dr)

    material_idx = torch.tensor([0, 1], device="cuda:0", dtype=torch.int64)
    energy_idx = torch.tensor([2, 3], device="cuda:0", dtype=torch.int64)

    material_idx_dr = UInt(material_idx)
    energy_idx_dr = UInt(energy_idx)

    sig_query = sig[material_idx, energy_idx]
    #print(dr_query)

    #help(sig_dr.array)
    
    xid, yid  =  dr.meshgrid(dr.arange(UInt, num_group), dr.arange(UInt, num_ray))
    # print(xid, yid)

    # print("material idx dr", material_idx_dr)
    material_idx_scattered = dr.gather(UInt, material_idx_dr, yid)
    energy_idx_scattered = dr.gather(UInt, energy_idx_dr, yid)
    # print(material_idx_scattered)
    # print(energy_idx_scattered)

    query_idx = material_idx_scattered * num_group * num_group + num_group * energy_idx_scattered + xid
    v = dr.gather(Float, sig_dr.array, query_idx)
    v = dr.reshape(TensorXf, v, (num_ray, num_group))
    #(material_idx_dr * num_group + energy_idx_dr)
    # v = sig_dr[material_idx_dr, energy_idx_dr]
    #print(qz)
    #help(dr.slice_index)
    #print(sig_dr[dr_query])S
    #exit(0)
    # sig_query_dr_material = sig_dr[]
    # how to specify query dim?
    # help(sig_dr)
    # print("query from dr tensor", v)
    # sig_query_dr = sig_query_dr_material[energy_idx]

    

    print("query from torch tensor", sig_query)


# exit(0)

class MaterialParameter:
    def __init__(self, cross_tot_t, alb, cdf=[]):
        self.ext = cross_tot_t
        self.alb = alb
        self.phase_cdfs = cdf

def get_shape_id(scene, shape_ptr):
    shape_id = dr.zeros(UInt, dr.width(shape_ptr))
    i = dr.opaque(UInt, 0)
    for s in scene.shapes():
        shape_id = dr.select(s == shape_ptr, i, shape_id)
        i += 1
    return shape_id


def geo_intersect(scene, ray, active_ray):
    its_list = []
    iter_ray = mi.Ray3f(ray)
    active = True & active_ray
    trace_active = True

    # i = 0
    continue_trace = True
    while continue_trace: # check whether there is a infinite while loop
        its = scene.ray_intersect(iter_ray)
        dr.eval(its)
        shape_id = get_shape_id(scene, its.shape)
        active = dr.select(its.is_valid() & active, True, False)
        # fact = dr.select(active, 1, 0)
        # trace_active = dr.sum(fact)
        continue_trace = dr.any(active)
        if continue_trace:
            its_list.append([its, active, shape_id])
            iter_ray = mi.Ray3f(its.spawn_ray(iter_ray.d))
        # else:
        #     continue_trace = False
        # i += 1
    return its_list

def count_active_intersect(it_list):
    count = 0
    for it in it_list:
        count = dr.select(~dr.isinf(it[0].t), count + 1, count)
    return count

def inside(insides, state):
    result = np.zeros(state.shape[0], dtype=np.int32)
    value = np.zeros(state.shape[1], dtype=np.int32)
    process_state = state % 2
    for cur_state in insides:
        temp = value + cur_state
        matching_idx = np.where((temp == process_state).all(axis=1))[0]
        result[matching_idx] = 1
    return result

def inside_by_node_id(query_state, scm, node_id):
    """
    Return: True(1) or False(0) the cur_state is in side the querying node
    Parameters: 
        query_state: The space state used to query
        scm: csg node informations
        node_id: we want to know whether the query state is in node_id
    """
    # TODO change torch tensor to dr
    # ret = torch.zeros(query_state.shape[0], dtype=torch.int64, device='cuda:0')
    # val = torch.zeros(scm.num_geo, dtype=torch.int64, device='cuda:0')
    ray_num = query_state.shape[0]
    ray_idx = dr.arange(UInt, ray_num)

    ret = dr.zeros(UInt, ray_num)
    val = dr.zeros(TensorXf, scm.num_geo)

    valid_state_list, shape_ids = scm.get_state_by_id(node_id)

    # shape_used = torch.zeros(scm.num_geo, dtype=torch.int64, device='cuda:0')
    # shape_used[shape_ids] = 1
    
    shape_used = dr.zeros(UInt, scm.num_geo)
    dr.scatter(shape_used, 1, shape_ids)
    shape_mask_out = dr.compress(shape_used == 0)
    for sid in range(dr.width(shape_mask_out)):
        query_state[:, shape_mask_out[sid]] = 0
    # dr.scatter(query_state, 0, ray_idx * scm.num_geo + shape_mask_out)
    # query_state[ray_idx * scm.num_geo + shape_mask_out] = 0
    # print(query_state)
    # exit(0)
    for state_idx in range(valid_state_list.shape[0]):
        # print(state, val)
        valid_state = val + TensorXf(valid_state_list[state_idx])
        # print("valid state", valid_state, state_idx)
        if shape_mask_out.shape[0] > 0:
            dr.scatter(valid_state.array, 0, shape_mask_out)
        # for sid in range(dr.width(shape_mask_out)):
        #     valid_state[shape_mask_out[sid]] = 0

        # print((valid_state == query_state))
        
        v = (valid_state == query_state)
        value = dr.select(v, 1.0, 0.0)
        
        t = dr.reduce(dr.ReduceOp.Add, value, axis=1)
        matching_idx = dr.compress((t == scm.num_geo).array)
   
        if matching_idx.shape[0] > 0:
            dr.scatter(ret, 1, matching_idx)
        # ret[matching_idx] = 1
        # print("matching idx", matching_idx)
    return ret

def material_node_ids(scm, space_state, ray_num):
    """
    Return: the material ids the current space belongs to
    parameters:
        scm: SceneMaterial informations
        space_state: state that we want to identify the material state
        ray_num: numbers of ray in parallel
    """
    material_ids = dr.zeros(UInt, ray_num) + scm.num_material
    # print(space_state)
    query_state_binary = 2.0 * (space_state / 2 - dr.floor(space_state / 2))
    # print("binary state", query_state_binary)
    # print("\n")
    # query_state_binary = space_state % 2
    for node_id in range(len(scm.csg_node_list)):
        in_node_mask = inside_by_node_id(query_state_binary, scm, node_id)
        material_ids = dr.select(in_node_mask == 1, node_id, material_ids)
    return material_ids

def get_material_space_along_ray(its, scm, ray_num, ray_dir):
    """ 
    # Return the list of material_spaces traveled along the ray (list length should equals to number of intersection + 1)
    # Parameters:
    #   its: list of intersction information, [mi.intersection, active_mask, shape_id]
    #   scm: SceneMaterial nodes info
    #   ray_num: number of ray in parallel
    #   ray_dir: direction of the ray is traveling TODO: this parameter might not be neccessary
    """
    geo_state_list = []
    # idx = 0
    
    # init_state_torch = torch.zeros([ray_num, scm.num_geo], dtype=torch.int64, device="cuda:0")
    
    init_state_dr = dr.zeros(TensorXf, shape=[ray_num, scm.num_geo])


    for intersect in its:
        it = intersect[0]
        # ignore the tagent intersection
        no_parallel = dr.abs(dr.dot(ray_dir, it.sh_frame.n)) > 0.0
        active_mask = (intersect[1] & no_parallel) 
        shape_id = intersect[2]

        # torch implementation
        # pidx_torch = shape_id.torch().long()
        # value = dr.select(active_mask, 1.0, 0.0)
        # active_idx_torch = torch.where(value.torch() == 1.0)[0].long()
        # active_pidx_torch = torch.gather(pidx_torch, 0, active_idx_torch)

        # dr implementation
        active_idx = dr.compress(active_mask)  
        active_pidx = dr.gather(UInt, UInt(shape_id), UInt(active_idx))
  
        # torch implementation
        # trace_space_torch = torch.zeros([ray_num, scm.num_geo], dtype=torch.int64, device="cuda:0")
        # trace_space_torch[active_idx_torch, active_pidx_torch] = 1
        # init_state_torch[active_idx_torch, active_pidx_torch] += 1
        # geo_state_list.append(trace_space_torch)
        

        trace_space = dr.zeros(TensorXf, shape=[ray_num, scm.num_geo])
        dr.scatter(trace_space.array, 1.0, active_idx * scm.num_geo + active_pidx)
        init_state_dr += trace_space
        geo_state_list.append(trace_space)
        # drtrace_space.array

        # idx += 1


        
    # summarize init state, where 1 stands for insides, and 0 stands for outsides
    # init_material_id = material_node_ids(scm, init_state_torch, ray_num)
    init_material_id = material_node_ids(scm, init_state_dr, ray_num)
    # print(type(init_material_id), init_material_id)
    # exit(0)

    material_spaces = [init_material_id]
    cur_state = dr.zeros(TensorXf, [ray_num, scm.num_geo]) + init_state_dr

    # loop through the geo_state_list and get information about material space along the ray
    for state in geo_state_list:
        # print("state_dr", init_state_dr)
        # print("cur_state", cur_state)
        
        cur_state = next_state(cur_state, state)
        # print("cur_state", cur_state, "init_state", init_state_dr)
        # exit(0)
        cur_material_id = material_node_ids(scm, cur_state, ray_num)
        # print(cur_material_id)
        material_spaces.append(cur_material_id)
    
    # exit(0)
    # print(material_spaces)
    return material_spaces


def get_init_state(its, shape_order, ray_num, geo_num, ray_dir):
    geo_states_list = []
    active_its = dr.zeros(UInt64, ray_num)
    idx = 0

    for itersect in its:
        it = itersect[0]
        cur_active = itersect[1]
        shape_id = itersect[2]
        active_its = dr.select(cur_active, active_its+1, active_its)

        idx += 1
        exit_geo = ((dr.dot(it.sh_frame.n, ray_dir) > 0) & (cur_active)).numpy()
        in_geo = ((dr.dot(it.sh_frame.n, ray_dir) < 0) & (cur_active)).numpy()
        pidx = shape_id.numpy()
    
        exit_rays = np.where(exit_geo == True)
        in_rays = np.where(in_geo == True)

        exit_pidx = pidx[np.where(exit_geo == True)]
        in_pidx = pidx[np.where(in_geo == True)]

        # 0 in geometry, 1 out of geometry, -1 invalid intersection
        trace_out_state = np.zeros([ray_num, geo_num], dtype=np.int32)
        trace_out_state[exit_rays, exit_pidx] = 1
        trace_out_state[in_rays, in_pidx] = 1
        geo_states_list.append(trace_out_state)


    # summarize init state
    init_state = np.zeros([ray_num, geo_num], dtype=np.int32)
    for state in geo_states_list:
        init_state += state
    return init_state, geo_states_list
        


        

def next_state(init_state, inter_state):
    init_state += inter_state
    return init_state


def scene_material_intersect(scene, rays, scm, active):
    """ 
      This function returns all intersections of ray with scene materials 
    # with list of inshape and fromshape indices to discribe from which material the ray is
    # traveling and to which material the ray is arriving through the intersection
    # the index wich is equal to the number of node in scm refers to vaccum
    #
    # Parameters: 
    #       sceme: mitsuba scene
    #       rays: mitsuba rays
    #       scm:  ScemeMaterial 
    # Return:
    #       it_lists: list of interesctions
    #       cur_material_space: list of index the ray travels from before intersect
    # 
    """     
    
    # get all intersction of ray with the scene geometries  
    its = dr.detach(geo_intersect(scene, rays, active))
    num_rays = dr.width(rays)
    # v = 1
    # print(type(num_rays), type(v))
    material_spaces = get_material_space_along_ray(its, scm, num_rays, rays.d)
    # print(material_spaces)
    # exit(0)
    # remove the invalid geometry ray interesction
    # num_intersections = len(its)
    # assert num_intersections == (len(material_spaces) - 1), "length of intersection and material space doesn't match"
    return its, material_spaces


def csg_intersect(scene, rays, csnode):
    shape_state, shape_order = csnode.state_list()
    # print(csnode.op, shape_state)
    its = geo_intersect(scene, rays)
    num_rays = dr.width(rays)
    num_geometry = len(shape_order)

    # determine initial space state of the ray.o
    init_state, geo_state_list = get_init_state(its, shape_order, num_rays, num_geometry, rays.d)
    cur_space = inside(shape_state, init_state)

    its_result = dr.zeros(mi.SurfaceInteraction3f, dr.width(rays))
    change_state = init_state
    num_valid_intersection = dr.zeros(FloatD, dr.width(rays))
    inout = dr.zeros(UInt32, dr.width(rays))
    shape_id = dr.zeros(UInt32, dr.width(rays))

    for it, intersection_state in zip(its, geo_state_list):
        change_state = next_state(change_state, intersection_state)
        next_space = inside(shape_state, change_state)
        next_space_state = UInt32(next_space.tolist())
        cur_space_state = UInt32(cur_space.tolist())
        valid_mask = ~dr.eq(next_space_state, cur_space_state)
        num_valid_intersection = dr.select(valid_mask, num_valid_intersection + 1.0, num_valid_intersection)
        its_result = dr.select(dr.eq(num_valid_intersection, 1.0) & valid_mask, it[0], its_result)
        cur_space = next_space.copy()

        inout = dr.select(dr.eq(num_valid_intersection, 1.0) & valid_mask, next_space_state, inout)
        shape_id = dr.select(dr.eq(num_valid_intersection, 1.0) & valid_mask, it[2], shape_id)
    del its, geo_state_list
    return its_result, inout, shape_id


def ith_hit_from_current(itersect_list, material_spaces, num_rays, current_id):
    """
    Return: The ith hit from the ray_origin
    Parameter: 
        intersect_list: list of intersection
        material_spaces: list of materials along ray
        num_rays: number of rays
        current_id: itersection id for query
    """
    assert current_id <= len(itersect_list), f"no more hit than {len(itersect_list)}"
    ith_intersect = dr.full(UInt32D, current_id, num_rays)
    its_result = dr.zeros(mi.SurfaceInteraction3f, num_rays)
    hit = dr.zeros(UInt32, num_rays)
    shape_id = dr.zeros(UInt32, num_rays)

    for it, cur_space, next_space in zip(itersect_list, material_spaces[:-1], material_spaces[1:]):
        cross_material_boundary = ~(UInt32D(cur_space) == UInt32D(next_space))
        valid_ith_hit = (UInt32D(hit) == ith_intersect)
        its_result = dr.select(valid_ith_hit & cross_material_boundary, it[0], its_result)
        shape_id = dr.select(valid_ith_hit & cross_material_boundary, it[2], shape_id)
        hit = dr.select(cross_material_boundary, hit+1, hit)
    return its_result, shape_id


# visualization test
def torus(precision, c, a):
    u = np.linspace(0, 2*np.pi, precision)
    v = np.linspace(0, 2*np.pi, precision)
    u, v = np.meshgrid(u, v)
    x = (c+a*np.cos(v))*np.cos(u)
    z = (c+a*np.cos(v))*np.sin(u)
    y = a*np.sin(v)
    return x, y, z

import matplotlib.pyplot as plt
from matplotlib import cm

def visualize_intersect(it_lists, material_spaces, ray_num):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # ax.set_box_aspect([2.5, 2, 2])
    ax.set_box_aspect([3, 1, 3])

    viridis = cm.get_cmap('viridis', 256)
    colors = viridis(np.linspace(0, 1, len(it_lists)))
    # print("number of max intersection", len(it_lists))

    i = 0
    while True:
        next_it, sid= ith_hit_from_current(it_lists, material_spaces, ray_num, i)
        # print(next_it[0])
        is_valid = np.where(next_it.is_valid().numpy() == True)
        # print(is_valid)
        
        # print(is_valid)
        if is_valid[0].shape[0] == 0:
            break
        # print(next_it.p.numpy().shape)
        
        point = next_it.p.numpy()[:, is_valid]
        # print(point.shape)
        # exit(0)
        ax.scatter(point[0, 0, :], point[1, 0, :], point[2, 0, :], marker="o", color=colors[i])
        i += 1
    
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

def test1():
    # intersection tests
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
            # 'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/torusA.obj",
            'bsdf': {'type': 'diffuse'}
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            # 'to_world': mi.ScalarTransform4f().translate([-0.5, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/torusB.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }
    scene = mi.load_dict(scene_dict)
    A = scene.shapes()[0]
    B = scene.shapes()[1]
    
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    nodeA = CSGNode("intersection", shape0, shape1)
    nodeB = CSGNode("difference", shape0, shape1)

    # two material here

    media = SceneMaterial([nodeA, nodeB], [1.5, 1.0], [0.1, 0.1], 2)
    
    num_ray = 100
    v = dr.zeros(mi.Vector3f, num_ray)
    v.x = -1.0
    o = dr.zeros(mi.Vector3f, num_ray)
    o.x = 5.0
    o.z = 0.0
    # >dr.linspace(Float, -1.0, 1.0, num_ray)
    rays = mi.Ray3f(o, v)

    its, material_spaces = scene_material_intersect(scene, rays, media, True)
    print(material_spaces)
    # print("its", its)
    # print("material ids", material_spaces)
    # first_hit = ith_hit_from_current(its, material_spaces, num_ray, 0)
    # print(first_hit.t)
    # visualize_intersect(its, material_spaces, num_ray)
    
if __name__ == "__main__":
    test1()
    # test_TensorXf(3, 4)
