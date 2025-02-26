
# used only on my windows machine
import sys
sys.path = ["."] + sys.path[2:]

# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64, TensorXf, Array2f, ArrayXf, Bool
from drjit.cuda.ad import Float as FloatD, UInt32 as UInt32D
import numpy as np
import random
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

class SceneMaterial:
    def __init__(self, csgnodes, cross_tots, cross_as, num_geo):
        self.csg_node_list = csgnodes
        self.cross_tot_list = cross_tots
        self.cross_a_list = cross_as
        self.node_state_lists = [] 
        self.node_shape_orders = []
        self.num_material = len(csgnodes)
        self.num_geo = num_geo
        

        for node in self.csg_node_list:
            state_list, shape_order = node.state_list()
            self.node_state_lists.append(state_list)
            self.node_shape_orders.append(shape_order)

    def get_state_by_id(self, node_id):
        return self.node_state_lists[node_id], self.node_shape_orders[node_id]

    def __repr__(self):
        return f"there are {len(self.csg_node_list)} materials in the scene"
    


def get_shape_id(scene, shape_ptr):
    shape_id = dr.zeros(UInt32, dr.width(shape_ptr))
    i = 0
    for s in scene.shapes():
        shape_id = dr.select(dr.eq(s, shape_ptr), i, shape_id)
        i += 1
    return shape_id


def geo_intersect(scene, ray):
    its_list = []
    iter_ray = mi.Ray3f(ray)
    active = True
    trace_active = True
    
    while trace_active:
        its = scene.ray_intersect(iter_ray)
        shape_id = get_shape_id(scene, its.shape)
        active = active & (~dr.isinf(its.t))
        trace_active = dr.any(active)

        if trace_active:
            its_list.append([its, active, shape_id])
            iter_ray = mi.Ray3f(its.spawn_ray(iter_ray.d))

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
    ret = np.zeros(query_state.shape[0], dtype=np.int32)
    val = np.zeros(scm.num_geo, dtype=np.int32)

    valid_state_list, shape_ids = scm.get_state_by_id(node_id)
    shape_used = np.zeros(scm.num_geo, dtype=np.int32)
    shape_used[shape_ids] = 1
    shape_mask_out = np.where(shape_used == 0)
    # print(query_state)
    # print(shape_mask_out)
    # exit(0)
    query_state[shape_mask_out] = 0
    for state in valid_state_list:
        valid_state = val + state
        valid_state[shape_mask_out] = 0
        
        matching_idx = np.where((valid_state == query_state).all(axis=1))[0]
        ret[matching_idx] = 1
    return ret

def material_node_ids(scm, space_state, ray_num):
    """
    Return: the material ids the current space belongs to
    parameters:
        scm: SceneMaterial informations
        space_state: state that we want to identify the material state
        ray_num: numbers of ray in parallel
    """
    material_ids = dr.zeros(UInt32, ray_num) + scm.num_geo
    query_state_binary = space_state % 2
    for node_id in range(len(scm.csg_node_list)):
        in_node = inside_by_node_id(query_state_binary, scm, node_id)
        # print(in_node)
        in_node_mask = Bool(in_node == 1)
        material_ids = dr.select(in_node_mask, node_id, material_ids)
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
    init_material_id = dr.zeros(UInt64, ray_num) + scm.num_geo
    geo_state_list = []
    idx = 0

    init_state = np.zeros([ray_num, scm.num_geo], dtype=np.int32)
    for intersect in its:
        it = intersect[0]
        # ignore the tagent intersection
        active_mask = intersect[1] & (dr.abs(dr.dot(ray_dir, it.sh_frame.n)) > 0.0)
        shape_id = intersect[2]

        pidx = shape_id.numpy()
        active_pidx = pidx[np.where(active_mask.numpy() == True)]
        active_idx = np.where(active_mask.numpy() == True)
        trace_space = np.zeros([ray_num, scm.num_geo], dtype=np.int32)
        trace_space[active_idx, active_pidx] = 1
        init_state[active_idx, active_pidx] += 1

        geo_state_list.append(trace_space)

    # summarize init state, where 1 stands for insides, and 0 stands for outsides
    # init_state = init_state % 2
    init_material_id = material_node_ids(scm, init_state, ray_num)

    
    material_spaces = [init_material_id]
    cur_state = init_state.copy()

    # loop through the geo_state_list and get information about material space along the ray
    for state in geo_state_list:
        cur_state = next_state(cur_state, state)
        cur_material_id = material_node_ids(scm, cur_state, ray_num)
        material_spaces.append(cur_material_id)
    
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


def scene_material_intersect(scene, rays, scm):
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
    its = geo_intersect(scene, rays)
    num_rays = dr.width(rays)
    material_spaces = get_material_space_along_ray(its, scm, num_rays, rays.d)

    # remove the invalid geometry ray interesction
    num_intersections = len(its)
    assert num_intersections == (len(material_spaces) - 1), "length of intersection and material space doesn't match"

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

    for it, cur_space, next_space in zip(itersect_list, material_spaces[:-1], material_spaces[1:]):
        cross_material_boundary = ~dr.eq(UInt32D(cur_space), UInt32D(next_space))
        valid_ith_hit = dr.eq(UInt32D(hit), ith_intersect)
        its_result = dr.select(valid_ith_hit & cross_material_boundary, it[0], its_result)
        hit = dr.select(cross_material_boundary, hit+1, hit)
    return its_result


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
        next_it = ith_hit_from_current(it_lists, material_spaces, ray_num, i)
        is_valid = np.where(next_it.is_valid().numpy() == True)
        # print(is_valid)
        if is_valid[0].shape[0] == 0:
            break
        point = next_it.p.numpy()[is_valid, :]
        # print(point.shape)
        ax.scatter(point[0, :, 0], point[0, :, 1], point[0, :, 2], marker="o", color=colors[i])
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

    media = SceneMaterial([nodeA, nodeB], [1.5, 1.0], [0.1, 0.1], 2)
    
    num_ray = 100
    v = dr.zeros(mi.Vector3f, num_ray)
    v.x = -1.0
    o = dr.zeros(mi.Vector3f, num_ray)
    o.x = 5.0
    o.z = dr.linspace(Float, -1.0, 1.0, num_ray)
    rays = mi.Ray3f(o, v)

    its, material_spaces = scene_material_intersect(scene, rays, media)
    # print("its", its)
    # print("material ids", material_spaces)
    # first_hit = ith_hit_from_current(its, material_spaces, num_ray, 0)
    # print(first_hit.t)
    visualize_intersect(its, material_spaces, num_ray)
    

test1()
