
# used only on my windows machine
import sys
sys.path = ["."] + sys.path[2:]

# print(sys.path)
# exit(0)

# need to change back to llvm on my mac
import mitsuba as mi
import drjit as dr
from drjit.cuda import Float, UInt32, UInt64, TensorXf, Array2f, ArrayXf
from drjit.cuda.ad import Float as FloatD
import numpy as np
import random
# dr.JitFlag.PacketOps=True

mi.set_variant('cuda_ad_rgb')
from constant import DATA_DIR

class CSGLeaf:
    def __init__(self, geoid):
        self.id = geoid

    def __repr__(self):
        return repr(self.geometry)
    
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
       return repr(self.operation) + "of" + repr(self.left) + "and" + repr(self.right)

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
        exit_geo = ((dr.dot(it.n, ray_dir) > 0) & (cur_active)).numpy()
        in_geo = ((dr.dot(it.n, ray_dir) < 0) & (cur_active)).numpy()
        pidx = shape_id.numpy()
    
        exit_rays = np.where(exit_geo == True)
        in_rays = np.where(in_geo == True)
        exit_pidx = pidx[np.where(exit_geo == True)]
        in_pidx = pidx[np.where(in_geo == True)]

        # 0 in geometry, 1 out of geometry, -1 invalid intersection
        trace_out_state = np.zeros([ray_num, geo_num], dtype=np.int32) - 1
        trace_out_state[exit_rays, exit_pidx] = 1
        trace_out_state[in_rays, in_pidx] = 0
        geo_states_list.append(trace_out_state)

    # summarize init state
    init_state = np.zeros([ray_num, geo_num], dtype=np.int32) - 1
    for state in geo_states_list:
        mask_0 = (state == 1) & ((init_state == -1) | (init_state == 0))
        mask_1 = (state == 0) & ((init_state == -1) | (init_state == 1))
        init_state[mask_0] = 1
        init_state[mask_1] = 0
    
    return init_state, geo_states_list
        

def inside(insides, state):
    result = np.zeros(state.shape[0], dtype=np.int32)
    value = np.zeros(state.shape[1], dtype=np.int32)
    for cur_state in insides:
        temp = value + cur_state
        matching_idx = np.where((temp == state).all(axis=1))[0]
        result[matching_idx] = 1

    return result
        

def next_state(init_state, inter_state):
    mask_0 = ((init_state == 1) & (inter_state == 1))
    mask_1 = ((init_state == 0) & (inter_state == 0))
    init_state[mask_0] = 0
    init_state[mask_1] = 1
    return init_state


def csg_intersect(scene, rays, csnode):
    shape_state, shape_order = csnode.state_list()
    its = geo_intersect(scene, rays)
    num_rays = dr.width(rays)
    num_geometry = len(shape_order)

    # determine initial space state of the ray.o
    print("get init state")
    init_state, geo_state_list = get_init_state(its, shape_order, num_rays, num_geometry, rays.d)
    cur_space = inside(shape_state, init_state)

    its_result = dr.zeros(mi.SurfaceInteraction3f, dr.width(rays))
    change_state = init_state
    num_valid_intersection = dr.zeros(FloatD, dr.width(rays))
    for it, intersection_state in zip(its, geo_state_list):
        change_state = next_state(change_state, intersection_state)
        
        next_space = inside(shape_state, change_state)
        valid_mask = ~dr.eq(UInt32(next_space.tolist()), UInt32(cur_space.tolist()))
        num_valid_intersection = dr.select(valid_mask, num_valid_intersection + 1.0, num_valid_intersection)
        its_result = dr.select(dr.eq(num_valid_intersection, 1.0), it[0], its_result)
    return its_result

# def intersect_csg_node(scene, rays, node):
    # csg_intersect(scene, rays, node)

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
            'to_world': mi.ScalarTransform4f().translate([0.0, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/init.obj",
            'bsdf': {'type': 'diffuse'}
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().translate([-0.5, 0.0, 0.0]),
            'filename': "E:/Research/NeutronInv/INT/scene/init.obj",
            'bsdf': {'type': 'diffuse'}
        },
    }
    scene = mi.load_dict(scene_dict)
    A = scene.shapes()[0]
    B = scene.shapes()[1]
    
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    node = CSGNode("union", shape0, shape1)

    num_ray = 5
    v = dr.zeros(mi.Vector3f, num_ray)
    v.x = -1.0
    o = dr.zeros(mi.Vector3f, num_ray)
    o.x = 0.0
    o.y = dr.linspace(Float, -1.0, 1.0, num_ray)
    rays = mi.Ray3f(o, v)

    it = csg_intersect(scene, rays, node)
    print(it.p, it.prim_index)

# test1()