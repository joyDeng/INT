from simulator import *
from csg import save_grid_data
from constant import DATA_DIR

def load_scene_and_sensor(params):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/sphere_radius_1.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                },
            }
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([3.0, 3.0, 3.0]),
            'filename': f"{DATA_DIR}/scene/sphere_radius_1.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.23, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    # print(scene)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    m1 = CSGNode("intersection", shape0, shape1)
    m2 = CSGNode("difference", shape0, shape1)

    scm = SceneMaterial([m1, m2], [[FloatD(0.9), FloatD(0.9)]], [[0.8, 0.8]], 2, 3)
    scm.set_material_sigma(TensorXfD(params["sig_t"]))
    scm.set_material_ald(TensorXfD(params["albedo"]))
    phase_function = TensorXf([
        params["phase"]
    ])
    
    scm.set_phase_function(phase_function)

    sensor_dict = {
        'type': 'scene',
        # 'plane': {
        # 'type': 'rectangle',
        # 'to_world': mi.ScalarTransform4f.translate([0.0, 0.0, -0.2]) @ mi.ScalarTransform4f.scale([0.5, 0.5, 1.0]),
        # 'bsdf': {
        #     'type': 'diffuse',
        #     'reflectance': {
        #         'type': 'rgb',
        #         'value': [0.8, 0.8, 0.8]
        #         }
        #     },
        # "emitter": {
        #         "type": "area",
        #         "radiance": {"type": "rgb", "value": [30.0, 30.0, 30.0]},
        #     },
        # }
         "mesh_emitter": {
            "type": "obj",
            "filename": f"{DATA_DIR}/scene/sphere_radius_1.obj",

            # Optional transforms (uncomment/tune as needed)
            "to_world":  mi.ScalarTransform4f.translate([1.0, 1.0, 1.0]) @ mi.ScalarTransform4f.scale([0.05, 0.05, 0.05]),
            

            # You can still attach a BSDF (useful for importance / consistency)
            "bsdf": {"type": "diffuse", "reflectance": 0.0},

            # Turn the mesh into an area emitter
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [30.0, 30.0, 30.0]},
            },
        },
    }
    
    sensor = mi.load_dict(sensor_dict)
    return scene, scm, sensor

def load_test_scene_heat():
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.5, 1.5, 1.5]),
            'filename': f"{DATA_DIR}/scene/sphere_sPVmooth.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                },
            }
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([6.0, 6.0, 6.0]),
            'filename': f"{DATA_DIR}/scene/sphere_smooth.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.23, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    # print(scene)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    m1 = CSGNode("intersection", shape0, shape1)
    m2 = CSGNode("difference", shape0, shape1)
    # scm = SceneMaterial([m1, m2], cross_tots, cross_as, 2, 1)

    scm = SceneMaterial([m1, m2], [[0.5, 0.3]], [[0.95, 0.95]], 2, 1)
    scm.set_material_sigma(TensorXf([[0.5, 0.3]]))
    scm.set_material_ald(TensorXf([[0.95, 0.95]]))
    phase_function = TensorXf([
        [[1.0, 1.0]],
    ])
    
    scm.set_phase_function(phase_function)
    
    return scene, scm

def load_exp1_torus_openmc(sig_t=0.1, albedo=0.9):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/torus_radius_1_section_radius_onethird.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                },
            }
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/torus_radius_1_section_radius_half.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.23, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    # print(scene)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    m1 = CSGNode("intersection", shape0, shape1)
    m2 = CSGNode("difference", shape0, shape1)
    # scm = SceneMaterial([m1, m2], cross_tots, cross_as, 2, 1)

    scm = SceneMaterial([m1, m2], [[FloatD(sig_t), FloatD(0.9)]], [[0.8, 0.8]], 2, 1)
    scm.set_material_sigma(TensorXfD([[FloatD(sig_t), FloatD(0.9)]]))
    scm.set_material_ald(TensorXfD([[FloatD(albedo), FloatD(0.9)]]))
    phase_function = TensorXf([
        [[1.0, 1.0]],
    ])
    
    scm.set_phase_function(phase_function)
    
    return scene, scm

def load_exp1_sphere_openmc(sig_t=0.1, albedo=0.8):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/sphere_radius_1.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                },
            }
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([3.0, 3.0, 3.0]),
            'filename': f"{DATA_DIR}/scene/sphere_radius_1.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.23, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    # print(scene)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    m1 = CSGNode("intersection", shape0, shape1)
    m2 = CSGNode("difference", shape0, shape1)
    # scm = SceneMaterial([m1, m2], cross_tots, cross_as, 2, 1)

    scm = SceneMaterial([m1, m2], [[FloatD(sig_t), FloatD(0.9)]], [[0.8, 0.8]], 2, 1)
    scm.set_material_sigma(TensorXfD([[FloatD(sig_t), FloatD(0.9)]]))
    scm.set_material_ald(TensorXfD([[FloatD(albedo), FloatD(0.9)]]))
    phase_function = TensorXf([
        [[1.0, 1.0]],
    ])
    
    scm.set_phase_function(phase_function)
    
    return scene, scm

def simulate_neutron_in_csg_shape(scene, scm, seed, Va, AD=False):
    # print("random seed", seed)
    rng = mi.PCG32(size=NUMBER_NEUTRONS, initstate=seed, initseq=seed*2)
    # set up tally that exit the shape
    
    # value = rng.next_float32()
    # print(value)
    # generate rays
    # ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
    # ray_vec, ray_origin = sample_direction_from_linear_source(NUMBER_NEUTRONS)
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, NUMBER_NEUTRONS), dr.zeros(mi.Point3f, NUMBER_NEUTRONS)
    ray_origin.x = 5.0
    ray_vec.x = -1.0
    
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    
    # load scene
    # temp example, a scene with torus
    # print("scene loading")
    # scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)
 
    # get scene parameter for optimization
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])

    Fa = dr.unravel(mi.Vector3i, params['A.faces'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Fb = dr.unravel(mi.Vector3i, params['B.faces'])
    # Va.y = Va.y + height

    
    params['A.vertex_positions'] = dr.ravel(Va)
    params['B.vertex_positions'] = dr.ravel(Vb)
    if AD:
        dr.enable_grad(params['A.vertex_positions'])
        dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    
    vertices_list = [Va, Vb]
    faces_list = [Fa, Fb]

    
    Etot = render_nuetron_in_csg_shape(scene, rng, scm, vertices_list, faces_list, ray_current, AD) 
    return Etot

# TODO validate the gradient computation 
def test_fd(scene, Va, scm, k, height):
    N = 20
    g = FloatD(0.0)
    grad_list = []
    delta = 0.001
    
    for i in range(N):
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y += (height + delta)
        v1 = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, False)
        #print(v1.numpy())
        
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y += (height - delta)
        v2 = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, False)
        #print(v2.numpy())
        #exit(0)

        # print(Va)
        gradient = (v1 - v2) / (2.0 * delta)
        dr.eval(gradient)
        del v2, v1

        g += gradient / N
        grad_list.append(gradient)
        del gradient

        gradients_array = np.array(grad_list)
        np.save( f"delta-gradient_fd_{k}_{height:.3f}.npy", gradients_array)
        print("finite difference gradient: ith ", i, g * N / (i+1))

    # print("finite difference gradient avg across", N, g)
    # print(h)
    
    dvdh = FloatD(0.0)
    for i in range(N):
        update_height = FloatD(height)
        dr.enable_grad(update_height)
        va_param = dr.zeros(mi.Point3f, dr.width(Va)) + Va
        va_param.y +=  update_height
        v_r = simulate_neutron_in_csg_shape(scene, scm, i + k * N, va_param, True)
        # dvdh_b = dr.grad(update_height)
        # print("before", dvdh_b.numpy())
        dr.backward(v_r)
        dvdh_g = dr.grad(update_height)
        # dvdh_g = dr.select(dr.isnan(dvdh_g), 0.0, dvdh_g)
        grad_list.append(dvdh_g.numpy())
        dvdh += dvdh_g / N
        # print("reparam, ", v_r, "no reparam", v_p)
        del v_r
        gradients_array = np.array(grad_list)
        np.save( f"delta-gradient_ad_{k}_{height:.3f}.npy", gradients_array)
        print("auto dif grandients avg across", i, dvdh * N / (i+1))
    print("auto dif grandients avg across", N, dvdh)
    
# dr.set_flag(dr.JitFlag.ReuseIndices, False)

def test_torus():
    scene, scm = load_scene_node([1.5], [0.9])
    params = mi.traverse(scene)

    # has offsets
    # aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['A.vertex_normals'])
    # aUV = dr.unravel(mi.Vector2f, params['A.vertex_texcoords'])
    # tensorxf = TensorXfD(params["offset.data"])
    # heights_map = mi.Texture2f(tensorxf, wrap_mode=dr.WrapMode.Repeat)
    # offsets = heights_map.eval_cubic(aUV)[0]
    # Va = aV + aN * offsets
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    init_height = -0.08

    for h in range(32):
        for i in range(3):
            print(f"h {h}, i {i}")
            test_fd(scene,  Va, scm, i, init_height + h * 0.0045)



def delta_emission(scene, scm, num_neutrons, seed, Va, AD):
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    # set up tally that exit the shape
  
    # generate rays
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    ray_origin.x = -2.0
    ray_origin.y = 1.0
    ray_vec.x = 1.0
    
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    
    params = mi.traverse(scene)
    Fa = dr.unravel(mi.Vector3i, mi.Int(params['A.faces']))
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    # Vb.x += 2.0
    # Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Fb = dr.unravel(mi.Vector3i, mi.Int(params['B.faces']))
    # Vb.y = Vb.y + height

    
    params['A.vertex_positions'] = dr.ravel(Va)
    params['B.vertex_positions'] = dr.ravel(Vb)
    if AD:
        dr.enable_grad(params['A.vertex_positions'])
        dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    vertices_list = [Va, Vb]
    faces_list = [Fa, Fb]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, AD) 
    # print("out", Etot)
    return Etot


# def test_energy_dependent(num_netrons, number_groups):

#     scene, scm = load_test_energy_dependent()
    
#     E = render_nuetron_in_csg_shape_energy_dependent(scene, rng, scm, vertices_list, faces_list, ray_current,  number_groups, False)

import random

def test_hemisphere(num_neutrons, variable):
    scene, scm = load_test_scene_hemisphere(1.0, 1.0)
    params = mi.traverse(scene)
    delta = 0.01

    theta = variable
    # height = 1.0 - theta

    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Va_ = dr.zeros(mi.Point3f, dr.width(Va)) + Va 
    Va_.y -= ( theta + delta )
    randseed = random.randint(0, 10000)
    energy_plus = delta_emission(scene, scm, num_neutrons, randseed, Va_, False)

    Va_ = dr.zeros(mi.Point3f, dr.width(Va)) + Va 
    Va_.y -= ( theta - delta )
    energy_minus = delta_emission(scene, scm, num_neutrons, randseed, Va_, False)

    gradient = ( energy_plus -  energy_minus ) / (2.0 * delta)
    print("finite difference f(x+delta), f(x-delta) and gradient are", energy_plus.numpy(), energy_minus.numpy(), gradient.numpy())
    

    # auto diff
    theta = FloatD(variable)
    dr.enable_grad(theta)

    Va_ = dr.zeros(mi.Point3f, dr.width(Va)) + Va 
    Va_.y -= theta
    energyauto = delta_emission(scene, scm, num_neutrons, randseed, Va_, True)
    dr.backward(energyauto)
    auto_grad = dr.grad(theta)
    print("\nauto diff value is ", energyauto)
    print("auto diff gradients: ", auto_grad.numpy())

    # validation
    theta = FloatD(variable)
    dr.enable_grad(theta)

    dtheta = dr.sqrt(2.0 * 2.0 - (theta + 1.0) * (theta + 1.0))
    energy_ana = dr.exp(-1.0) - dr.exp(-2.0 * dtheta - 1.0)
    dr.backward(energy_ana)
    energy_grdient_wrt_theta = dr.grad(theta)
    print("\nanalytic value is", energy_ana)
    # energy_grdient_wrt_theta = - dr.exp( - 2.0 * dtheta - 1.0) * (2.0 * (theta + 1.0) / (dtheta))
    print("analytic gradient is", energy_grdient_wrt_theta)

    return gradient.numpy(), auto_grad.numpy(), energy_grdient_wrt_theta.numpy(), energy_ana.numpy(), energyauto.numpy()

import json

def dump_history(kernel_history, filename="history.txt"):
    print(type(kernel_history))
    with open(filename, 'w') as file:
        # file.write(str(kernel_history))
        for h in kernel_history:
            file.writelines(str(h) + "\n")

def test_hemisphere_range():
    init_value = 0.0
    fd = []
    ad = []
    vd = []
    vad = []
    vmd = []
    for i in range(30):

        with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
            f, a, v, v_ana, v_mc = test_hemisphere(400000, init_value + i * 0.03)
        hist = dr.kernel_history()
        dump_history(hist)
        exit(0)
 
        fd.append(f)
        ad.append(a)
        vd.append(v)
        vad.append(v_ana)
        vmd.append(v_mc)
        # print(init_value + i * 0.05)

    np.save("hemisphere_test_fd.npy", np.array(fd))
    np.save("hemisphere_test_ad.npy", np.array(ad))
    np.save("hemisphere_test_vd.npy", np.array(vd))
    np.save("hemisphere_test_vana.npy", np.array(vad))
    np.save("hemisphere_test_vmc.npy", np.array(vmd))


def test_2cubes(num_neutrons):
    scene, scm = load_test_scene_2cubes(1.0, 1.0)
    params = mi.traverse(scene)
    delta = 0.01

    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    print('validation: value', dr.exp(-0.5) * (1.0 - dr.exp(-1)) * (1 + dr.exp(-1)))
    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *=(2.0+delta)
    vb_.x += 2.0

    energy = delta_emission(scene, scm, num_neutrons, 0, vb_, False)
    print("mc", energy.numpy())

    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *= ( 1.5 + delta )
    vb_.x += 2.0
    
    print("validate: ", dr.exp(-0.5) - dr.exp(-3.5))
    energy1 = delta_emission(scene, scm, num_neutrons, 0, vb_, False)
    print(energy1.numpy())
    
    
    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *= ( 1.5 - delta )
    vb_.x += 2.0
    energy2 = delta_emission(scene, scm, num_neutrons, 0, vb_, False)
    print(energy2.numpy())

    gradients = (energy1 - energy2) / (2.0 * delta)
    print("finite difference gradients ", gradients.numpy())
    print("gradient validation analytic results: ", dr.exp(-3.5))
    del energy1, energy2

    offset = FloatD(1.5)
    dr.enable_grad(offset)
    vb_ = dr.zeros(mi.Point3f, dr.width(Vb)) + Vb
    # vb_.x *= offset
    vb_.x += 2.0
    energyA = delta_emission(scene, scm, num_neutrons, 0, vb_, True)
    print(energyA.numpy())

    dr.backward(energyA)
    gradientAD = dr.grad(offset)
    print("auto diff gradients", gradientAD)

    del gradientAD, energyA


def test_pdf_1d(seed, number_of_neutrons):
    r_a, r_b, r_c = 1.0, 1.0, 1.0
    cs_a, cs_b, cs_c = 0.5, 0.75, 1.0
    rng = mi.PCG32(size=number_of_neutrons, initstate=seed, initseq=seed*2)
    # sample distance
    t = sample_distance(cs_a, rng)
    stop_in_a = (t < r_a)
    t_b = ( t * cs_a - r_a * cs_a ) / cs_b
    stop_in_b = (~stop_in_a) & (t_b < r_b)
    t_c = (t * cs_a - r_a * cs_a - r_b * cs_b) / cs_c
    stop_in_c = (~stop_in_a) & (~stop_in_b) & (t_c < r_c)

    t_real = dr.select(stop_in_a, t, dr.select(stop_in_b, r_a + t_b, dr.select(stop_in_c, r_a + r_b + t_c, r_a + r_b + t_c)))
    distances = t_real.numpy()
    t_real = distances[distances > 0.0]

    x = dr.linspace(Float, 0, 3.5, 1000)
    y = dr.exp(-x * cs_a)
    value = dr.select(x < r_a, cs_a * y, dr.select(x < (r_b + r_a), cs_b * dr.exp( - ((x-r_a) * cs_b + cs_a * r_a)) , dr.select(x < (r_b + r_a + r_c), cs_c * dr.exp(-((x-r_b-r_a) * cs_c + cs_a * r_a + cs_b * r_b)), dr.exp(- cs_a * r_a - cs_b * r_b - cs_c * r_c))))
    # value = dr.select(x < r_a, cs_a * dr.exp(-x * cs_a), dr.select(x < (r_b + r_a), cs_b *  dr.exp(-x * cs_b), dr.select(x < (r_b + r_a + r_c), cs_c * dr.exp(-x * cs_b), 0.0)))

    colors = plt.cm.Dark2(np.linspace(0.0, 1.0, 8))
    plt.hist(t_real, bins=2000, density=True, color = colors[0], alpha=0.5, label="sample histogram")
    plt.plot(x, value, color = colors[1], label="pdf")
    plt.ylim(0.0, 0.6)
    plt.xlim(0.0, 3.0)
    plt.legend()
    plt.show()
    # draw a histogram here
    
def test_gradient_multi_1d(seed, number_of_neutrons):
    r_a, r_b, r_c = FloatD(1.0), FloatD(1.0), FloatD(1.0)
    dr.enable_grad(r_a), dr.enable_grad(r_b), dr.enable_grad(r_c)
    cs_a, cs_b, cs_c = 0.5, 0.75, 1.0
    rng = mi.PCG32(size=number_of_neutrons, initstate=seed, initseq=seed*2)
    t = sample_distance(cs_a, rng)

    attenuation = dr.exp(-t * cs_a)

    stop_in_a = (t < r_a)
    t_b = ( t * cs_a - r_a * cs_a ) / cs_b
    stop_in_b = (~stop_in_a) & (t_b < r_b)
    t_c = (t * cs_a - r_a * cs_a - r_b * cs_b) / cs_c
    stop_in_c = (~stop_in_a) & (~stop_in_b) & (t_c < r_c)

    value = dr.select( (~stop_in_a) & (~stop_in_b) & (~stop_in_c),  dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c) / dr.detach(dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)) , 0.0)
    sum = dr.sum(value) / number_of_neutrons
    dr.backward(sum)
    dEdra = dr.grad(r_a)
    dEdrb = dr.grad(r_b)
    dEdrc = dr.grad(r_c)
    print("                       dEda,                     dEdb,                       dEdc")
    print("Auto diff, monte carlo", dEdra, dEdrb, dEdrc)
    AdEdra = -cs_a * dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)
    AdEdrb = -cs_b * dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)
    AdEdrc = -cs_c * dr.exp(-cs_a * r_a - cs_b * r_b - cs_c * r_c)
    print("Analytic              ", AdEdra, AdEdrb, AdEdrc)
    

def test_track_length(num_neutrons, theta, file_id, es="tr"):
    Offset = FloatD(theta)
    dr.enable_grad(Offset)
    scene, scm = load_scene_node_track_length_test()
    rng = mi.PCG32(size=num_neutrons, initstate=1994)
    # set up tally that exit the shape
  
    # generate rays
    # ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    # ray_origin.x = -2.0
    # ray_vec.x = 1.0
    # ray_origin.z = (rng.next_float32() - 1.0 ) * 2.0
    ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)
    
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    
    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    aN = dr.unravel(mi.Vector3f, params['A.vertex_normals'])

    params['B.vertex_positions'] = dr.ravel(aV + aN * Offset)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()

    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    
    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]
    beam_list = []



    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, True, beam_list) 

    
    for b in beam_list:    
        b.save_beams("{:02}_vertices.npy".format(file_id), 'ab')
    resolution = mi.Vector3i(75, 75, 75)
    boundingbox = [mi.Vector3f(-3.0, -3.0, -3.0), mi.Vector3f(3.0, 3.0, 3.0)]

    spatial_distribution = accumulate_photon_beams_faster(beam_list, resolution, boundingbox)
    
    voxels_array = spatial_distribution.numpy() / num_neutrons
    save_grid_data(voxels_array, resolution, boundingbox, "{:02}_voxel_array.npy".format(file_id), 'wb')



def load_exp1_sphere_openmc_multi_group(params):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/sphere_radius_1.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                },
            }
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([3.0, 3.0, 3.0]),
            'filename': f"{DATA_DIR}/scene/sphere_radius_1.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.23, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    # print(scene)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    m1 = CSGNode("intersection", shape0, shape1)
    m2 = CSGNode("difference", shape0, shape1)

    scm = SceneMaterial([m1, m2], [[FloatD(0.9), FloatD(0.9)]], [[0.8, 0.8]], 2, 3)
    scm.set_material_sigma(TensorXfD(params["sig_t"]))
    scm.set_material_ald(TensorXfD(params["albedo"]))
    phase_function = TensorXf([
        params["phase"]
    ])
    
    scm.set_phase_function(phase_function)

    scm.print_materials()
    
    return scene, scm


def load_exp1_torus_openmc_multi_group(params):
    scene_dict = {
        'type': 'scene',
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/torus_radius_1_section_radius_onethird.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.25, 0.7]
                },
            }
        },
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': f"{DATA_DIR}/scene/torus_radius_1_section_radius_half.obj",
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                    'type': 'rgb',
                    'value': [0.23, 0.25, 0.7]
                },
            }
        },
    }
    scene = mi.load_dict(scene_dict)
    # print(scene)
    shape0 = CSGLeaf(0)
    shape1 = CSGLeaf(1)

    m1 = CSGNode("intersection", shape0, shape1)
    m2 = CSGNode("difference", shape0, shape1)

    scm = SceneMaterial([m1, m2], [[FloatD(0.9), FloatD(0.9)]], [[0.8, 0.8]], 2, 3)
    scm.set_material_sigma(TensorXfD(params["sig_t"]))
    scm.set_material_ald(TensorXfD(params["albedo"]))
    phase_function = TensorXf([
        params["phase"]
    ])
    
    scm.set_phase_function(phase_function)

    scm.print_materials()
    
    return scene, scm

def two_sphere_multi_energy_sensor(num_neutrons, param_dict, seed):
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    sig_t_shell = [FloatD(0.9), FloatD(0.9), FloatD(0.9)]
    albedo_shell = [0.8 / 0.9, 0.8 / 0.9, 0.8 / 0.9]

    # sig_s[g_out, g_in]
    Phase_shell = np.array([
        [0.75, 0.125, 0.125],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    sig_t_inner = [param_dict["sig_t"], FloatD(0.8), FloatD(0.8)]
    albedo_inner = [0.7, 0.7, 0.7]
    Phase_inner =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    params = {}
    params["sig_t"] = TensorXfD([sig_t_inner, sig_t_shell])
    params["albedo"] = [albedo_inner, albedo_shell]
    params["phase"] = [Phase_inner, Phase_shell]

    scene, scm, sensor = load_scene_and_sensor(params)


    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    radius_scale = param_dict["geo"]

    # point light
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    random_cos = sample_float_32(rng) * 2.0 - 1.0
    random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    phi = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec.x += random_sin * dr.cos(phi)
    ray_vec.y += random_sin * dr.sin(phi)
    ray_vec.z += random_cos

    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])

    aV.y = aV.y * radius_scale
    aV.x = aV.x * radius_scale
    aV.z = aV.z * radius_scale
    params['A.vertex_positions'] = dr.ravel(aV)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))

    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list, sensor)
    Etot = render_neutron_in_csg_shape_energy_dependent_with_sensor(sceneinfo, ray_current, True)

    # print("returns", Etot.shape, " \nvalues: ", Etot)
    return Etot

def two_torus_multi_energy(num_neutrons, param_dict, seed):
    # Row out_going, Column in_going
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    sig_t_shell = [FloatD(0.9), FloatD(0.9), FloatD(0.9)]
    albedo_shell = [0.8 / 0.9, 0.8 / 0.9, 0.8 / 0.9]

    # sig_s[g_out, g_in]
    Phase_shell = np.array([
        [0.75, 0.125, 0.125],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    sig_t_inner = [param_dict["sig_t"], FloatD(0.8), FloatD(0.8)]
    albedo_inner = [0.7, 0.7, 0.7]
    Phase_inner =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    params = {}
    params["sig_t"] = TensorXfD([sig_t_inner, sig_t_shell])
    params["albedo"] = [albedo_inner, albedo_shell]
    params["phase"] = [Phase_inner, Phase_shell]

    scene, scm = load_exp1_torus_openmc_multi_group(params)


    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    offset_y = param_dict["geo"]

    # point light
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    random_cos = sample_float_32(rng) * 2.0 - 1.0
    random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    phi = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec.x += random_sin * dr.cos(phi)
    ray_vec.y += random_sin * dr.sin(phi)
    ray_vec.z += random_cos

    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['.vertex_normals'])

    aV.y = aV.y + offset_y
    params['A.vertex_positions'] = dr.ravel(aV)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))

    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, True)

    # print("returns", Etot.shape, " \nvalues: ", Etot)
    return Etot

def two_sphere_multi_energy(num_neutrons, param_dict, seed):
    # Row out_going, Column in_going
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    sig_t_shell = [FloatD(0.9), FloatD(0.9), FloatD(0.9)]
    albedo_shell = [0.8 / 0.9, 0.8 / 0.9, 0.8 / 0.9]

    # sig_s[g_out, g_in]
    Phase_shell = np.array([
        [0.75, 0.125, 0.125],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    # set up in openmc which is inverse from my assumption, in openmc lower idx indicate lower energy group
    # here lower idx indicate higher energy group
    # sig_t_inner = [0.9, 0.8, 0.8]
    # albedo_inner = [0.9, 0.99, 0.7]
    # Phase_inner =  np.array([
    #     [1.00, 0.50, 0.10],
    #     [0.00, 0.50, 0.50],
    #     [0.00, 0.00, 0.40],
    # ])

    sig_t_inner = [param_dict["sig_t"], FloatD(0.8), FloatD(0.8)]
    albedo_inner = [0.7, 0.7, 0.7]
    Phase_inner =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])
    # Phase_inner =  np.array([
    #     [0.40, 0.50, 0.10],
    #     [0.00, 0.50, 0.50],
    #     [0.00, 0.00, 1.00],
    # ])


    params = {}
    params["sig_t"] = TensorXfD([sig_t_inner, sig_t_shell])
    params["albedo"] = [albedo_inner, albedo_shell]
    params["phase"] = [Phase_inner, Phase_shell]

    scene, scm = load_exp1_sphere_openmc_multi_group(params)


    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    radius_scale = param_dict["geo"]

    # point light
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    random_cos = sample_float_32(rng) * 2.0 - 1.0
    random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    phi = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec.x += random_sin * dr.cos(phi)
    ray_vec.y += random_sin * dr.sin(phi)
    ray_vec.z += random_cos

    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['.vertex_normals'])

    aV.y = aV.y * radius_scale
    aV.x = aV.x * radius_scale
    aV.z = aV.z * radius_scale
    params['A.vertex_positions'] = dr.ravel(aV)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))

    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, True)

    # print("returns", Etot.shape, " \nvalues: ", Etot)
    return Etot



def two_sphere_radius(num_neutrons, param_dict, seed):
    """
    save value data and gradient data, compare with openmc results
    """
    scene, scm = load_exp1_sphere_openmc(param_dict["sig_t"], param_dict["albedo"])
    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    radius_scale = param_dict["geo"]

    # point light
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    random_cos = sample_float_32(rng) * 2.0 - 1.0
    random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    phi = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec.x += random_sin * dr.cos(phi)
    ray_vec.y += random_sin * dr.sin(phi)
    ray_vec.z += random_cos

    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['.vertex_normals'])

    aV.y = aV.y * radius_scale
    aV.x = aV.x * radius_scale
    aV.z = aV.z * radius_scale
    params['A.vertex_positions'] = dr.ravel(aV)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))

    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, True)
    return Etot



def two_torus_offset(num_neutrons, param_dict, seed):
    """
    save value data and gradient data, compare with openmc results
    """
    scene, scm = load_exp1_torus_openmc(param_dict["sig_t"], param_dict["albedo"])
    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    offset = param_dict["geo"]
    

    # ring light
    # ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    # x_phi = sample_float_32(rng) * 2.0 * dr.pi
    # ray_origin.x += dr.sin(x_phi)
    # ray_origin.z += dr.cos(x_phi)

    # random_cos = sample_float_32(rng) * 2.0 - 1.0
    # random_sin = dr.sqrt(1.0 - random_cos * random_cos)
    # phi = sample_float_32(rng) * 2.0 * dr.pi
    # ray_vec.x += random_sin * dr.cos(phi)
    # ray_vec.y += random_sin * dr.sin(phi)
    # ray_vec.z += random_cos
    ray_vec, ray_origin = sample_dir_from_unit_ring(rng, 1.0, cos_theta=0.0)

    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)
    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['.vertex_normals'])

    aV.y = aV.y + offset
    # aV.x = aV.x * re_scale
    # aV.z = aV.z * re_scale
    params['A.vertex_positions'] = dr.ravel(aV)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))

    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, True)
    return Etot

function_dict = {
    "sphere": two_sphere_radius,
    "torus": two_torus_offset,
    "sphere_multi": two_sphere_multi_energy,
    "torus_multi": two_torus_multi_energy
}


def save_data(name, gradients, grad_error, values, val_eror):
    np.save(f"{name}_ad_gradients.npy", gradients)
    np.save(f"{name}_ad_values.npy", values)
    np.save(f"{name}_ad_values_errors.npy", val_eror)
    np.save(f"{name}_ad_gradients_errors.npy", grad_error)

def test_parameter_gradient(num_neutrons, param_range, steps, func_name, param_name, param_set, N, id=1):
    step_size = (param_range[1] - param_range[0]) / steps
    r = param_range[0]
    values = []
    gradients = []
    values_error = []
    gradients_error = []

    for i in range(steps+1):
        grads = []
        vals = []
        
        for j in range(N):
            param_ = FloatD(r)
            dr.enable_grad(param_)
            param_set[param_name] = param_
            print(" parameter: ", param_set)
            Etot = function_dict[func_name](num_neutrons, param_set, i * N + 1994 + j)
            dr.backward(Etot)
            dEdr = dr.grad(param_)
            grads.append(dEdr.numpy())
            vals.append(Etot.numpy())
            print("dEdr", dEdr)
            del dEdr, Etot
        gradients.append(np.mean(np.array(grads)))
        values.append(np.mean(np.array(vals)))
        values_error.append(np.std(np.array(vals)))
        gradients_error.append(np.std(np.array(grads)))
        r += step_size
    grads = np.array(gradients)
    vals = np.array(values)
    print("gradients: ", grads)
    print("values: ", vals)
    save_data(f"{func_name}_{param_name}_{id}", grads, np.array(gradients_error), vals, np.array(values_error))

def test_parameter_gradient_multi(num_neutrons, param_range, steps, func_name, param_name, param_set, N, id=1):
    step_size = (param_range[1] - param_range[0]) / steps
    r = param_range[0]
    values = []
    gradients = []
    values_error = []
    gradients_error = []

    for i in range(steps+1):
        grads = []
        vals = []
        
        for j in range(N):
            param_ = FloatD(r)
            dr.enable_grad(param_)
            param_set[param_name] = param_
            print(" parameter: ", param_set)
            Etot = function_dict[func_name+"_multi"](num_neutrons, param_set, i * N + 1994 + j)
            dr.forward(param_)
            # print(Etot.shape, Etot)
            dEdr = dr.grad(Etot)
            print("gradient shape", dEdr.shape, "gradient value", dEdr)
            grads.append(dEdr.numpy())
            vals.append(Etot.numpy())
            # print("dEdr", dEdr)
            del dEdr, Etot
        gradients.append(np.mean(np.array(grads), axis=0))
        values.append(np.mean(np.array(vals), axis=0))
        values_error.append(np.std(np.array(vals), axis=0))
        gradients_error.append(np.std(np.array(grads), axis=0))
        r += step_size
    grads = np.array(gradients)
    vals = np.array(values)
    print("gradients: ", grads)
    print("values: ", vals)
    save_data(f"{func_name}_{param_name}_multi_{id}", grads, np.array(gradients_error), vals, np.array(values_error))

# def test_sig_t_gradient(num_neutrons, sig_t_range, steps):
#     step_size = (sig_t_range[1] - sig_t_range[0]) / steps
#     sig_t = sig_t_range[0]
#     values = []
#     gradients = []
#     for i in range(steps+1):
#         print(" sig_t: ", sig_t)
#         SIG_T = FloatD(sig_t)
#         dr.enable_grad(SIG_T)
#         Etot = two_sphere_radius(num_neutrons, 1.0, i + 1994, SIG_T)
#         dr.backward(Etot)
#         dEdr = dr.grad(SIG_T)
#         gradients.append(dEdr.numpy())
#         values.append(Etot.numpy())
#         del dEdr, Etot
#         sig_t += step_size
#     grads = np.array(gradients)
#     vals = np.array(values)
#     print("gradients: ", grads)
#     print("values: ", vals)
#     save_data("sig_t", grads, vals)

    


def two_sphere_get_spatial_distribution(num_neutrons, Offset, seed, resolution, boundingbox, saveVerticeFile=False, file_id=0, es="tr", bid=-1):
    """
    Save the Gradients of the radiance field with respect to the parameter to file_id

    The scene is consists of two spheres, the parameter we are caring about is the vertical offset of the
    sphere on the right.
    """
    scene, scm = load_test_scene_heat()
    dr.make_opaque(seed)
    rng = mi.PCG32(size=num_neutrons, initstate=seed)
    dr.make_opaque(rng)
    # set up tally that exit the shape
  
    # generate rays
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    ray_origin.x -= 3.0
    # ray_origin.y = 0.0
    ray_origin.y = sample_float_32(rng) * 2.0 - 1.0
    # random_cos = sample_float_32(rng) * 2.0 - 1.0
    # ray_vec.x += dr.sqrt(1.0 - random_cos * random_cos)
    # ray_vec.y += random_cos
    ray_vec.x += 1.0
    # ray_vec.y -= 0.25
    ray_vec /= dr.norm(ray_vec)
    # print(ray_vec)
    # exit(0)
    # ray_vec /= dr.norm(ray_vec)
    
    ray_current = mi.Ray3f(ray_origin, ray_vec)
    dr.make_opaque(ray_current)

    params = mi.traverse(scene)
    aV = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    # aN = dr.unravel(mi.Vector3f, params['.vertex_normals'])

    # aV.y = aV.y + Offset
    params['A.vertex_positions'] = dr.ravel(aV)
    
    
    dr.enable_grad(params['A.vertex_positions'])
    dr.enable_grad(params['B.vertex_positions'])
    params.update()
    
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    
    Fa = dr.unravel(mi.Vector3i, mi.Int(params[f'A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params[f'B.faces']))

    vertices_list = [Vb, Va]
    faces_list = [Fb, Fa]
    beam_list = []

    sceneinfo = SceneInfo(scene, rng, scm, vertices_list, faces_list)
    Etot = render_nuetron_in_csg_shape_energy_dependent(sceneinfo, ray_current, True, beam_list, True) 
    
    if es == "tr":
        # i = 0
        # for b in beam_list:   
        #     b.save_before_compress("{}_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))
        #     i += 1
            # b.save_beams("{}_two_sphere_collision_density_vertices.npy".format(file_id), 'ab')
        spatial_distribution = accumulate_photon_beams_faster(beam_list, resolution, boundingbox, bid)
    else:
        spatial_distribution = accumulate_photon_point(beam_list, resolution, boundingbox, bid)

    voxels_array = spatial_distribution

    if saveVerticeFile:
        save_grid_data(voxels_array.numpy(), resolution, boundingbox, "{}_two_sphere_collision_density.npy".format(file_id), "wb")
        
    return voxels_array

def two_sphere_get_spatial_gradient(nuetron_number, seed, file_id, epho, bid=0):
    delta = 0.001
    resolution = mi.Vector3i(50, 50, 50)
    boundingbox = [mi.Vector3f(-3.5, -3.5, -3.5), mi.Vector3f(3.5, 3.5, 3.5)]

    # finite difference
    N = epho
    for i in range(N):
        print("finite diff iteration", i)
        print("+: ")
        spatial_plus = two_sphere_get_spatial_distribution(nuetron_number, delta, seed+i * 10, resolution, boundingbox, True, str(file_id)+"+", 'tr', bid)
        print("-: ")
        spatial_min = two_sphere_get_spatial_distribution(nuetron_number, -delta, seed+i * 10, resolution, boundingbox, True, str(file_id)+"-", 'tr', bid)
        if i == 0:
            gradients_fd = (spatial_plus.numpy() - spatial_min.numpy()) / (2 * delta)
        else:
            gradients_fd += (spatial_plus.numpy() - spatial_min.numpy()) / (2 * delta)
    
    save_grid_data(gradients_fd / N, resolution, boundingbox, "{:02d}_two_sphere_collision_density_gradients_fd.npy".format(file_id), 'wb')

    
    for i in range(N):
        print("auto diff iterations", i)
        # dr.set_flag(dr.JitFlag.Debug, False)
        Offset = FloatD(0.0)
        dr.enable_grad(Offset)
        spatial_distribution = two_sphere_get_spatial_distribution(nuetron_number, Offset, seed + i * 10, resolution, boundingbox, True, file_id, 'tr', bid)
        dr.forward(Offset)
        if i == 0:
            gradients_ad = dr.grad(spatial_distribution).numpy()
            dr.eval(gradients_ad)
        else:
            gradients_ad += dr.grad(spatial_distribution).numpy()
            dr.eval(gradients_ad)
    save_grid_data(gradients_ad / N, resolution, boundingbox, "{:02d}_two_sphere_collision_density_gradients_ad.npy".format(file_id), 'wb')

# @dr.syntax
def test_beams_hat(num_neutrons, reso):
    rng = mi.PCG32(size=num_neutrons, initstate=1)
    ray_vec, ray_origin = dr.zeros(mi.Vector3f, num_neutrons), dr.zeros(mi.Point3f, num_neutrons)
    ray_origin.x -= 2.5
    # ray_origin.y = 0.0
    # ray_origin.y = sample_float_32(rng) * 2.0 - 1.0
    # random_cos = sample_float_32(rng) * 2.0 - 1.0
    # ray_vec.x = dr.sqrt(1.0 - random_cos * random_cos)
    # ray_vec.y = random_cos
    ray_vec.x += 1.0
    ray_vec.y += 2.0
    ray_vec /= dr.norm(ray_vec)
    raylength = dr.full(FloatD, 4.0)

    b = Beams(ray_origin, ray_origin + raylength * ray_vec, ray_vec, raylength, True)
    beam_list = []
    beam_list.append(b)
     
    resolution = mi.Vector3i(reso, reso, reso)
    boundingbox = [mi.Vector3f(-3.5, -3.5, -3.5), mi.Vector3f(3.5, 3.5, 3.5)]

    sum_array = accumulate_photon_beam_hat(beam_list, resolution, boundingbox, 0)
    # print(type(sum_array))
    save_grid_data(sum_array.numpy(), resolution+1, boundingbox, "test_beam_hat.npy", 'wb')
    # return sum_array
    
def uncertainty_f(geo_offset, ad=True):
    func_name = "torus"
    num_netrons = 100000
    param = FloatD(geo_offset)
    # print(geo_offset)
    if ad:
        print("gradient enabled")
        dr.enable_grad(param)

    param_set = {
        "geo": param,
        "sig_t": 0.1,
        "albedo": 0.9
    }
    Etot = function_dict[func_name](num_netrons, param_set, 1994)

    if ad:
        dr.backward(Etot)
        dEdr =  dr.grad(param)
        val = Etot.numpy()
        grad = dEdr.numpy()
        print("Etot, dEdr", Etot, dEdr)
        del Etot, dEdr
    else:
        val = Etot.numpy()
        grad = 0
        del Etot
    return val, grad

def valid_sensor(N, hight):
    x, y = (np.random.rand(N) - 0.5), (np.random.rand(N) - 0.5)
    pdf = 1
    distance_sqr = (np.power(x, 2.0) + np.power(y, 2.0) + hight * hight)
    distance = np.sqrt(distance_sqr)
    tr = np.exp(-0.9 * distance)
    cos_theta = hight / distance 
    
    energy = np.sum(INV_FOUR_PI * tr * cos_theta / distance_sqr / pdf) / N
    print("energy collected: ", energy)


if __name__ == "__main__":
    # valid_sensor(5000000, 0.2)
    # exit(0)
    # test_gradient_multi_1d(0, 200000)
    # test_2cubes(400000)
    # test_hemisphere_range()
    # seed = 1991
    # v = np.zeros(1)
    # for i in range(12000):
    # two_sphere_get_spatial_gradient(100000, seed, file_id=2, epho=1, bid=1)
    # print(v)
    # test_track_length(20000, 0.1, "18")
    # test_kernel(1000, 4)


    # test beam hat
    # test_beams_hat(1, 120)
    num_neutrons = 2000000
    param_range = [0.1, 2.0]
    torus_scale_range = [-0.195, 0.195]
    sphere_scale_range = [0.1, 2.0]
    albedo_range = [0.8, 0.995]
    N = 10

    # Etot = []
    # for i in range(N):
    #     e = two_sphere_multi_energy_sensor(num_neutrons, {"geo": 1.0, "sig_t": FloatD(0.9)}, 1994 + i)
    #     Etot.append(e.numpy())
    #     del e
    # Etot = np.mean(np.array(Etot), axis=0)
    # print("Etot: ", Etot)
    
    # exit(0)
    steps = 25
    
    # test_radius_gradient(num_neutrons, radius_range, steps)

    # test sig_t
    # test_sig_t_gradient(num_neutrons, [0.1, 2.0], steps)
    # num_neutrons, param_range, steps, func_name, param_name

    # param_set = {
    #     "geo": 0.1,
    #     "sig_t": 0.1,
    #     "albedo": 0.9
    # }
    # sig_t_range = [0.1, 2.0]
    test_parameter_gradient_multi(num_neutrons, torus_scale_range, steps, "torus", "geo", {"geo": 0.0, "sig_t": FloatD(0.9)}, N, id=2)
    
    # test load the parameters for multi energy groups
    # two_sphere_multi_energy(num_neutrons, {"geo": 0.5}, 1994)