from area_tally import *

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

    
    Etot = render_nuetron_in_csg_shape_energy_dependent(scene, rng, scm, vertices_list, faces_list, ray_current, AD) 
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
    

def test_track_length(num_neutrons, theta):
    Offset = FloatD(theta)
    dr.enable_grad(Offset)
    scene, scm = load_scene_node_track_length_test()
    # params = mi.traverse(scene)
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
    
    vertices_list = [Va, Vb]
    faces_list = [Fa, Fb]
    beam_list = []


    Etot = render_nuetron_in_csg_shape_energy_dependent(scene, rng, scm, vertices_list, faces_list, ray_current, True, beam_list) 

    
    for b in beam_list:    
        # if b == beam_list[0]:
        b.save_beams("01_vertices.npy", 'ab')
    # print(len(beam_list))

    # spatial_distribution = accumulate_photon_beams(beam_list, mi.Vector3i(25, 25, 25), mi.Vector3f(-2.0, -2.0, -2.0), mi.Vector3f(2.0, 2.0, 2.0))

    # print(spatial_distribution)
    
    # voxels_array = spatial_distribution.numpy().reshape(25, 25, 25)
    # np.save("voxel_volume.npy", voxels_array)

    # exit(0)


if __name__ == "__main__":
    # test_gradient_multi_1d(0, 200000)
    # test_2cubes(400000)
    # test_hemisphere_range()
    test_track_length(10000, 0.1)