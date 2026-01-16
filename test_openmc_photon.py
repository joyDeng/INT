import numpy as np
import openmc
import openmc.mgxs
import math

# -----------------------
# Problem parameters
# -----------------------

BATCHES = 50

def make_1group_mgxs_two_materials(
    mats,  # list of tuples: (xs_name, SIGMA_T, SIGMA_A, SIGMA_S)
    filename="mgxs.h5"
):
    groups = openmc.mgxs.EnergyGroups(group_edges=[0.0, 20.0e6])  # eV
    mg_lib = openmc.MGXSLibrary(groups)

    for xs_name, SIGMA_T, SIGMA_A, SIGMA_S in mats:
        xsdata = openmc.XSdata(xs_name, groups)
        xsdata.order = 0
        xsdata.set_total(np.array([SIGMA_T], dtype=float))
        xsdata.set_absorption(np.array([SIGMA_A], dtype=float))

        scatter_matrix = np.zeros((1, 1, 1), dtype=float)
        scatter_matrix[0, 0, 0] = SIGMA_S
        xsdata.set_scatter_matrix(scatter_matrix)

        mg_lib.add_xsdata(xsdata)

    mg_lib.export_to_hdf5(filename)

def build_and_run_torus(param_set, SOURCE_E_EV, PARTICLES, seed_value):
    tot_cs_dense = 0.9
    make_1group_mgxs_two_materials(
        mats=[
            ("out",  tot_cs_dense,                  tot_cs_dense * (1.0 - 0.9),        tot_cs_dense * 0.9),
            ("inner", param_set["sig_t"],       param_set["sig_t"] * (1.0 - param_set["albedo"]),  param_set["sig_t"] * param_set["albedo"]),
            ("void", 0.0, 0.0, 0.0),
        ],
        filename="mgxs.h5"
    )
    openmc.config["mg_cross_sections"] = "mgxs.h5"

    # tot_cs_dense = tot_cs * 0.2
    # make_1group_mgxs(tot_cs_dense * 2.0, tot_cs_dense * (1.0 - albedo), tot_cs_dense * albedo, filename="mgxs.h5", xs_name="mat2")
    # openmc.config["mg_cross_sections"] = "mgxs.h5"  # :contentReference[oaicite:5]{index=5}

    # 2) Material using macroscopic MGXS
    macro = openmc.Macroscopic("out")
    mat_shell = openmc.Material(name="shell")
    mat_shell.set_density("macro", 1.0)     # required for macroscopic data
    mat_shell.add_macroscopic(macro)

    macro2 = openmc.Macroscopic("inner")
    mat_in = openmc.Material(name="inner")
    mat_in.set_density("macro", 1.0)     # required for macroscopic data
    mat_in.add_macroscopic(macro2)

    macro_void = openmc.Macroscopic("void")
    mat_void = openmc.Material(name="void")
    mat_void.set_density("macro", 1.0)
    mat_void.add_macroscopic(macro_void)

    materials = openmc.Materials([mat_in, mat_shell, mat_void])

    R_MAJOR_IN  = 1.0   # distance from center to tube centerline
    R_MINOR_IN  = 0.3   # tube radius (minor)
    R_MAJOR_OUT = 1.0
    R_MINOR_OUT = 0.5

    # x0, y0, z0 = translation  # keep your translation convention

    # print("r_sphere_cm minor radius", R_MINOR_IN, R_MINOR_OUT)
    # exit(0)


    outer = openmc.Sphere(r=2.0, boundary_type="vacuum") 
    # cell_void = openmc.Cell(region=(+tor_out & -outer))
    
    tor_in  = openmc.ZTorus(x0=0.0, y0=0.0, z0=param_set["geo"], a=R_MAJOR_IN,  b=R_MINOR_IN,  c=R_MINOR_IN, boundary_type="transmission")
    tor_out = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R_MAJOR_OUT, b=R_MINOR_OUT, c=R_MINOR_OUT, boundary_type="transmission")

    # Regions:
    # -tor_in  : inside inner torus
    # -tor_out : inside outer torus
    # +tor_in  : outside inner torus
    cell_in = openmc.Cell(region=-tor_in, fill=mat_in)
    cell_sh = openmc.Cell(region=(-tor_out & +tor_in), fill=mat_shell)
    cell_out = openmc.Cell(region=(+tor_out & -outer), fill=mat_void)
    
    # help(openmc.material.Material)

    geom = openmc.Geometry(openmc.Universe(cells=[cell_in, cell_sh, cell_out]))

    # cell = openmc.Cell(region=-sph_outer & +sph, fill=mat)
    # geom = openmc.Geometry(openmc.Universe(cells=[cell_in, cell]))

    # 4) Settings: fixed source, multi-group
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.energy_mode = "multi-group"  # :contentReference[oaicite:6]{index=6}
    settings.batches = BATCHES
    settings.particles = PARTICLES
    settings.seed = seed_value

    src = openmc.IndependentSource()
    # src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    # src.angle = openmc.stats.Isotropic()

    R_RING = R_MAJOR_IN  # common choice: emit along the tube centerline
    Z_RING = 0.0

    # src = openmc.IndependentSource()

    src.space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.Discrete([R_RING], [1.0]),
        phi=openmc.stats.Uniform(0.0, 2.0 * math.pi),
        z=openmc.stats.Discrete([Z_RING], [1.0]),
        origin=(0.0, 0.0, 0.0),
    )

    # Delta (monodirectional) emission in +x direction
    # src.angle = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
    src.energy = openmc.stats.Discrete([SOURCE_E_EV], [1.0])
    settings.source = src

    # 5) Tally: net current leaving the sphere surface
    # For a vacuum boundary, inward current should be ~0, so "current" ~ outward leakage probability.
    NMAX = 2
    coll_bins = list(range(NMAX + 1))

    tallies = openmc.Tallies()
    t = openmc.Tally(name="leakage_current")
    t.filters = [openmc.SurfaceFilter(outer), openmc.CollisionFilter(coll_bins), ]  # <-- 0-collision only
    t.scores = ["current"]
    tallies.append(t)

    model = openmc.Model(materials=materials, geometry=geom, settings=settings, tallies=tallies)
    model.export_to_xml()
    openmc.run()

    # 6) Postprocess: energy leaving per source particle
    sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")
    tally = sp.get_tally(name="leakage_current")

    mean = tally.mean.flatten()
    std  = tally.std_dev.flatten()

    total_current = mean.sum()
    print("mean flux: ", mean)

    return total_current




def build_and_run_sphere(param_set, SOURCE_E_EV, PARTICLES, seed_value):
    # 1) Write MGXS file and point OpenMC to it
    # inner sphere: mat2, outer shell: mat
    tot_cs_dense = 0.9
    make_1group_mgxs_two_materials(
        mats=[
            ("mat",  tot_cs_dense,                  tot_cs_dense * (1.0 - param_set["albedo"]),        tot_cs_dense * param_set["albedo"]),
            ("mat2", param_set["sig_t"],       param_set["sig_t"] * (1.0 - param_set["albedo"]),  param_set["sig_t"] * param_set["albedo"]),
        ],
        filename="mgxs.h5"
    )
    openmc.config["mg_cross_sections"] = "mgxs.h5"

    # tot_cs_dense = tot_cs * 0.2
    # make_1group_mgxs(tot_cs_dense * 2.0, tot_cs_dense * (1.0 - albedo), tot_cs_dense * albedo, filename="mgxs.h5", xs_name="mat2")
    # openmc.config["mg_cross_sections"] = "mgxs.h5"  # :contentReference[oaicite:5]{index=5}

    # 2) Material using macroscopic MGXS
    macro = openmc.Macroscopic("mat")
    mat = openmc.Material(name="shell")
    mat.set_density("macro", 1.0)     # required for macroscopic data
    mat.add_macroscopic(macro)

    macro2 = openmc.Macroscopic("mat2")
    mat2 = openmc.Material(name="inner")
    mat2.set_density("macro", 1.0)     # required for macroscopic data
    mat2.add_macroscopic(macro2)

    materials = openmc.Materials([mat2, mat])

    # 3) Geometry: sphere of radius R in vacuum
    sph = openmc.Sphere(x0=0, y0=0, z0=0, r=param_set["geo"])
    sph_outer = openmc.Sphere(r=3.0, boundary_type="vacuum")

    # cell = openmc.Cell(region=-sph, fill=mat)
    cell_in = openmc.Cell(region=-sph, fill=mat2)

    cell = openmc.Cell(region=-sph_outer & +sph, fill=mat)
    geom = openmc.Geometry(openmc.Universe(cells=[cell_in, cell]))

    # 4) Settings: fixed source, multi-group
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.energy_mode = "multi-group"  # :contentReference[oaicite:6]{index=6}
    settings.batches = BATCHES
    settings.particles = PARTICLES
    settings.seed = seed_value

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()
    # Delta (monodirectional) emission in +x direction
    # src.angle = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
    src.energy = openmc.stats.Discrete([SOURCE_E_EV], [1.0])
    settings.source = src

    # 5) Tally: net current leaving the sphere surface
    # For a vacuum boundary, inward current should be ~0, so "current" ~ outward leakage probability.
    NMAX = 2
    coll_bins = list(range(NMAX + 1))
    # print(coll_bins)
    # exit(0)

    tallies = openmc.Tallies()
    t = openmc.Tally(name="leakage_current")
    t.filters = [openmc.SurfaceFilter(sph_outer), openmc.CollisionFilter(coll_bins), ]  # <-- 0-collision only
    t.scores = ["current"]
    tallies.append(t)

    model = openmc.Model(materials=materials, geometry=geom, settings=settings, tallies=tallies)
    model.export_to_xml()
    openmc.run()

    # 6) Postprocess: energy leaving per source particle
    sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")
    tally = sp.get_tally(name="leakage_current")

    mean = tally.mean.flatten()
    std  = tally.std_dev.flatten()

    total_current = mean.sum()
    # J0, J1 = mean[0], mean[1]

    # mean = tally.mean.flatten()[0]
    # std = tally.std_dev.flatten()[0]

    print("mean flux: ", mean)

    return total_current

    # mean has units of "particles crossing surface per source particle" (net current)
    # E_out_mean_eV = mean * SOURCE_E_EV
    # E_out_std_eV  = std  * SOURCE_E_EV

    # # Also convert to Joules if you want
    # EV_TO_J = 1.602176634e-19
    # E_out_mean_J = E_out_mean_eV * EV_TO_J
    # E_out_std_J  = E_out_std_eV  * EV_TO_J

    # print("=== Leakage results (per source particle) ===")
    # print(f"Leakage current (net): {mean:.6e} ± {std:.6e}")
    # print(f"Energy out: {E_out_mean_eV:.6e} ± {E_out_std_eV:.6e} eV")
    # print(f"Energy out: {E_out_mean_J:.6e} ± {E_out_std_J:.6e} J")

def save_data(name, values, gradient, gradient_errors, xs, val_std):
    np.save(f"{name}_openmc_fd_value.npy", values)
    np.save(f"{name}_openmc_fd_value_errors.npy", val_std)
    np.save(f"{name}_openmc_fd_gradient.npy", gradient)
    np.save(f"{name}_openmc_fd_gradient_errors.npy", gradient_errors)
    np.save(f"{name}_openmc_fd_xs.npy", xs)

# import numpy as np
import matplotlib.pyplot as plt

function_dict = {
    "sphere": build_and_run_sphere,
    "torus": build_and_run_torus
}

def load_data_fd(name):
    values = np.load(f"{name}_openmc_fd_value.npy")
    gradient = np.load(f"{name}_openmc_fd_gradient.npy")
    errors = np.load(f"{name}_openmc_fd_gradient_errors.npy")
    xs     = np.load(f"{name}_openmc_fd_xs.npy")

    return xs, values, gradient, errors

def plot_with_errors(name):
    xs, values, gradient, gradient_errors = load_data_fd(name)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 6)
    )

    # Color-blind friendly colormap
    cmap = plt.get_cmap("cividis")
    color_value = cmap(0.35)
    color_grad  = cmap(0.75)

    print(xs)

    # --- Value plot ---
    ax1.plot(
        xs,
        values,
        "o-",
        linewidth=2,
        markersize=5,
        color=color_value,
        label="Value",
    )
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Gradient plot with error bars ---
    ax2.errorbar(
        xs,
        gradient,
        yerr=gradient_errors,
        fmt="o-",
        linewidth=2,
        markersize=5,
        capsize=4,
        color=color_grad,
        label="Gradient",
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("Gradient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name}_gradient.png", bbox_inches="tight")
    plt.close(fig)



def sample_sigma_t(mu, theta, size=1, rng=None):
    """
    Sample sigma_t ~ N(mu, theta^2)

    Parameters
    ----------
    mu : float
        Mean of the Gaussian
    theta : float
        Standard deviation
    size : int
        Number of samples
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Samples of sigma_t
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=mu, scale=theta, size=size)

def save_uncertainty_file(name, values, inputs):
    np.save(f"{name}_values.npy", values)
    np.save(f"{name}_inputs.npy", inputs)

def uncertainty_quantification_sphere(name, mu, std, N, source_power, num_particles, func):
    sig = sample_sigma_t(mu, std, N)
    values = []
    print("sig: ", sig)
    for i in range(N):
        print("sig_t: ", i, sig[i])
        Etot = func(0.5, sig[i], 0.9, source_power, num_particles, i + 1994, translation=[0, 0, 0])
        values.append(Etot)
    Etot_array = np.array(values)
    save_uncertainty_file(name, Etot_array, sig)
    

# Example usage
def finite_difference_sig_t(name, sig_t_range, steps, N, delta, sigma_t, albedo, source_power, num_particles, func):

    step_size = (sig_t_range[1] - sig_t_range[0]) / steps
    print("step_size", step_size)

    sig_t = sig_t_range[0]
    value_list = []
    gradient_list = []
    error_list = []
    radius_list = []
    for i in range(steps+1):
        gradient_at_r = []
        value_at_r = []
        for j in range(N):
            cur_seed = i * N + j + 1994
            t = func(0.3, sig_t, albedo, source_power, num_particles, cur_seed)
            t1 = func(0.3, sig_t + delta, albedo, source_power, num_particles, cur_seed)
            t2 = func(0.3, sig_t - delta, albedo, source_power, num_particles, cur_seed)
            gradient_r = (t1 - t2) / (delta * 2.0)
            gradient_at_r.append(gradient_r)
            value_at_r.append(t)
        gradient_at_r_np = np.array(gradient_at_r)
        gradient_list.append(np.mean(gradient_at_r_np))
        error_list.append(np.std(gradient_at_r_np))
        radius_list.append(sig_t)
        value_list.append(np.mean(value_at_r))
        sig_t = sig_t + step_size
    save_data(f"{name}", np.array(value_list), np.array(gradient_list), np.array(error_list), np.array(radius_list))

def finite_difference_param(param_range, param_name, param_set, shape_name, steps, N, delta, source_power, num_particles, id=1):
    step_size = (param_range[1] - param_range[0]) / steps
    print("step_size", step_size)

    r = param_range[0]
    value_list = []
    gradient_list = []
    error_list = []
    radius_list = []
    value_std_list = []

    function_call = function_dict[shape_name]

    for i in range(steps+1):
        gradient_at_r = []
        value_at_r = []
        for j in range(N):
            cur_seed = i * N + j + 1994
            param_set[param_name] = r
            t = function_call(param_set, source_power, num_particles, cur_seed)
            param_set[param_name] = r + delta 
            t1 = function_call(param_set, source_power, num_particles, cur_seed)
            param_set[param_name] = r - delta
            t2 = function_call(param_set, source_power, num_particles, cur_seed)
            gradient_r = (t1 - t2) / (delta * 2.0)
            gradient_at_r.append(gradient_r)
            value_at_r.append(t)
        gradient_at_r_np = np.array(gradient_at_r)
        gradient_list.append(np.mean(gradient_at_r_np))
        error_list.append(np.std(gradient_at_r_np))
        radius_list.append(r)
        value_list.append(np.mean(value_at_r))
        value_std_list.append(np.std(value_at_r))
        r = r + step_size
    save_data(f"{shape_name}_{param_name}_{id}", np.array(value_list), np.array(gradient_list), np.array(error_list), np.array(radius_list), np.array(value_std_list))

def finite_difference_radius(radius_range, steps, N, delta, sigma_t, albedo, source_power, num_particles):

    step_size = (radius_range[1] - radius_range[0]) / steps
    print("step_size", step_size)

    r = radius_range[0]
    value_list = []
    gradient_list = []
    error_list = []
    radius_list = []
    for i in range(steps+1):
        gradient_at_r = []
        value_at_r = []
        for j in range(N):
            cur_seed = i * N + j + 1994
            t = build_and_run(r, sigma_t, albedo, source_power, num_particles, cur_seed)
            t1 = build_and_run(r + delta, sigma_t, albedo, source_power, num_particles, cur_seed)
            t2 = build_and_run(r - delta, sigma_t, albedo, source_power, num_particles, cur_seed)
            gradient_r = (t1 - t2) / (delta * 2.0)
            gradient_at_r.append(gradient_r)
            value_at_r.append(t)
        gradient_at_r_np = np.array(gradient_at_r)
        gradient_list.append(np.mean(gradient_at_r_np))
        error_list.append(np.std(gradient_at_r_np))
        radius_list.append(r)
        value_list.append(np.mean(value_at_r))
        r = r + step_size
    save_data("radius_2", np.array(value_list), np.array(gradient_list), np.array(error_list), np.array(radius_list))
    

        

if __name__ == "__main__":
    # R_SPHERE_CM = 2.0

    # SIGMA_T = 0.5   # 1/cm
    # SIGMA_A = 0.1   # 1/cm
    # SIGMA_S = SIGMA_T - SIGMA_A  # 0.4 1/cm

    # Source energy (monoenergetic). If you truly want an energy distribution,
    # we can replace this with a spectrum and integrate with an energy-filtered tally.
    source_power = 1.0e6  # 1 MeV
    sigma_t = 0.5
    albedo = 0.8

    num_particles = 300000  # increase for tighter uncertainty
    function_sim = build_and_run_torus
    # function_sim = build_and_run_sphere
    steps = 25
    N = 10
    delta = 0.0002
    
    # finite_difference_radius([1.0, 2.0], 3, 5, 0.002, sigma_t, albedo, source_power, num_particles)
    # finite_difference_sig_t("geo", [0.1, 2.0], 25, 5, delta, sigma_t, albedo, source_power, num_particles, function_sim)
    param_set = {
        "sig_t": 0.1,
        "albedo": 0.9,
        "geo": 0.1
    }
    param_range = [-0.195, 0.195]
    albedo_range = [0.8, 0.995]
    sig_range = [0.1, 2.0]
    sphere_scale_range = [0.1, 2.0]
    id = 1
    # finite_difference_param(sphere_scale_range, "geo", param_set, "sphere", steps, N, delta, source_power, num_particles, id=1)
    plot_with_errors(f"torus_geo_{id}")
    # uncertainty_quantification_sphere("uncertainty", 0.5, 0.1, 100, source_power, num_particles, function_sim)