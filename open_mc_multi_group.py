import numpy as np
import openmc
import openmc.mgxs
import math
import matplotlib.pyplot as plt

BATCHES=50

def build_mg_from_probs(sig_t, albedo, P=None, G=3, enforce_downscatter=False):
    """
    sig_t: (G,) macroscopic total XS per incident group
    albedo:   (G,) probability of scattering in each incident group
    P:     (G,G) conditional probabilities P[g_in, g_out] given a scatter
           Columns must sum to 1 (for each g_in).
           If None, default is within-group only (identity).
    Returns:
      sig_t (G,)
      sig_a (G,)
      sig_s_tot (G,)           # total scattering XS per incident group
      P (G,G)                  # conditional redistribution probabilities
      sig_s_mat (G,G)          # OpenMC-ready scatter XS matrix (g_in, g_out)
    """
    sig_t = np.asarray(sig_t, dtype=float).reshape((G,))
    albedo = np.asarray(albedo, dtype=float).reshape((G,))

    # Clamp for safety
    albedo = np.clip(albedo, 0.0, 1.0)

    sig_s_tot = albedo * sig_t
    sig_a = sig_t - sig_s_tot  # = (1-p_s)*sig_t

    # Default redistribution: within-group only
    if P is None:
        P = np.eye(G, dtype=float)
    else:
        P = np.asarray(P, dtype=float).reshape((G, G))

    # print(" ------------- P: ", P)
    # Optional: enforce only downscatter + within-group (g_out <= g_in)
    if enforce_downscatter:
        mask = np.triu(np.ones((G, G), dtype=bool), k=0)  # g_out <= g_in if we interpret indices low->high?
        # Careful: define group ordering. If group 0 is lowest energy, downscatter means g_out < g_in.
        # If group 0 is highest energy, reverse. Here we assume group 0 is LOW energy (like your edges),
        # so downscatter means g_out <= g_in is *upscatter*, not downscatter.
        # With your edges [0, 1e5, 1e6, 2e7], group index increases with energy.
        # Downscatter (energy decreases) means g_out < g_in.
        mask = np.zeros((G, G), dtype=bool)
        for g_in in range(G):
            for g_out in range(G):
                if g_out <= g_in:  # lower-or-equal energy index => downscatter or same
                    mask[g_out, g_in] = True
        P = np.where(mask, P, 0.0)
    
    # Normalize columns to sum to 1 (conditional on scattering)
    row_sums = P.sum(axis=1, keepdims=True)
    # print(" ------------- P: ", P)

    if np.any(row_sums <= 0):
        bad = np.where(row_sums.ravel() <= 0)[0].tolist()
        raise ValueError(f"Some incident-group row in P sum to 0 (groups {bad}). "
                         f"Each g_in must have at least one allowed g_out with nonzero prob.")
    P = P / row_sums
    # print(" ------------- P: ", P)

    # Build OpenMC scatter XS matrix Σs[g_out, g_in] = Σs_tot[g_in] * P[g_out, g_in]
    print("sig_s_tot:", sig_s_tot)
    sig_s_mat = sig_s_tot.reshape((1, G)) * P.transpose()
    print("sig_s_mat:", sig_s_mat)
    print("sig P", P)

    return sig_t, sig_a, sig_s_tot, P, sig_s_mat.transpose()


def _validate_mgxs(sig_t, sig_a, sig_s_mat, name=""):
    """Check non-negativity and Σt = Σa + Σs_total per incident group."""
    sig_t = np.asarray(sig_t, float).reshape((3,))
    sig_a = np.asarray(sig_a, float).reshape((3,))
    sig_s_mat = np.asarray(sig_s_mat, float).reshape((3, 3))

    if np.any(sig_t < 0) or np.any(sig_a < 0) or np.any(sig_s_mat < 0):
        raise ValueError(f"[{name}] Negative XS detected.")

    # OpenMC convention: sig_s_mat[g_out, g_in]
    sig_s_tot = sig_s_mat.sum(axis=1)  # sum over g_out for each g_in
    print("sig_s_tot", sig_s_tot)
    print("sig_a", sig_a)
    print("sig_t",sig_t)
    # exit(0)
    err = sig_t - (sig_a + sig_s_tot)

    if np.max(np.abs(err)) > 1e-10:
        raise ValueError(
            f"[{name}] XS not conserved per group.\n"
            f"sig_t       = {sig_t}\n"
            f"sig_a       = {sig_a}\n"
            f"sig_s_tot   = {sig_s_tot}\n"
            f"sig_t-(a+s) = {err}\n"
            f"Reminder: sig_s_mat must be (g_out, g_in)."
        )


def make_3group_mgxs_two_materials(
    mats,
    group_edges_eV,
    filename="mgxs.h5",
    scatter_order=0,
    validate=True,
):
    """
    Create a 3-group MGXS library (HDF5) for OpenMC.

    Parameters
    ----------
    mats : list of tuples
        Each tuple: (xs_name, sig_t, sig_a, sig_s)
        - xs_name: str
        - sig_t: array-like shape (3,)
        - sig_a: array-like shape (3,)
        - sig_s: either
            * shape (3,)  -> within-group scatter only (diagonal)
            * shape (3,3) -> full group-to-group scatter matrix in OpenMC convention:
                            sig_s[g_out, g_in]

    group_edges_eV : array-like
        4 energy edges in eV (length 4), defining 3 groups.
        Example: [0.0, 1e5, 1e6, 2e7]

    filename : str
        Output HDF5 filename (e.g., "mgxs.h5")

    scatter_order : int
        Legendre order (0 means isotropic in angle).

    validate : bool
        If True, checks Σt = Σa + Σs_total per incident group.
    """
    # --- validate group edges ---
    edges = np.asarray(group_edges_eV, dtype=float).ravel()
    if len(edges) != 4:
        raise ValueError(f"3-group requires 4 edges; got {len(edges)} edges: {edges}")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"group_edges must be strictly increasing; got {edges}")

    G = len(edges) - 1
    if G != 3:
        raise ValueError(f"Expected 3 groups, got G={G} from edges {edges}")

    # --- build MG library ---
    groups = openmc.mgxs.EnergyGroups(group_edges=edges.tolist())
    mg_lib = openmc.MGXSLibrary(groups)
    mg_lib.scatter_format = "legendre"
    mg_lib.scatter_order = int(scatter_order)

    for xs_name, sig_t, sig_a, sig_s in mats:
        sig_t = np.asarray(sig_t, dtype=float).reshape((G,))
        sig_a = np.asarray(sig_a, dtype=float).reshape((G,))

        # Build scatter matrix of shape (G, G, order+1)
        # OpenMC convention: scatter_matrix[g_out, g_in, ell]
        scat = np.zeros((G, G, scatter_order + 1), dtype=float)

        sig_s_arr = np.asarray(sig_s, dtype=float)
        
        if sig_s_arr.ndim == 1:
            # Diagonal-only scatter (within-group)
            sig_s_arr = sig_s_arr.reshape((G,))
            for g in range(G):
                scat[g, g, 0] = sig_s_arr[g]
            sig_s_mat_for_check = np.diag(sig_s_arr)
        elif sig_s_arr.ndim == 2:
            # Full group-to-group scatter
            if sig_s_arr.shape != (G, G):
                raise ValueError(f"sig_s matrix must be {(G, G)}; got {sig_s_arr.shape}")
            scat[:, :, 0] = sig_s_arr
            
            sig_s_mat_for_check = sig_s_arr
            # print(" sig_a", sig_a)
            # print(" sig_t", sig_t)
            # print(" sig_s", sig_s)
            # print("for check: ", sig_s_mat_for_check)
        else:
            raise ValueError("sig_s must be shape (3,) or (3,3)")

        if validate:
            _validate_mgxs(sig_t, sig_a, sig_s_mat_for_check, name=xs_name)

        xsdata = openmc.XSdata(xs_name, groups)
        xsdata.order = scatter_order
        xsdata.set_total(sig_t)
        xsdata.set_absorption(sig_a)
        xsdata.set_scatter_matrix(scat)

        mg_lib.add_xsdata(xsdata)

    mg_lib.export_to_hdf5(filename)
    return filename

def build_and_run_torus_multi_group(param_set, SOURCE_E_EV, PARTICLES, seed_value):
    # Row out_going, Column in_going
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    sig_t_shell = [0.9, 0.9, 0.9]
    sig_a_shell = [0.1, 0.1, 0.1]

    # sig_s[g_out, g_in]
    sig_s_shell = np.array([
        [0.60, 0.10, 0.10],
        [0.00, 0.40, 0.40],
        [0.00, 0.00, 0.80],
    ])

    sig_t = [param_set["sig_t"], 0.8, 0.8]
    albedo = [0.7, 0.7, 0.7]
    Phase_matrix =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    sig_t_inner, sig_a_inner, sig_s_tot_inner, P_inner, sig_s_mat_inner = build_mg_from_probs(sig_t, albedo, P=Phase_matrix, G=3, enforce_downscatter=False)
    
    # energy all get absorbed
    sig_t_vaccum = [0.0, 0.0, 0.0]
    albedo_vaccum = [1.0, 1.0, 1.0]
    Phase_matrix_vaccum = np.array([
        [1.0, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00],
    ])

    sig_t_vaccum, sig_a_vaccum, sig_s_tot_vaccum, P_vaccum, sig_s_mat_vaccum = build_mg_from_probs(sig_t_vaccum, albedo_vaccum, P=Phase_matrix_vaccum, G=3, enforce_downscatter=False)

    def to_g(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.full((G,), float(x))
        x = x.reshape((G,))
        return x



    # 1) Write 3-group MGXS file and point OpenMC to it
    make_3group_mgxs_two_materials(
        mats=[
            ("mat",  sig_t_shell, sig_a_shell, sig_s_shell),
            ("mat2", sig_t_inner, sig_a_inner, sig_s_mat_inner),
            ("vaccum", sig_t_vaccum, sig_a_vaccum, sig_s_mat_vaccum)
        ],
        group_edges_eV=group_edges,
        filename="mgxs.h5"
    )
    openmc.config["mg_cross_sections"] = "mgxs.h5"

    # 2) Materials using macroscopic MGXS
    macro = openmc.Macroscopic("mat")
    mat_shell = openmc.Material(name="shell")
    mat_shell.set_density("macro", 1.0)
    mat_shell.add_macroscopic(macro)

    macro2 = openmc.Macroscopic("mat2")
    mat_inner = openmc.Material(name="inner")
    mat_inner.set_density("macro", 1.0)
    mat_inner.add_macroscopic(macro2)

    macro2 = openmc.Macroscopic("vaccum")
    mat_void = openmc.Material(name="vaccum")
    mat_void.set_density("macro", 1.0)
    mat_void.add_macroscopic(macro2)

    materials = openmc.Materials([mat_inner, mat_shell, mat_void])

    # 3) Geometry
    # print("geomery: ----------", param_set["geo"])
    R_MAJOR_IN  = 1.0   # distance from center to tube centerline
    R_MINOR_IN  = 0.3   # tube radius (minor)
    R_MAJOR_OUT = 1.0
    R_MINOR_OUT = 0.5

    outer = openmc.Sphere(r=2.0, boundary_type="vacuum") 
    tor_in  = openmc.ZTorus(x0=0.0, y0=0.0, z0=param_set["geo"], a=R_MAJOR_IN,  b=R_MINOR_IN,  c=R_MINOR_IN, boundary_type="transmission")
    tor_out = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R_MAJOR_OUT, b=R_MINOR_OUT, c=R_MINOR_OUT, boundary_type="transmission")

    cell_in = openmc.Cell(region=-tor_in, fill=mat_inner)
    cell_sh = openmc.Cell(region=(-tor_out & +tor_in), fill=mat_shell)
    cell_out = openmc.Cell(region=(+tor_out & -outer), fill=mat_void)
    
    geom = openmc.Geometry(openmc.Universe(cells=[cell_in, cell_sh, cell_out]))

    # 4) Settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.energy_mode = "multi-group"
    settings.batches = BATCHES
    settings.particles = PARTICLES
    settings.seed = seed_value

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()

    # print("g_idx: ", g_idx)
    # exit(0)

    # In multi-group mode, OpenMC expects source energy in terms of GROUPS.
    # Use Discrete over group indices 1..G (OpenMC groups are 1-based).
    # src.energy = openmc.stats.Discrete([1], [1.0])
    src.energy = openmc.stats.Discrete([1.5e+06], [1.0])
    settings.source = src

    # 5) Tally: optionally add an EnergyFilter to see per-group leakage
    NMAX = 4
    coll_bins = list(range(NMAX + 1))

    tallies = openmc.Tallies()
    t = openmc.Tally(name="leakage_current")
    t.filters = [
        openmc.SurfaceFilter(outer),
        openmc.CollisionFilter(coll_bins),
        openmc.EnergyFilter(group_edges),  # <-- optional, gives groupwise current
    ]
    t.scores = ["current"]
    tallies.append(t)

    model = openmc.Model(materials=materials, geometry=geom, settings=settings, tallies=tallies)
    model.export_to_xml()
    openmc.run()

    # 6) Postprocess
    sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")
    tally = sp.get_tally(name="leakage_current")
    
    mean = tally.mean  # now 3D-ish because of collision bins and energy bins
    std = tally.std_dev
    df = tally.get_pandas_dataframe()
    print(df)
    # print(tally)
    # print("openmc_group = = = = ", openmc_group, asc_idx)

    # print("mean shape:", mean.shape)
    # print(mean)
    # print("std shape", std.shape)
    mean_reshape = mean.reshape(NMAX+1, G)
  
    # print(mean_reshape.shape, mean_reshape)
    mean_energies = np.sum(mean_reshape, axis=0)
    # print(mean_energies.shape, mean_energies)
    # exit(0)
    # print(mean_reshape.shape)
    # print(mean_energies.shape)
    # exit(0)
    # std_energies = np.sum(std.reshape(G, NMAX+1), axis=0)
    # If you want total current summed over collisions and energy:
    # total_current = float(mean.sum())
    return mean_energies

def build_and_run_sphere_multi_group(param_set, SOURCE_E_EV, PARTICLES, seed_value):
    # Row out_going, Column in_going
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    sig_t_shell = [0.9, 0.9, 0.9]
    sig_a_shell = [0.1, 0.1, 0.1]

    # sig_s[g_out, g_in]
    sig_s_shell = np.array([
        [0.60, 0.10, 0.10],
        [0.00, 0.40, 0.40],
        [0.00, 0.00, 0.80],
    ])

    sig_t = [param_set["sig_t"], 0.8, 0.8]
    albedo = [0.7, 0.7, 0.7]
    Phase_matrix =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    sig_t_inner, sig_a_inner, sig_s_tot_inner, P_inner, sig_s_mat_inner = build_mg_from_probs(sig_t, albedo, P=Phase_matrix, G=3, enforce_downscatter=False)
    # print(sig_t_inner, sig_a_inner, sig_s_tot_inner, sig_s_mat_inner)
    # exit(0)
    def to_g(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.full((G,), float(x))
        x = x.reshape((G,))
        return x



    # 1) Write 3-group MGXS file and point OpenMC to it
    make_3group_mgxs_two_materials(
        mats=[
            ("mat",  sig_t_shell, sig_a_shell, sig_s_shell),
            ("mat2", sig_t_inner, sig_a_inner, sig_s_mat_inner),
        ],
        group_edges_eV=group_edges,
        filename="mgxs.h5"
    )
    openmc.config["mg_cross_sections"] = "mgxs.h5"

    # 2) Materials using macroscopic MGXS
    macro = openmc.Macroscopic("mat")
    mat = openmc.Material(name="shell")
    mat.set_density("macro", 1.0)
    mat.add_macroscopic(macro)

    macro2 = openmc.Macroscopic("mat2")
    mat2 = openmc.Material(name="inner")
    mat2.set_density("macro", 1.0)
    mat2.add_macroscopic(macro2)

    materials = openmc.Materials([mat2, mat])

    # 3) Geometry
    # print("geomery: ----------", param_set["geo"])
    sph = openmc.Sphere(x0=0, y0=0, z0=0, r=param_set["geo"])
    sph_outer = openmc.Sphere(r=3.0, boundary_type="vacuum")

    cell_in = openmc.Cell(region=-sph, fill=mat2)
    cell = openmc.Cell(region=-sph_outer & +sph, fill=mat)
    geom = openmc.Geometry(openmc.Universe(cells=[cell_in, cell]))

    # 4) Settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.energy_mode = "multi-group"
    settings.batches = BATCHES
    settings.particles = PARTICLES
    settings.seed = seed_value

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()

    # In multi-group mode, OpenMC expects source energy in terms of GROUPS.
    # Use Discrete over group indices 1..G (OpenMC groups are 1-based).
    # src.energy = openmc.stats.Discrete([1], [1.0])
    src.energy = openmc.stats.Discrete([1.5e+06], [1.0])
    settings.source = src

    # 5) Tally: optionally add an EnergyFilter to see per-group leakage
    NMAX = 4
    coll_bins = list(range(NMAX + 1))

    tallies = openmc.Tallies()
    t = openmc.Tally(name="leakage_current")
    t.filters = [
        openmc.SurfaceFilter(sph_outer),
        openmc.CollisionFilter(coll_bins),
        openmc.EnergyFilter(group_edges),  # <-- optional, gives groupwise current
    ]
    t.scores = ["current"]
    tallies.append(t)

    model = openmc.Model(materials=materials, geometry=geom, settings=settings, tallies=tallies)
    model.export_to_xml()
    openmc.run()

    # 6) Postprocess
    sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")
    tally = sp.get_tally(name="leakage_current")
    
    mean = tally.mean  # now 3D-ish because of collision bins and energy bins
    std = tally.std_dev
    df = tally.get_pandas_dataframe()
    print(df)

    mean_reshape = mean.reshape(NMAX+1, G)
    mean_energies = np.sum(mean_reshape, axis=0)
    return mean_energies

def build_and_run_sphere_multi_group_sensor(param_set, SOURCE_E_EV, PARTICLES, seed_value):
    # Row out_going, Column in_going
    G = 3
    group_edges = [0.0, 1.0e5, 1.0e6, 2.0e7]  # eV

    sig_t_shell = [0.9, 0.9, 0.9]
    sig_a_shell = [0.1, 0.1, 0.1]

    # sig_s[g_out, g_in]
    sig_s_shell = np.array([
        [0.60, 0.10, 0.10],
        [0.00, 0.40, 0.40],
        [0.00, 0.00, 0.80],
    ])

    sig_t = [param_set["sig_t"], 0.8, 0.8]
    albedo = [0.7, 0.7, 0.7]
    Phase_matrix =  np.array([
        [0.10, 0.90, 0.00],
        [0.00, 0.50, 0.50],
        [0.00, 0.00, 1.00],
    ])

    sig_t_inner, sig_a_inner, sig_s_tot_inner, P_inner, sig_s_mat_inner = build_mg_from_probs(sig_t, albedo, P=Phase_matrix, G=3, enforce_downscatter=False)

    # energy all get absorbed
    # sig_t_vaccum = [1e6, 1e6, 1e6]
    # albedo_vaccum = [0.0, 0.0, 0.0]
    # Phase_matrix_vaccum = np.array([
    #     [1.0, 0.00, 0.00],
    #     [0.00, 1.00, 0.00],
    #     [0.00, 0.00, 1.00],
    # ])

    # sig_t_vaccum, sig_a_vaccum, sig_s_tot_vaccum, P_vaccum, sig_s_mat_vaccum = build_mg_from_probs(sig_t_vaccum, albedo_vaccum, P=Phase_matrix_vaccum, G=3, enforce_downscatter=False)
    
    def to_g(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.full((G,), float(x))
        x = x.reshape((G,))
        return x

    # 1) Write 3-group MGXS file and point OpenMC to it
    make_3group_mgxs_two_materials(
        mats=[
            ("shell",  sig_t_shell, sig_a_shell, sig_s_shell),
            ("inner", sig_t_inner, sig_a_inner, sig_s_mat_inner),
            # ("vaccum", sig_t_vaccum, sig_a_vaccum, sig_s_mat_vaccum),
        ],
        group_edges_eV=group_edges,
        filename="mgxs.h5"
    )
    openmc.config["mg_cross_sections"] = "mgxs.h5"

    # 2) Materials using macroscopic MGXS
    macro = openmc.Macroscopic("shell")
    mat = openmc.Material(name="shell")
    mat.set_density("macro", 1.0)
    mat.add_macroscopic(macro)

    macro2 = openmc.Macroscopic("inner")
    mat2 = openmc.Material(name="inner")
    mat2.set_density("macro", 1.0)
    mat2.add_macroscopic(macro2)

    # macro2 = openmc.Macroscopic("vaccum")
    # mat_vac = openmc.Material(name="vaccum")
    # mat_vac.set_density("macro", 1.0)
    # mat_vac.add_macroscopic(macro2)

    materials = openmc.Materials([mat2, mat])

    # 3) Geometry
    # print("geomery: ----------", param_set["geo"])
    sph = openmc.Sphere(x0=0, y0=0, z0=0, r=param_set["geo"])
    sph_outer = openmc.Sphere(r=3.0, boundary_type="vacuum")
    sph_sensor = openmc.Sphere(x0=1.0, y0=1.0, z0=1.0, r=0.05, boundary_type="vacuum")

    cell_in = openmc.Cell(region=-sph & +sph_sensor, fill=mat2)
    cell = openmc.Cell(region=-sph_outer & +sph & +sph_sensor, fill=mat)
    cell_vaccum = openmc.Cell(region = -sph_sensor)
    geom = openmc.Geometry(openmc.Universe(cells=[cell_in, cell, cell_vaccum]))

    # 4) Settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.energy_mode = "multi-group"
    settings.batches = BATCHES
    settings.particles = PARTICLES
    settings.seed = seed_value

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()

    # In multi-group mode, OpenMC expects source energy in terms of GROUPS.
    # Use Discrete over group indices 1..G (OpenMC groups are 1-based).
    # src.energy = openmc.stats.Discrete([1], [1.0])
    src.energy = openmc.stats.Discrete([SOURCE_E_EV], [1.0])
    settings.source = src

    # 5) Tally: optionally add an EnergyFilter to see per-group leakage
    NMAX = 4
    coll_bins = list(range(NMAX + 1))

    tallies = openmc.Tallies()
    t = openmc.Tally(name="leakage_current")
    t.filters = [
        openmc.SurfaceFilter(sph_sensor),
        openmc.CollisionFilter(coll_bins),
        openmc.EnergyFilter(group_edges),  # <-- optional, gives groupwise current
    ]
    t.scores = ["current"]
    tallies.append(t)

    model = openmc.Model(materials=materials, geometry=geom, settings=settings, tallies=tallies)
    model.export_to_xml()
    openmc.run()

    # 6) Postprocess
    sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")
    tally = sp.get_tally(name="leakage_current")
    
    mean = tally.mean  # now 3D-ish because of collision bins and energy bins
    std = tally.std_dev
    df = tally.get_pandas_dataframe()
    # print(df["mean"])

    mean_reshape = mean.reshape(NMAX+1, G)
    mean_energies = np.sum(mean_reshape, axis=0)
    print("mean energies:", mean_energies)
    return mean_energies



function_dict = {
    "sphere": build_and_run_sphere_multi_group,
    "torus": build_and_run_torus_multi_group
}

def save_data(name, values, gradient, gradient_errors, xs, val_std):
    np.save(f"{name}_openmc_fd_value.npy", values)
    np.save(f"{name}_openmc_fd_value_errors.npy", val_std)
    np.save(f"{name}_openmc_fd_gradient.npy", gradient)
    np.save(f"{name}_openmc_fd_gradient_errors.npy", gradient_errors)
    np.save(f"{name}_openmc_fd_xs.npy", xs)

def finite_difference_param(param_range, param_name, param_set, shape_name, steps, N, delta, source_power, num_particles, id=1):
    step_size = (param_range[1] - param_range[0]) / steps
    # print("step_size", step_size)

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
        gradient_list.append(np.mean(gradient_at_r_np, axis=0))
        error_list.append(np.std(gradient_at_r_np, axis=0))

        radius_list.append(r)
        value_list.append(np.mean(value_at_r, axis=0))
        value_std_list.append(np.std(value_at_r, axis=0))

        r = r + step_size
    save_data(f"{shape_name}_{param_name}_multi_{id}", np.array(value_list), np.array(gradient_list), np.array(error_list), np.array(radius_list), np.array(value_std_list))

def load_data_fd(name):
    values = np.load(f"{name}_openmc_fd_value.npy")
    gradient = np.load(f"{name}_openmc_fd_gradient.npy")
    errors = np.load(f"{name}_openmc_fd_gradient_errors.npy")
    xs     = np.load(f"{name}_openmc_fd_xs.npy")

    return xs, values, gradient, errors

def plot_with_errors(name):
    xs, values, gradient, gradient_errors = load_data_fd(name)

    xs = np.asarray(xs)
    values = np.asarray(values)
    gradient = np.asarray(gradient)
    gradient_errors = np.asarray(gradient_errors)

    if values.ndim != 2:
        raise ValueError(f"`values` must be (N, G). Got shape {values.shape}")
    if gradient.ndim != 2:
        raise ValueError(f"`gradient` must be (N, G). Got shape {gradient.shape}")
    if gradient_errors.ndim != 2:
        raise ValueError(f"`gradient_errors` must be (N, G). Got shape {gradient_errors.shape}")

    N, G = values.shape
    if gradient.shape != (N, G) or gradient_errors.shape != (N, G):
        raise ValueError(
            "Shape mismatch. Expected `gradient` and `gradient_errors` to match `values`.\n"
            f"values: {values.shape}, gradient: {gradient.shape}, gradient_errors: {gradient_errors.shape}"
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 7))

    # Use a color-blind friendly colormap; pick distinct colors for each group
    cmap = plt.get_cmap("cividis")
    if G == 1:
        colors = [cmap(0.6)]
    else:
        colors = [cmap(t) for t in np.linspace(0.25, 0.85, G)]

    # --- Value plot: one curve per energy group ---
    for g in range(G):
        ax1.plot(
            xs,
            values[:, g],
            "o-",
            linewidth=2,
            markersize=5,
            color=colors[g],
            label=f"Group {g+1}",
        )
    ax1.set_ylabel("Value")
    ax1.legend(title="Energy group", ncol=min(G, 3))
    ax1.grid(True, alpha=0.3)

    # --- Gradient plot with error bars: one curve per energy group ---
    for g in range(G):
        ax2.errorbar(
            xs,
            gradient[:, g],
            yerr=gradient_errors[:, g],
            fmt="o-",
            linewidth=2,
            markersize=5,
            capsize=4,
            color=colors[g],
            label=f"Group {g+1}",
        )
    ax2.set_xlabel("x")
    ax2.set_ylabel("Gradient")
    ax2.legend(title="Energy group", ncol=min(G, 3))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name}_gradient.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sphere_scale_range = [0.1, 2.0]
    sig_t_range = [0.1, 2.0]
    # build_and_run_sphere_multi_group({"geo":0.2}, 1.5e+07, 10000, 1994)
    param_set = {"geo":1.0,
    "sig_t": 0.9,
    }

    torus_param_set = {"geo":0.0,
    "sig_t": 0.9,
    }

    torus_geo_range = [-0.195, 0.195]

    steps = 25
    N = 20
    delta = 0.01
    source_power = 1.5e+07
    num_particles = 300000
    id = 2
    v = []
    for i in range(N):
        v.append(build_and_run_sphere_multi_group_sensor(param_set, source_power, num_particles, 1994+i))
    v = np.mean(np.array(v), axis=0)
    print("Total: ", v)
    # finite_difference_param(torus_geo_range, "geo", torus_param_set, "torus", steps, N, delta, source_power, num_particles, id)
    # finite_difference_param(sig_t_range, "sig_t", param_set, "sphere", steps, N, delta, source_power, num_particles, id)
    # plot_with_errors(f"torus_geo_multi_{id}")



