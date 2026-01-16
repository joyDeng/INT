# spheres_mg_openmc.py
# Run with Docker (PowerShell):
#   docker run --rm -it -v "${PWD}:/work" -w /work openmc/openmc:v0.13.2 python3 spheres_mg_openmc.py

from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path
import numpy as np

# =========================
# User-tweakable defaults
# =========================
RESULT_DIR     = "results_spheres"
SEED           = 12345

# Geometry radii (meters)
R_INNER        = 0.50
R_OUTER        = 1.00
R_BOUND        = 3.00

# Source: point outside, monodirectional toward origin
SRC_POS        = (1.80, 0.0, 0.0)     # must be > R_OUTER
SRC_DIR        = (-1.0, 0.0, 0.0)
SRC_E_EV       = 2.0                  # eV (inside 1–20 eV group)

# Run controls
PARTICLES      = 100_000
BATCHES        = 50
MAX_BOUNCE     = 5                    # K (max scatters allowed via staging, via sub-bins)

# Real energy grid: 3 groups, ascending energies (eV)
E_REAL_ASC     = np.array([1.0e-5, 0.5, 1.0, 20.0], dtype=float)

# Toy macroscopic XS in H/M/L order — will be mapped to engine order internally
SIGMA_T_HML = np.array([
    [1.0, 5.0, 2.0],   # inner
    [0.7, 1.1, 0.6],   # outer
], dtype=float)

ALBEDO_HML = np.array([
    [0.95, 0.95, 0.95],
    [0.97, 0.97, 0.96],
], dtype=float)

# Per-material phase (row sums = 1). Indices: [mat, incoming(H/M/L), outgoing(H/M/L)].
PHASE_P_HML = np.array([
    [[0.08, 0.72, 0.20],
     [0.08, 0.72, 0.20],
     [0.10, 0.25, 0.65]],  # inner
    [[0.80, 0.18, 0.02],
     [0.05, 0.92, 0.03],
     [0.00, 0.08, 0.92]],  # outer
], dtype=float)

# =========================
# Helpers
# =========================
def _assert_finite_nonzero(x, context):
    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite values in {context}")
    if np.all(x == 0.0):
        raise RuntimeError(f"All-zero values in {context}")

def _list_group(g, indent=""):
    try:
        import h5py
    except Exception:
        return
    items = []
    for name, obj in g.items():
        try:
            import h5py as _h
            if isinstance(obj, _h.Group):
                items.append(f"{indent}[G] {name}/")
            else:
                items.append(f"{indent}[D] {name}  shape={obj.shape}")
        except Exception:
            pass
    for s in sorted(items):
        print(s)

def _find_temp_group(gmat):
    # Find the single temperature subgroup like '294K'
    for key, obj in gmat.items():
        if key.endswith('K') and hasattr(obj, 'keys'):
            return obj
    return None

def _verify_scatter_in_hdf5(h5_path: str):
    """Log the actual datasets OpenMC uses (mat/xxxK/scatter_data/{scatter_matrix, multiplicity_matrix})."""
    try:
        import h5py
    except Exception:
        print("[note] h5py not available; skipping scatter verification.")
        return

    with h5py.File(h5_path, "r") as f:
        for root_name in ("xsdata", "macroscopic"):
            if root_name not in f:
                continue
            root = f[root_name]
            print(f"[verify] Listing '{root_name}/' contents:")
            _list_group(root, indent="  ")
            for mat in root.keys():
                gmat = root[mat]
                gt = _find_temp_group(gmat)
                if gt is None:
                    print(f"[verify] [{root_name}] {mat:>6s}  no temperature subgroup found")
                    continue
                sd = gt.get("scatter_data", None)
                if sd is None:
                    print(f"[verify] [{root_name}] {mat:>6s}  no 'scatter_data' subgroup")
                    continue
                s3 = sd.get("scatter_matrix", None)
                mm = sd.get("multiplicity_matrix", None)
                if s3 is not None:
                    A = s3[...]
                    if A.ndim == 3 and A.shape[2] >= 1:
                        S = A[:, :, 0]  # P0
                        rs = S.sum(axis=1)
                        cs = S.sum(axis=0)
                        print(f"[verify] [{root_name}] {mat:>6s}  scatter_matrix(3D) shape={A.shape}  "
                              f"row-sum (out) min/mean/max={float(rs.min()):.3f}/{float(rs.mean()):.3f}/{float(rs.max()):.3f}  "
                              f"col-sum (in)  min/mean/max={float(cs.min()):.3f}/{float(cs.mean()):.3f}/{float(cs.max()):.3f}")
                else:
                    print(f"[verify] [{root_name}] {mat:>6s}  no 'scatter_matrix' dataset in scatter_data")
                if mm is not None:
                    z = mm[...]
                    print(f"[verify] [{root_name}] {mat:>6s}  multiplicity_matrix shape={z.shape} min/max={float(z.min()):.3f}/{float(z.max()):.3f}")

def build_mgxs_with_bounce_cap(max_bounce: int, out_path: str):
    """Create an MGXS enforcing ≤K scatters via staged sub-bins; export with correct engine ordering and shape."""
    import openmc, openmc.mgxs as mgxs
    K = int(max_bounce)
    G = 3
    Gtot = G*(K+1)

    # Expanded edges (ASC): subdivide each real group into K+1 bins
    edges = []
    for g in range(G):
        lo, hi = float(E_REAL_ASC[g]), float(E_REAL_ASC[g+1])
        sub = np.linspace(lo, hi, K+2)      # K+1 bins
        if g > 0:
            sub = sub[1:]                   # drop duplicate boundary
        edges.append(sub)
    E_edges = np.concatenate(edges)         # ascending in energy, as usual for filters
    eg  = mgxs.EnergyGroups(E_edges)
    lib = openmc.MGXSLibrary(eg)

    # Convert HML → ASC arrays (pure energy sense) for convenience
    SIGMA_T_ASC = SIGMA_T_HML[:, ::-1].copy()
    ALBEDO_ASC  = ALBEDO_HML[:,  ::-1].copy()
    PHASE_ASC   = np.zeros_like(PHASE_P_HML)
    for m in range(PHASE_P_HML.shape[0]):
        PHASE_ASC[m] = PHASE_P_HML[m, ::-1, ::-1]
    PHASE_ASC /= np.maximum(PHASE_ASC.sum(axis=-1, keepdims=True), 1.0)

    # -------- Engine group ordering helper --------
    # OpenMC MG expects group indices increasing from fast→thermal (decreasing energy).
    # Our real groups are indexed g=0..G-1 in ASC energy. Each is subdivided into K+1 bins (also ASC).
    # Map an (ASC g, k) to *engine* index j such that j=0 is highest energy bin overall:
    def engine_idx(g: int, k: int) -> int:
        return (G - 1 - g) * (K + 1) + (K - k)

    mat_names = []

    def add_mat(name: str, mat_row: int):
        xs = openmc.XSdata(name=name, energy_groups=eg, num_delayed_groups=0)
        xs.order = 0
        xs.scattering_format = 'legendre'
        xs.representation = 'isotropic'

        total_eng      = np.zeros(Gtot)
        absorption_eng = np.zeros(Gtot)
        S3             = np.zeros((Gtot, Gtot, 1))   # rows=outgoing, cols=incoming, ℓ=0
        scatter_per_in = np.zeros(Gtot)

        for g_in in range(G):                   # ASC energy sense
            st   = float(SIGMA_T_ASC[mat_row, g_in])
            alb  = float(ALBEDO_ASC[mat_row, g_in])
            phase = PHASE_ASC[mat_row, g_in]    # distribution over ASC outgoing groups
            for k in range(K+1):
                col = engine_idx(g_in, k)       # incoming column in engine ordering
                total_eng[col] = st
                if k < K:
                    absorption_eng[col] = st*(1.0 - alb)
                    # Scatter "advances stage": k -> k+1, and changes (or not) energy group by phase
                    for g_out in range(G):
                        row = engine_idx(g_out, k+1)       # outgoing row in engine ordering
                        val = st * alb * float(phase[g_out])
                        # OpenMC v0.13.x: rows=outgoing, cols=incoming
                        S3[row, col, 0] = val
                    scatter_per_in[col] = st * alb
                else:
                    # Terminal stage is purely absorbing to cap at ≤K scatters
                    absorption_eng[col] = st

        defect = total_eng - (absorption_eng + scatter_per_in)
        print(f"[xs:{name}] max|defect|={np.max(np.abs(defect)):.2e}  "
              f"scatter-out/total mean={float(np.mean(np.where(total_eng>0, scatter_per_in/total_eng, 0.0))):.3f} "
              f"min={float(np.min(np.where(total_eng>0, scatter_per_in/total_eng, 0.0))):.3f} "
              f"max={float(np.max(np.where(total_eng>0, scatter_per_in/total_eng, 0.0))):.3f}")

        xs.set_total(total_eng,      temperature=294.0)
        xs.set_absorption(absorption_eng, temperature=294.0)
        # *** NECESSARY CHANGE: transpose to [g_in, g_out, ℓ] ***
        xs.set_scatter_matrix(S3.transpose(1, 0, 2),    temperature=294.0)  # 3-D dataset for P0 Legendre

        # Multiplicity (set to 1 everywhere; OpenMC will warn and set to 1 if absent)
        try:
            xs.set_multiplicity_matrix(np.ones((Gtot, Gtot), dtype=float), temperature=294.0)
        except AttributeError:
            pass

        xs.set_chi(np.zeros(Gtot),          temperature=294.0)
        xs.set_nu_fission(np.zeros(Gtot),   temperature=294.0)
        xs.set_fission(np.zeros(Gtot),      temperature=294.0)

        lib.add_xsdata(xs)
        mat_names.append(name)

    add_mat("inner", 0)
    add_mat("outer", 1)

    # Void (tiny Σt, no scatter)
    import openmc
    EPS = 1e-8
    xs_v = openmc.XSdata(name="void", energy_groups=eg, num_delayed_groups=0)
    xs_v.order = 0
    xs_v.scattering_format = 'legendre'
    xs_v.representation = 'isotropic'
    xs_v.set_total(np.full(Gtot, EPS),      temperature=294.0)
    xs_v.set_absorption(np.full(Gtot, EPS), temperature=294.0)
    # *** NECESSARY CHANGE (consistent orientation): transpose even though it's zeros ***
    xs_v.set_scatter_matrix(np.zeros((Gtot, Gtot, 1)).transpose(1, 0, 2), temperature=294.0)
    try:
        xs_v.set_multiplicity_matrix(np.ones((Gtot, Gtot), dtype=float), temperature=294.0)
    except AttributeError:
        pass
    xs_v.set_chi(np.zeros(Gtot),        temperature=294.0)
    xs_v.set_nu_fission(np.zeros(Gtot), temperature=294.0)
    xs_v.set_fission(np.zeros(Gtot),    temperature=294.0)
    lib.add_xsdata(xs_v)
    mat_names.append("void")

    # Export
    if hasattr(lib, "build_library"):
        lib.build_library()

    xs_abs = os.path.abspath(out_path)
    lib.export_to_hdf5(xs_abs)

    # Verify what the engine will actually read
    _verify_scatter_in_hdf5(xs_abs)
    return xs_abs, E_edges

# ---------- Tally helpers ----------
def _energy_bins_to_edges(bins):
    b = np.asarray(bins)
    if b.ndim == 1:
        return b
    if b.ndim == 2 and b.shape[1] == 2:
        edges = np.empty(b.shape[0] + 1, dtype=float)
        edges[0] = b[0, 0]
        edges[1:] = b[:, 1]
        return edges
    return b.ravel()

def _dump_tallies(sp):
    print("[discover] Available tallies in statepoint:")
    for t in sp.tallies.values():
        filters = [type(f).__name__ for f in t.filters]
        extra = ""
        for f in t.filters:
            if hasattr(f, "bins"):
                try:
                    extra = f"  {type(f).__name__} bins shape={np.asarray(f.bins).shape}"
                except Exception:
                    pass
        print(f"  id={t.id:4d}  name={t.name!r}  scores={t.scores}  estimator={t.estimator}  filters={filters}{extra}")

def _discover_tally(sp, intended_name: str, want_edges: np.ndarray | None = None, want_score: str | None = None):
    _dump_tallies(sp)
    try:
        T = sp.get_tally(name=intended_name)
        print(f"[discover] Using tally id={T.id} (name={T.name!r}) via get_tally(name=...) for {intended_name}")
        return T
    except Exception:
        pass
    for t in sp.tallies.values():
        if t.name == intended_name:
            print(f"[discover] Using tally id={t.id} (name={t.name!r}) by exact name match")
            return t
    if want_edges is not None:
        for t in sp.tallies.values():
            ef = None
            for f in t.filters:
                if f.__class__.__name__ == "EnergyFilter":
                    ef = f; break
            if ef is None:
                continue
            try:
                have_edges = _energy_bins_to_edges(ef.bins)
                if np.allclose(have_edges, want_edges):
                    if (want_score is None) or (want_score in t.scores):
                        print(f"[discover] Using tally id={t.id} (name={t.name!r}) by EnergyFilter edges match")
                        return t
            except Exception:
                continue
    raise LookupError(f"Unable to discover tally {intended_name!r}")

def _flatten_tally_mean(T):
    m = T.mean
    if m.ndim == 1:
        return m
    if m.ndim == 2 and m.shape[1] == 1:
        return m[:, 0]
    if m.ndim == 3 and m.shape[-2:] == (1, 1):
        return m[:, 0, 0]
    return m.ravel()

def _pairs_from_energy_filter(T):
    for f in T.filters:
        if f.__class__.__name__ == "EnergyFilter":
            return np.asarray(f.bins)
    return None

def _build_pair_to_gk_map(E_edges_master: np.ndarray, G: int, K: int):
    # Map exact (lo,hi) pairs back to (g,k) in ASC energy sense for collapsing
    pairs = []
    start = 0
    for g in range(G):
        for k in range(K+1):
            lo = float(E_edges_master[start + k])
            hi = float(E_edges_master[start + k + 1])
            lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
            pairs.append(((lo2, hi2), (g, k)))
        start += (K+1)
    def _key(p):
        lo, hi = p
        return (round(lo, 12), round(hi, 12))
    d = {_key(x): y for x, y in pairs}
    return d

def collapse_flux_and_rates_from_pairs(values: np.ndarray,
                                       bin_pairs: np.ndarray,
                                       E_edges_master: np.ndarray,
                                       G: int,
                                       K: int):
    assert bin_pairs.ndim == 2 and bin_pairs.shape[1] == 2
    assert values.size == bin_pairs.shape[0]
    map_pair = _build_pair_to_gk_map(E_edges_master, G, K)

    flux_per_g = np.zeros(G)
    per_stage  = np.zeros(K+1)

    missing = 0
    for v, (a, b) in zip(values, bin_pairs):
        lo, hi = (float(a), float(b))
        if hi < lo:
            lo, hi = hi, lo
        key = (round(lo, 12), round(hi, 12))
        gk = map_pair.get(key, None)
        if gk is None:
            missing += 1
            continue
        g, k = gk
        flux_per_g[g] += float(v)
        per_stage[k]  += float(v)

    if missing:
        print(f"[warn] {missing} EnergyFilter bins could not be mapped to (g,k) by edges match; they were ignored.")

    flux_ASC = flux_per_g
    flux_HML = flux_ASC[::-1].copy()
    return flux_ASC, flux_HML, per_stage

# =========================
# Runner
# =========================
def run_openmc_case(seed: int, outdir: str):
    import openmc

    os.makedirs(outdir, exist_ok=True)
    xs_abs, E_edges = build_mgxs_with_bounce_cap(MAX_BOUNCE, out_path=os.path.join(outdir, "mgxs.h5"))

    # ---------- Materials ----------
    def macr(name):
        m = openmc.Material(name=name)
        m.add_macroscopic(openmc.Macroscopic(name))
        m.set_density("macro", 1.0)
        try: m.temperature = 294.0
        except Exception: pass
        return m
    m_inner = macr("inner")
    m_outer = macr("outer")
    m_void  = macr("void")

    mats = openmc.Materials([m_inner, m_outer, m_void])
    mats.cross_sections = xs_abs  # embed MGXS path

    # ---------- Geometry ----------
    s_inner = openmc.Sphere(r=R_INNER)
    s_outer = openmc.Sphere(r=R_OUTER)
    s_bound = openmc.Sphere(r=R_BOUND, boundary_type="vacuum")

    reg_inner = -s_inner
    reg_shell = +s_inner & -s_outer
    reg_void  = +s_outer & -s_bound

    c_inner = openmc.Cell(region=reg_inner, fill=m_inner); c_inner.id = 1001
    c_shell = openmc.Cell(region=reg_shell, fill=m_outer); c_shell.id = 1002
    c_void  = openmc.Cell(region=reg_void,  fill=m_void ); c_void.id  = 1003

    geom = openmc.Geometry(openmc.Universe(cells=[c_inner, c_shell, c_void]))

    # ---------- Source ----------
    d = np.array(SRC_DIR, float)
    n = float(np.linalg.norm(d))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("SRC_DIR must be a nonzero vector.")
    d = (d / n).astype(float)

    try:
        angle = openmc.stats.Monodirectional(reference_uvw=(float(d[0]), float(d[1]), float(d[2])))
    except TypeError:
        try:
            angle = openmc.stats.Monodirectional((float(d[0]), float(d[1]), float(d[2])))
        except Exception:
            angle = openmc.stats.Isotropic()

    try:
        src = openmc.IndependentSource()
        src.space  = openmc.stats.Point(SRC_POS)
        src.angle  = angle
        src.energy = openmc.stats.Discrete([float(SRC_E_EV)], [1.0])   # eV
        src.strength = 1.0
        src_list_required = True
    except AttributeError:
        src = openmc.Source()
        src.space  = openmc.stats.Point(SRC_POS)
        src.angle  = angle
        src.energy = openmc.stats.Discrete([float(SRC_E_EV)], [1.0])   # eV
        src.strength = 1.0
        src_list_required = False

    # ---------- Settings ----------
    sets = openmc.Settings()
    sets.run_mode  = "fixed source"
    sets.particles = int(np.ceil(PARTICLES / max(BATCHES, 1)))
    sets.batches   = int(BATCHES)
    sets.inactive  = 0
    sets.seed      = int(seed)
    sets.energy_mode = "multi-group"
    sets.max_order   = 0

    if src_list_required: sets.source = [src]
    else:                 sets.source = src

    sets.max_lost_particles     = max(int(PARTICLES), 1000)
    sets.rel_max_lost_particles = 0.99
    sets.statepoint = {"batches": [BATCHES]}

    # ---------- Tallies ----------
    ef = openmc.EnergyFilter(E_edges)
    cf = openmc.CellFilter([c_inner, c_shell, c_void])

    # Track-length tallies (flux, total, absorption)
    T_flux_total = openmc.Tally(name="Flux_expanded_bins");             T_flux_total.scores       = ["flux"];        T_flux_total.filters       = [ef]
    T_total_rate = openmc.Tally(name="Total_expanded_bins");            T_total_rate.scores       = ["total"];       T_total_rate.filters       = [ef]
    T_rr_absorb  = openmc.Tally(name="Absorption_expanded_bins");       T_rr_absorb.scores        = ["absorption"]; T_rr_absorb.filters        = [ef]

    # Region-resolved flux (optional)
    T_flux_by_cell = openmc.Tally(name="Flux_expanded_bins_by_cell");   T_flux_by_cell.scores     = ["flux"];       T_flux_by_cell.filters     = [cf, ef]

    # Analog scatter collisions to prove sampling actually happens
    T_scatter_analog = openmc.Tally(name="Scatter_analog_bins");        T_scatter_analog.scores   = ["scatter"];    T_scatter_analog.filters   = [ef]; T_scatter_analog.estimator = "analog"

    tallies = openmc.Tallies([T_flux_total, T_total_rate, T_rr_absorb, T_flux_by_cell, T_scatter_analog])
    model = openmc.Model(materials=mats, geometry=geom, settings=sets, tallies=tallies)

    # ---------- Export & Run ----------
    run_dir = os.path.join(outdir, f"run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        model.export_to_xml(directory=run_dir)
    except TypeError:
        here = os.getcwd()
        os.chdir(run_dir)
        try:
            mats.export_to_xml(); geom.export_to_xml(); sets.export_to_xml(); tallies.export_to_xml()
        finally:
            os.chdir(here)

    env = os.environ.copy()
    env["OPENMC_MG_CROSS_SECTIONS"] = xs_abs
    env["OPENMC_CROSS_SECTIONS"]    = xs_abs

    print("[config]")
    print(f"  Particles total         : {PARTICLES}")
    print(f"  Batches                 : {BATCHES}  (particles/batch = {sets.particles})")
    print(f"  K (max scatters)        : {MAX_BOUNCE}")
    print(f"  Real groups (ASC) edges : {E_REAL_ASC.tolist()}")
    print(f"[beam] src={tuple(map(float, SRC_POS))}, dir={tuple(map(float, d))}, E={float(SRC_E_EV)} eV")

    exe = shutil.which("openmc") or "/usr/local/bin/openmc"
    subprocess.run([exe], cwd=run_dir, check=True, env=env)

    # ---------- Read results & collapse ----------
    sp_path = next((os.path.join(run_dir, fn) for fn in os.listdir(run_dir)
                    if fn.startswith("statepoint.") and fn.endswith(".h5")), None)
    if sp_path is None:
        raise RuntimeError("Statepoint not found.")

    import openmc
    sp = openmc.StatePoint(sp_path)

    T_flux_total  = _discover_tally(sp, "Flux_expanded_bins",   want_edges=E_edges, want_score="flux")
    T_total_rate  = _discover_tally(sp, "Total_expanded_bins",  want_edges=E_edges, want_score="total")
    T_rr_absorb   = _discover_tally(sp, "Absorption_expanded_bins", want_edges=E_edges, want_score="absorption")
    T_scatt_anlg  = _discover_tally(sp, "Scatter_analog_bins",  want_edges=E_edges, want_score="scatter")

    tf_mean = _flatten_tally_mean(T_flux_total)
    _assert_finite_nonzero(tf_mean, "expanded-bin flux tally")
    bins_pairs = _pairs_from_energy_filter(T_flux_total)

    Gtot = tf_mean.size
    if Gtot % 3 != 0:
        raise RuntimeError("Expanded bin count not divisible by 3.")
    K_infer = Gtot // 3 - 1
    print(f"[shape] global flux mean flat size={tf_mean.size}, expected Gtot={3*(K_infer+1)}")

    rr_total = _flatten_tally_mean(T_total_rate)
    rr_abs   = _flatten_tally_mean(T_rr_absorb)
    rr_scat_from_diff = np.clip(rr_total - rr_abs, 0.0, None)
    rr_scat_analog = _flatten_tally_mean(T_scatt_anlg)

    flux_ASC, flux_HML, stage_totals_flux        = collapse_flux_and_rates_from_pairs(tf_mean,           bins_pairs, E_edges, G=3, K=K_infer)
    scat_ASC_diff, _, stage_totals_scat_diff     = collapse_flux_and_rates_from_pairs(rr_scat_from_diff, bins_pairs, E_edges, G=3, K=K_infer)
    scat_ASC_analog, _, stage_totals_scat_analog = collapse_flux_and_rates_from_pairs(rr_scat_analog,    bins_pairs, E_edges, G=3, K=K_infer)
    abs_ASC,  _, stage_totals_abs                = collapse_flux_and_rates_from_pairs(rr_abs,            bins_pairs, E_edges, G=3, K=K_infer)

    print("\n=== SUMMARY (collapsed to real groups) ===")
    print(f"Groups (ASC low→high E): {E_REAL_ASC.tolist()}")
    print(f"Expanded bins per group : {K_infer+1}  (stages 0..{K_infer})\n")

    def pct(v):
        s = float(np.sum(v)) or 1.0
        return [f"{(100*x/s):.1f}%" for x in v]

    print("[Global track-length flux]")
    print(f"  ASC : {flux_ASC.tolist()}")
    print(f"  HML : {flux_HML.tolist()}")
    print(f"  Share ASC : {pct(flux_ASC)}\n")

    print("[Global reaction rates (track-length estimator)]")
    print(f"  Total ASC      : {collapse_flux_and_rates_from_pairs(rr_total, bins_pairs, E_edges, 3, K_infer)[0].tolist()}")
    print(f"  Absorption ASC : {abs_ASC.tolist()}")
    print(f"  Scatter ASC TL : {scat_ASC_diff.tolist()}  (computed as total-abs)\n")

    print("[Per-stage global totals]")
    print(f"  flux             stages 0..{K_infer} : {stage_totals_flux.tolist()}")
    print(f"  scatter (tot-abs)stages 0..{K_infer} : {stage_totals_scat_diff.tolist()}")
    print(f"  scatter analog   stages 0..{K_infer} : {stage_totals_scat_analog.tolist()}")
    print(f"  absorption       stages 0..{K_infer} : {stage_totals_abs.tolist()}\n")

    if np.allclose(stage_totals_scat_analog.sum(), 0.0):
        print("[warn] Analog scatter collisions are zero — unexpected now. Double-check engine group ordering and scatter_matrix population.")
    if np.allclose(stage_totals_flux[1:].sum(), 0.0):
        print("[warn] No flux reached stages ≥1 — indicates no scattering reached next sub-bins.]")

def main():
    if not (R_INNER > 0 and R_OUTER > R_INNER and R_BOUND > R_OUTER):
        raise ValueError("Require 0 < R_INNER < R_OUTER < R_BOUND.")
    if np.linalg.norm(np.asarray(SRC_POS)) <= R_OUTER:
        raise ValueError("SRC_POS must be outside the outer sphere.")

    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    run_openmc_case(SEED, RESULT_DIR)

if __name__ == "__main__":
    main()
