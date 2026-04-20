"""
render_geo_voxels.py
====================
Load an OBJ geometry from MEDIA_DIR as the inner material (material 0),
bound it with a large sphere of radius 6 as the outer material (material 1),
and render an N^3 voxel flux map using the track-length estimator.

CSG layout (same convention as tests.py):
  shape0 = inner OBJ  (CSGLeaf 0)
  shape1 = sphere r6  (CSGLeaf 1)
  m0 = intersection(shape0, shape1)   -> inner OBJ region
  m1 = difference(shape1, shape0)     -> shell between OBJ and sphere r6

Usage (command line):
  python render_geo_voxels.py

Or import and call render_flux_voxels() directly.
"""

import os
import area_tally                          # we patch MAX_BOUNCE here
from area_tally import *
from csg import save_grid_data
from constant import DATA_DIR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import imageio

# ---------------------------------------------------------------------------
# Path to the OBJ folder on the media drive
# ---------------------------------------------------------------------------
MEDIA_DIR = "/media/xideng/973f0610-62ab-4e4e-9192-dde2782393be1/data"
GEO_OBJ_DIR = os.path.join(MEDIA_DIR, "geo_norm_obj")

# sphere_smooth.obj scaled ×6 is used as the outer bounding sphere (r=6)
OUTER_SPHERE_OBJ = os.path.join(DATA_DIR, "scene", "sphere_smooth.obj")


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def load_geo_scene(obj_filename,
                   sig_t_inner=0.5,  albedo_inner=0.8,
                   sig_t_shell=0.1,  albedo_shell=0.9,
                   ad=True):
    """
    Build a two-material CSG scene:
      material 0  – inner OBJ geometry
      material 1  – shell between the OBJ and the r=6 bounding sphere

    Parameters
    ----------
    obj_filename  : str  – full path to the inner OBJ file
    sig_t_inner   : float or FloatD – total cross section of inner material
    albedo_inner  : float           – single-scatter albedo of inner material
    sig_t_shell   : float or FloatD – total cross section of shell material
    albedo_shell  : float           – single-scatter albedo of shell material
    ad            : bool            – enable gradients on vertex positions

    Returns
    -------
    scene, scm, vertices_list, faces_list
    """
    scene_dict = {
        'type': 'scene',
        # shape index 0 → inner OBJ geometry
        'A': {
            'id': 'A',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]),
            'filename': obj_filename,
            'face_normals': True,   # recompute from faces; ignores invalid vertex normals
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.2, 0.25, 0.7]},
            },
        },
        # shape index 1 → bounding sphere radius 6
        'B': {
            'id': 'B',
            'type': 'obj',
            'to_world': mi.ScalarTransform4f().scale([6.0, 6.0, 6.0]),
            'filename': OUTER_SPHERE_OBJ,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.23, 0.25, 0.7]},
            },
        },
    }
    scene = mi.load_dict(scene_dict)

    shape0 = CSGLeaf(0)   # inner OBJ
    shape1 = CSGLeaf(1)   # bounding sphere r=6

    # material 0: region strictly inside the OBJ  (OBJ ∩ sphere6 ≈ OBJ interior)
    m0 = CSGNode("intersection", shape0, shape1)
    # material 1: shell between OBJ surface and sphere r=6  (sphere6 \ OBJ)
    m1 = CSGNode("difference",   shape1, shape0)

    # single-group cross sections  – layout [1, num_materials] = [[mat0, mat1]]
    sig_t_inner_d = FloatD(sig_t_inner) if not isinstance(sig_t_inner, FloatD) else sig_t_inner
    sig_t_shell_d = FloatD(sig_t_shell) if not isinstance(sig_t_shell, FloatD) else sig_t_shell

    scm = SceneMaterial([m0, m1],
                        [[sig_t_inner_d, sig_t_shell_d]],
                        [[float(albedo_inner), float(albedo_shell)]],
                        num_geo=2,
                        energy_groups=1)

    scm.set_material_sigma(TensorXfD([[sig_t_inner_d, sig_t_shell_d]]))
    scm.set_material_ald(TensorXfD([[FloatD(albedo_inner), FloatD(albedo_shell)]]))
    # isotropic, single-group: phase[mat, inc_g, out_g] = [[1,1]] per material
    scm.set_phase_function(TensorXf([[[1.0, 1.0]]]))

    # ── assemble vertex / face lists ─────────────────────────────────────────
    params = mi.traverse(scene)
    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])
    Fa = dr.unravel(mi.Vector3i, mi.Int(params['A.faces']))
    Fb = dr.unravel(mi.Vector3i, mi.Int(params['B.faces']))

    params['A.vertex_positions'] = dr.ravel(Va)
    params['B.vertex_positions'] = dr.ravel(Vb)
    if ad:
        dr.enable_grad(params['A.vertex_positions'])
        dr.enable_grad(params['B.vertex_positions'])
    params.update()

    Va = dr.unravel(mi.Point3f, params['A.vertex_positions'])
    Vb = dr.unravel(mi.Point3f, params['B.vertex_positions'])

    # convention from tests.py: vertices_list = [Vb, Va]  (outer first)
    return scene, scm, [Vb, Va], [Fb, Fa]


# ---------------------------------------------------------------------------
# Source sampler
# ---------------------------------------------------------------------------

def sample_isotropic_source(rng, num_neutrons):
    """Planar source on x=-1 face, emitting uniformly into the +x hemisphere."""
    # origin: x=-1, y and z uniform in [-1, 1]
    ray_orig   = dr.zeros(mi.Point3f, num_neutrons)
    ray_orig.x -= 2.0
    ray_orig.y  = sample_float_32(rng) * 2.0 - 1.0
    ray_orig.z  = sample_float_32(rng) * 2.0 - 1.0

    # direction: uniform hemisphere toward +x
    cos_theta = sample_float_32(rng)                          # in [0, 1]
    sin_theta = dr.sqrt(dr.maximum(0.0, 1.0 - cos_theta * cos_theta))
    phi       = sample_float_32(rng) * 2.0 * dr.pi
    ray_vec   = mi.Vector3f(cos_theta,
                            sin_theta * dr.cos(phi),
                            sin_theta * dr.sin(phi))

    return mi.Ray3f(ray_orig, ray_vec)


# ---------------------------------------------------------------------------
# Core voxel-flux renderer
# ---------------------------------------------------------------------------

def render_flux_voxels(obj_filename,
                       N=64,
                       max_bounce=2,
                       sig_t_inner=0.5,
                       albedo_inner=0.8,
                       sig_t_shell=0.1,
                       albedo_shell=0.9,
                       num_neutrons=100000,
                       seed=1994,
                       boundingbox=None,
                       bounce_id=-1,
                       source_power=1.0,
                       num_passes=1):
    """
    Render an N×N×N voxel flux map using the track-length estimator.

    Parameters
    ----------
    obj_filename  : str   – path to the inner geometry OBJ (inside GEO_OBJ_DIR)
    N             : int   – voxel grid resolution along each axis
    max_bounce    : int   – maximum number of scattering events (M)
    sig_t_inner   : float – total cross section of the inner material
    albedo_inner  : float – single-scatter albedo of the inner material
    sig_t_shell   : float – total cross section of the shell material
    albedo_shell  : float – single-scatter albedo of the shell material
    num_neutrons  : int   – number of source particles
    seed          : int   – RNG seed
    boundingbox   : list  – [[xmin,ymin,zmin],[xmax,ymax,zmax]] as mi.Vector3f;
                            defaults to a cube slightly larger than sphere r=6
    bounce_id     : int   – which bounce to tally (-1 = all bounces summed)

    Returns
    -------
    voxels : FloatD drjit array of length N^3  (flux per voxel per source neutron)
    """
    # ── resolve full path ──────────────────────────────────────────────────
    if not os.path.isabs(obj_filename):
        obj_filename = os.path.join(GEO_OBJ_DIR, obj_filename)

    # ── default bounding box: just outside the r=6 sphere ─────────────────
    if boundingbox is None:
        r_bb = 2.0
        boundingbox = [mi.Vector3f(-r_bb, -r_bb, -r_bb),
                       mi.Vector3f( r_bb,  r_bb,  r_bb)]

    resolution = mi.Vector3i(N, N, N)

    # ── temporarily override the module-level MAX_BOUNCE ──────────────────
    original_max_bounce = area_tally.MAX_BOUNCE
    area_tally.MAX_BOUNCE = max_bounce

    # ── build scene ───────────────────────────────────────────────────────
    scene, scm, verts, faces = load_geo_scene(
        obj_filename,
        sig_t_inner=sig_t_inner,  albedo_inner=albedo_inner,
        sig_t_shell=sig_t_shell,  albedo_shell=albedo_shell,
        ad=False)

    # ── multi-pass accumulation ───────────────────────────────────────────
    accumulated = None

    for pass_idx in range(num_passes):
        pass_seed = seed + pass_idx
        dr.make_opaque(pass_seed)
        rng = mi.PCG32(size=num_neutrons, initstate=pass_seed)
        dr.make_opaque(rng)

        ray_current = sample_isotropic_source(rng, num_neutrons)
        dr.make_opaque(ray_current)

        sceneinfo = SceneInfo(scene, rng, scm, verts, faces)

        # ── render with track-length beam collection ──────────────────────
        beam_list = []
        Etot = render_nuetron_in_csg_shape_energy_dependent(
            sceneinfo, ray_current,
            reparam=True,
            beams=beam_list,
            save_beam=True)

        # ── accumulate beams into voxels (track-length estimator) ─────────
        voxels = accumulate_photon_beams_faster(beam_list, resolution,
                                                boundingbox, bounce_id)

        if accumulated is None:
            accumulated = voxels
        else:
            accumulated = accumulated + voxels

    # ── restore original MAX_BOUNCE ────────────────────────────────────────
    area_tally.MAX_BOUNCE = original_max_bounce

    return accumulated / (num_passes * num_neutrons) * source_power


# ---------------------------------------------------------------------------
# Convenience: render and save to .npy
# ---------------------------------------------------------------------------

def render_and_save(obj_filename,
                    out_prefix="flux",
                    N=64,
                    max_bounce=5,
                    sig_t_inner=0.5,
                    albedo_inner=0.8,
                    sig_t_shell=0.1,
                    albedo_shell=0.9,
                    num_neutrons=100000,
                    seed=1994,
                    source_power=1.0,
                    num_passes=1):
    """
    Render voxel flux and save the result together with scene metadata.

    Saves
    -----
    {out_prefix}_flux.npy  – flat float32 array of length N^3
    """
    r_bb = 2.0
    boundingbox = [mi.Vector3f(-r_bb, -r_bb, -r_bb),
                   mi.Vector3f( r_bb,  r_bb,  r_bb)]
    resolution = mi.Vector3i(N, N, N)

    print(f"Rendering {N}^3 voxels, max_bounce={max_bounce}, "
          f"N_neutrons={num_neutrons}, num_passes={num_passes}")
    print(f"  inner obj : {obj_filename}")
    print(f"  sig_t     : inner={sig_t_inner}  shell={sig_t_shell}")
    print(f"  albedo    : inner={albedo_inner}  shell={albedo_shell}")

    voxels = render_flux_voxels(
        obj_filename,
        N=N,
        max_bounce=max_bounce,
        sig_t_inner=sig_t_inner,
        albedo_inner=albedo_inner,
        sig_t_shell=sig_t_shell,
        albedo_shell=albedo_shell,
        num_neutrons=num_neutrons,
        seed=seed,
        boundingbox=boundingbox,
        source_power=source_power,
        num_passes=num_passes)

    out_path = f"{out_prefix}_flux.npy"
    save_grid_data(voxels.numpy(), resolution, boundingbox, out_path, 'wb')
    print(f"Saved {out_path}  (shape {N}^3, total flux={float(dr.sum(voxels).numpy()):.4f})")
    return voxels


# ---------------------------------------------------------------------------
# Multi-bounce sweep: render once per bounce count 1..max_bounce
# ---------------------------------------------------------------------------

def render_bounce_sweep(obj_filename,
                        out_prefix="flux_bounce",
                        N=64,
                        max_bounce=5,
                        sig_t_inner=0.5,
                        albedo_inner=0.8,
                        sig_t_shell=0.1,
                        albedo_shell=0.9,
                        num_neutrons=100000,
                        seed=1994):
    """
    Render a separate voxel map for each bounce count b in 1..max_bounce,
    tallying only the b-th scattering contribution (bounce_id=b-1).

    Saves one .npy per bounce:
        {out_prefix}_b{b}.npy
    Also saves the accumulated total (all bounces summed):
        {out_prefix}_total.npy
    """
    r_bb = 2.0
    boundingbox = [mi.Vector3f(-r_bb, -r_bb, -r_bb),
                   mi.Vector3f( r_bb,  r_bb,  r_bb)]
    resolution = mi.Vector3i(N, N, N)

    total = None

    for b in range(1, max_bounce + 1):
        print(f"\n── bounce {b}/{max_bounce} ──────────────────────────────")
        voxels = render_flux_voxels(
            obj_filename,
            N=N,
            max_bounce=b,           # only simulate up to bounce b
            sig_t_inner=sig_t_inner,
            albedo_inner=albedo_inner,
            sig_t_shell=sig_t_shell,
            albedo_shell=albedo_shell,
            num_neutrons=num_neutrons,
            seed=seed,
            boundingbox=boundingbox,
            bounce_id=b - 1)        # tally only the b-th bounce segment

        arr = voxels.numpy()
        out_path = f"{out_prefix}_b{b}.npy"
        save_grid_data(arr, resolution, boundingbox, out_path, 'wb')
        print(f"  saved {out_path}  total flux = {arr.sum():.4f}")

        if total is None:
            total = arr.copy()
        else:
            total += arr

        del voxels

    out_total = f"{out_prefix}_total.npy"
    save_grid_data(total, resolution, boundingbox, out_total, 'wb')
    print(f"\nSaved summed total: {out_total}")
    return total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _load_volume(voxels_or_path, N):
    """
    Return a (N, N, N) float32 numpy array from either:
      - a drjit FloatD / numpy flat array  (length N^3)
      - a path to a *_numpy.npy file saved by save_grid_data  (shape already (N,N,N))
      - a path to a raw flux .npy file  (flat, length N^3)
    """
    if isinstance(voxels_or_path, str):
        arr = np.load(voxels_or_path)
    elif hasattr(voxels_or_path, 'numpy'):   # drjit array
        arr = voxels_or_path.numpy()
    else:
        arr = np.asarray(voxels_or_path)

    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        dim = int(round(arr.shape[0] ** (1/3)))
        arr = arr.reshape(dim, dim, dim)
    return arr                               # shape (N, N, N) = (Z, Y, X)


def _best_slice_idx(volume, axis):
    """
    Return the index along `axis` whose 2-D slab has the largest mean flux.
    This finds the most 'active' slice rather than blindly using the midpoint.
    """
    means = np.mean(volume, axis=tuple(a for a in range(3) if a != axis))
    return int(np.argmax(means))


def plot_flux_slice(voxels_or_path,
                    N=64,
                    boundingbox=None,
                    title="Flux",
                    cmap="plasma",
                    log_scale=False,
                    save_path=None,
                    show=True):
    """
    Plot the three orthogonal slices (XY, XZ, YZ) of a voxel flux field.
    For each axis the slice with the highest mean flux is chosen automatically
    so the plot always shows a "valid" (non-empty) cut through the geometry.

    Parameters
    ----------
    voxels_or_path : drjit array | numpy array | str
        Flat N^3 array, already-shaped (N,N,N) array, or path to a .npy file
        saved by save_grid_data (the *_numpy.npy variant with shape (N,N,N)).
    N              : int   – grid resolution (used only if shape must be inferred)
    boundingbox    : list  – [[xmin,ymin,zmin],[xmax,ymax,zmax]] as mi.Vector3f
                             or plain floats; used for axis labels.
                             Defaults to [-7,7]^3.
    title          : str   – figure super-title
    cmap           : str   – matplotlib colormap
    log_scale      : bool  – apply log10(1+x) before plotting
    save_path      : str or None – if given, save figure to this path
    show           : bool  – call plt.show()

    Returns
    -------
    fig, axes  (matplotlib Figure and 1-D array of 3 Axes)
    """
    vol = _load_volume(voxels_or_path, N)   # (Z, Y, X)

    if log_scale:
        vol = vol / (1 + vol)
        # np.log10(0.00001 + vol)


    # ── world-space axis ticks ────────────────────────────────────────────
    if boundingbox is None:
        lo, hi = -2.0, 2.0
    else:
        lo = float(boundingbox[0][0]) if hasattr(boundingbox[0], '__getitem__') \
             else float(boundingbox[0])
        hi = float(boundingbox[1][0]) if hasattr(boundingbox[1], '__getitem__') \
             else float(boundingbox[1])

    extent = [lo, hi, lo, hi]   # same for all slices (cubic domain)

    # ── centre slice for each axis ───────────────────────────────────────
    # vol axes: 0=Z, 1=Y, 2=X
    iz = vol.shape[0] // 2
    iy = vol.shape[1] // 2
    ix = vol.shape[2] // 2

    slices = {
        f"XY  (z-slice {iz})": vol[iz, :, :],
        f"XZ  (y-slice {iy})": vol[:, iy, :],
        f"YZ  (x-slice {ix})": vol[:, :, ix],
    }

    # ── shared color scale from data ──────────────────────────────────────
    vmin = min(s.min() for s in slices.values())
    vmax = max(s.max() for s in slices.values())
    if vmax == vmin:
        vmax = vmin + 1e-8

    # ── save raw EXR slices ───────────────────────────────────────────────
    if save_path:
        stem = save_path.rsplit('.', 1)[0]
        exr_names = [f"{stem}_xy.exr", f"{stem}_xz.exr", f"{stem}_yz.exr"]
        for exr_path, (_, sdata) in zip(exr_names, slices.items()):
            imageio.imwrite(exr_path, sdata.astype(np.float32))
            print(f"Saved EXR: {exr_path}")

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05],
                            wspace=0.25)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    cax  = fig.add_subplot(gs[3])

    label_scale = "log₁₀(1+flux)" if log_scale else "flux"
    axis_labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

    for ax, (slice_name, sdata), (xlabel, ylabel) in zip(
            axes, slices.items(), axis_labels):
        im = ax.imshow(sdata, origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=extent, aspect='equal')
        ax.set_title(slice_name, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.colorbar(im, cax=cax, label=label_scale)
    fig.suptitle(title, fontsize=12, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved slice plot: {save_path}")
    if show:
        plt.show()

    return fig, axes


def plot_bounce_sweep_slices(bounce_npy_prefix,
                              max_bounce,
                              N=64,
                              boundingbox=None,
                              cmap="plasma",
                              log_scale=False,
                              save_path=None,
                              show=True):
    """
    Load the per-bounce .npy files written by render_bounce_sweep() and show
    one mid-Z slice per bounce count side-by-side.

    Files expected:  {bounce_npy_prefix}_b{b}_numpy.npy  for b=1..max_bounce
    (the *_numpy.npy suffix is written automatically by save_grid_data)

    Parameters
    ----------
    bounce_npy_prefix : str  – prefix passed to render_bounce_sweep as out_prefix
    max_bounce        : int
    N, boundingbox, cmap, log_scale, save_path, show  – same as plot_flux_slice
    """
    if boundingbox is None:
        lo, hi = -2.0, 2.0
    else:
        lo = float(boundingbox[0][0]) if hasattr(boundingbox[0], '__getitem__') \
             else float(boundingbox[0])
        hi = float(boundingbox[1][0]) if hasattr(boundingbox[1], '__getitem__') \
             else float(boundingbox[1])
    extent = [lo, hi, lo, hi]

    volumes, titles = [], []
    for b in range(1, max_bounce + 1):
        path = f"{bounce_npy_prefix}_b{b}_numpy.npy"
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping bounce {b}")
            continue
        vol = _load_volume(path, N)
        if log_scale:
            vol = np.log10(1.0 + vol)
        iz = _best_slice_idx(vol, axis=0)
        volumes.append((vol[iz, :, :], iz))
        titles.append(f"bounce {b}  (z={iz})")

    if not volumes:
        print("No bounce files found.")
        return None, None

    # shared color scale
    vmin = min(s.min() for s, _ in volumes)
    vmax = max(s.max() for s, _ in volumes)
    if vmax == vmin:
        vmax = vmin + 1e-8

    n_plots = len(volumes)
    fig = plt.figure(figsize=(4 * n_plots + 0.6, 4.5))
    gs  = gridspec.GridSpec(1, n_plots + 1,
                            width_ratios=[1] * n_plots + [0.05],
                            wspace=0.25)
    axes = [fig.add_subplot(gs[i]) for i in range(n_plots)]
    cax  = fig.add_subplot(gs[n_plots])

    label_scale = "log₁₀(1+flux)" if log_scale else "flux"
    for ax, (sdata, _), t in zip(axes, volumes, titles):
        im = ax.imshow(sdata, origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=extent, aspect='equal')
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("X");  ax.set_ylabel("Y")

    fig.colorbar(im, cax=cax, label=label_scale)
    fig.suptitle("Per-bounce XY flux slices", fontsize=12, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved bounce sweep plot: {save_path}")
    if show:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Render N^3 voxel flux from OBJ geometry")
    parser.add_argument("--obj",           type=str,   default=None,   help="Path to inner OBJ file (default: first in GEO_OBJ_DIR)")
    parser.add_argument("--N",             type=int,   default=64,     help="Voxel grid resolution (default: 64)")
    parser.add_argument("--max_bounce",    type=int,   default=1,      help="Max scattering events (default: 5)")
    parser.add_argument("--sig_t_inner",   type=float, default=0.5,    help="Total cross section of inner material (default: 0.5)")
    parser.add_argument("--albedo_inner",  type=float, default=0.8,    help="Scattering albedo of inner material (default: 0.8)")
    parser.add_argument("--sig_t_shell",   type=float, default=0.1,    help="Total cross section of shell material (default: 0.1)")
    parser.add_argument("--albedo_shell",  type=float, default=0.9,    help="Scattering albedo of shell material (default: 0.9)")
    parser.add_argument("--num_neutrons",  type=int,   default=200000, help="Number of source neutrons (default: 200000)")
    parser.add_argument("--seed",          type=int,   default=1994,   help="RNG seed (default: 1994)")
    parser.add_argument("--out_prefix",    type=str,   default="geo_flux", help="Output file prefix (default: geo_flux)")
    parser.add_argument("--source_power",  type=float, default=1.0,      help="Source power multiplier (default: 1.0)")
    parser.add_argument("--num_passes",    type=int,   default=1,         help="Number of render passes to accumulate (default: 1)")
    parser.add_argument("--log_scale",     action="store_true",            help="Plot flux in log10(1+x) scale")
    args = parser.parse_args()

    # ── discover OBJ ──────────────────────────────────────────────────────
    if args.obj:
        obj_path = args.obj
    else:
        obj_files = sorted(glob.glob(os.path.join(GEO_OBJ_DIR, "*.obj")))
        if not obj_files:
            raise FileNotFoundError(
                f"No .obj files found in {GEO_OBJ_DIR}. "
                "Check that the media drive is mounted.")
        obj_path = obj_files[0]
    print(f"Using geometry: {obj_path}")

    # ── single render ──────────────────────────────────────────────────────
    render_and_save(
        obj_filename=obj_path,
        out_prefix=args.out_prefix,
        N=args.N,
        max_bounce=args.max_bounce,
        sig_t_inner=args.sig_t_inner,
        albedo_inner=args.albedo_inner,
        sig_t_shell=args.sig_t_shell,
        albedo_shell=args.albedo_shell,
        num_neutrons=args.num_neutrons,
        seed=args.seed,
        source_power=args.source_power,
        num_passes=args.num_passes)

    # ── per-bounce sweep ───────────────────────────────────────────────────
    # render_bounce_sweep(
    #     obj_filename=obj_path,
    #     out_prefix="geo_flux",
    #     N=64,
    #     max_bounce=5,
    #     sig_t_inner=0.5,
    #     albedo_inner=0.8,
    #     sig_t_shell=0.1,
    #     albedo_shell=0.9,
    #     num_neutrons=200000,
    #     seed=1994)

    # ── plot total flux (three orthogonal slices) ──────────────────────────
    plot_flux_slice(
        "geo_flux_flux_numpy.npy",
        N=64,
        title="Total flux – best orthogonal slices",
        cmap="plasma",
        log_scale=args.log_scale,
        save_path="geo_flux_slices.png",
        show=False)

    # ── plot per-bounce XY slices side-by-side ─────────────────────────────
    # plot_bounce_sweep_slices(
    #     bounce_npy_prefix="geo_flux",
    #     max_bounce=5,
    #     N=64,
    #     cmap="plasma",
    #     log_scale=False,
    #     save_path="geo_flux_bounce_sweep.png",
    #     show=False)
