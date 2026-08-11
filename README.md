# Inverse Neutron Transport (INT)

A differentiable Monte Carlo neutron transport simulator built on [Mitsuba](https://mitsuba.readthedocs.io/en/stable/index.html) / [dr.jit](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html), used to validate against [OpenMC](https://docs.openmc.org/), run sensitivity/uncertainty analysis, and optimize sensor and shielding geometry.

## Setup

```bash
pip install mitsuba drjit numpy scipy matplotlib gpytoolbox
```

OpenMC is only needed for the validation experiments and is easiest to run via Docker:

```bash
docker run --rm -it -v "${PWD}:/work" -w /work openmc/openmc:v0.13.2 python3 <script>.py
```

Before running anything, set the two paths in [constant.py](constant.py):

- `DATA_DIR` — repo root (contains `scene/`, the `.obj` geometry used by the CSG scenes)
- `TEMP_DIR` — scratch directory for optimizer output (meshes, `.npy`/`.png` dumps)

## Module map

| Module | Role |
|---|---|
| [csg.py](csg.py) | Core CSG/ray-scene intersection engine (Mitsuba-backed). Example scene setup in `test1()`. Currently the active/maintained engine module. |
| [area_tally.py](area_tally.py) | differentiable neutron transport simulation (track-length/collision tallies, energy-dependent transport). **Active development target** . |
| [simulator.py](simulator.py) | Stable version of the transport core (area_tally.py). `opt_sensor.py` still imports from it. |
| [tests.py](tests.py) | Test/experiment driver built on `area_tally.py` (single-group). |
| [tests_mg_pbox.py](tests_mg_pbox.py) | Multi-group experiment driver, incl. OpenMC-comparison scene loaders. |
| [constant.py](constant.py) | `DATA_DIR` / `TEMP_DIR` paths — edit these first. |

---

##  Sensitivity analysis (p-box / uncertainty quantification)

Compares a Monte-Carlo-sampled probability box against a gradient-based (delta-method) p-box, using the differentiable renderer's gradients — the point of having a differentiable simulator.

- [uncertainty.py](uncertainty.py) — single-parameter case. Imports `uncertainty_f` from [tests.py](tests.py) (`tests.py:1371`) as the forward+gradient function, samples parameter uncertainty (`mu`, `sigma`), and plots MC vs. gradient p-boxes to `uncertainty_geo.png`.
- [uncertainty_mg_pbox.py](uncertainty_mg_pbox.py) — multi-group, multi-parameter version, with and without cross-parameter covariance. Produces `uncertainty_mg_diag.png` (no covariance) and `uncertainty_mg_cov.png` (with covariance).

Run directly:

```bash
python uncertainty.py
python uncertainty_mg_pbox.py
```

To point either script at a different observable, edit/extend `uncertainty_f` in `tests.py`.

---

## Sensor optimization

[opt_sensor.py](opt_sensor.py) optimizes sensor mesh geometry (radius/shape) to maximize captured energy, using `simulator.py` for the transport core and `area_tally.py` for sampling helpers (`sample_dir_from_unit_ring`), plus [gpytoolbox](https://github.com/sgsellan/gpytoolbox)'s `remesh_botsch` for remeshing during shape optimization.

Key entry points (selected in the `__main__` block):

- `scan_radius()` — brute-force sweep over sensor radius, saves `sensor_opt_value.npy` / `sensor_opt_rs.npy`.
- `opt(iteration_count, key, nuetron_number, param_dict)` — gradient-based optimization of a single mesh parameter.
- `opt_mesh(...)` / `opt_mesh_volume_constraints(...)` — full mesh-vertex optimization with periodic remeshing, the latter with a volume constraint penalty.

```bash
python opt_sensor.py
```

Edit `constant.py`'s `TEMP_DIR` first — intermediate meshes and loss curves are written there.

---

## Shape optimization without CSG (`optimization.py`)

[optimization.py](optimization.py) optimizes shielding mesh geometry directly against a Mitsuba-rendered mesh scene, **without** going through the CSG scene graph in `csg.py` — it drives `mi.render`/`mi.traverse` on plain `.obj` shapes and reparameterizes the transport probabilities (scattering direction pdf `sample_direction_hg`/`hg`, distance sampling `sample_distance`, cross-section normalization `cross_section_nor`) imported from `area_tally.py`.

```bash
python optimization.py
```

The `__main__` block calls `opt("hetero.xml", 1250, "shield", remesh=100)` by default — a shield mesh optimization over 1250 iterations with periodic Botsch remeshing every 100 steps. Swap in `render_geo(...)` to re-render a saved sequence of optimization checkpoints instead of re-optimizing.

For the CSG-based counterpart (optimizing CSG primitive parameters instead of a free mesh), see [opt_csg.py](opt_csg.py).

##  Validation against OpenMC (multi-group)

Compares the differentiable renderer's tallies against ground-truth OpenMC multi-group Monte Carlo runs on the same sphere/torus geometry.

**Step 1 — generate the OpenMC reference:**

```bash
docker run --rm -it -v "${PWD}:/work" -w /work openmc/openmc:v0.13.2 python3 spheres_mg_openmc.py
# or the voxelized-tally variant:
docker run --rm -it -v "${PWD}:/work" -w /work openmc/openmc:v0.13.2 python3 spheres_voxel_mg_openmc.py
```

- [spheres_mg_openmc.py](spheres_mg_openmc.py) — two-sphere shielding geometry, 3-group energy structure (`E_REAL_ASC`), global track-length/reaction-rate tallies. Writes to `results_spheres/`.
- [spheres_voxel_mg_openmc.py](spheres_voxel_mg_openmc.py) — same problem with a voxel mesh tally (`.npz` + metadata JSON), for comparing spatially-resolved flux. Writes to `results_spheres_voxel/`.
- [open_mc_multi_group.py](open_mc_multi_group.py) / [test_openmc_photon.py](test_openmc_photon.py) — supporting single/multi-group XS builders and sphere/torus finite-difference-vs-OpenMC comparison utilities.

**Step 2 — run the matching differentiable-renderer scene** in [tests_mg_pbox.py](tests_mg_pbox.py). The OpenMC-comparable scene loaders are:

- `load_exp1_sphere_openmc_multi_group(params)` / `load_exp1_torus_openmc_multi_group(params)` — multi-group
- `load_exp1_sphere_openmc(sig_t, albedo)` / `load_exp1_torus_openmc(sig_t, albedo)` — single-group

Driver functions such as `two_sphere_radius`, `two_torus_offset`, and `test_parameter_gradient_multi` run the renderer with the same cross sections/geometry and are set up (docstring: *"save value data and gradient data, compare with openmc results"*) to be diffed against the `results_spheres*` output from Step 1. Edit the `__main__` block at the bottom of `tests_mg_pbox.py` to select which case to run.

---