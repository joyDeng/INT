"""
fno_geo_flux.py
===============
Fourier Neural Operator (FNO-3D) that predicts the neutron voxel-flux map
for a given geometry + material configuration.

Input to the FNO  (3 channels, all at resolution N³ over [-r_bb, r_bb]³):
  ch0  – SDF φ of the inner OBJ surface (resampled from geo_sdf/*.pt)
  ch1  – σt occupation: σt_inner where φ < 0 (inside OBJ), σt_shell elsewhere
  ch2  – albedo occupation: same spatial mask, albedo_inner / albedo_shell

Output (1 channel):  predicted neutron flux field  (N, N, N)

Supervision:
  Ground-truth flux is produced by calling
  render_geo_voxels.render_flux_voxels(..., num_passes=1).

SDF file format (geo_sdf/):
  Each file is a .pt dict with at least:
    'phi'      : (N, N, N) float32 – signed distance (negative = inside geometry)
    'bbox_min' : (3,) tensor
    'bbox_max' : (3,) tensor

Naming convention:
  geo_norm_obj/<stem>.obj  ↔  geo_sdf/<stem>.pt

Usage:
  python fno_geo_flux.py [--obj_dir ...] [--sdf_dir ...] [--N 64] [--epochs 200]
  python fno_geo_flux.py --predict --ckpt fno_flux_best.pt --obj <path>
  python fno_geo_flux.py --list_cache          # show cached ground-truth files
"""

from __future__ import annotations
import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local import (must be on PYTHONPATH / same directory) ──────────────────
import render_geo_voxels as rgv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MEDIA_DIR   = "/media/xideng/973f0610-62ab-4e4e-9192-dde2782393be1/data"
GEO_OBJ_DIR = os.path.join(MEDIA_DIR, "geo_norm_obj")
GEO_SDF_DIR = os.path.join(MEDIA_DIR, "geo_sdf")

# ---------------------------------------------------------------------------
# FNO-3D  (Li et al., 2020 – Fourier Neural Operator)
# ---------------------------------------------------------------------------

class SpectralConv3d(nn.Module):
    """3-D Fourier integral operator layer.

    Keeps the lowest modes1×modes2×modes3 Fourier modes and multiplies
    them by a learnable complex-valued weight tensor.  The full transform
    is applied via rfftn / irfftn so we only store four octant weights.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes1, modes2, modes3)
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, in_ch, x, y, z) × (in_ch, out_ch, x, y, z) → (B, out_ch, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", u, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, X, Y, Z = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        m1, m2, m3 = self.modes1, self.modes2, self.modes3

        out = torch.zeros(B, self.w1.shape[1], X, Y, Z // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :,  :m1,  :m2, :m3] = self._mul(x_ft[:, :,  :m1,  :m2, :m3], self.w1)
        out[:, :, -m1:,  :m2, :m3] = self._mul(x_ft[:, :, -m1:,  :m2, :m3], self.w2)
        out[:, :,  :m1, -m2:, :m3] = self._mul(x_ft[:, :,  :m1, -m2:, :m3], self.w3)
        out[:, :, -m1:, -m2:, :m3] = self._mul(x_ft[:, :, -m1:, -m2:, :m3], self.w4)
        return torch.fft.irfftn(out, s=(X, Y, Z))


class FNOBlock3d(nn.Module):
    """One FNO layer: spectral conv + pointwise bypass + GELU."""

    def __init__(self, width: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.spec = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.lin  = nn.Conv3d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.lin(x))


class FNO3d(nn.Module):
    """
    Full 3-D Fourier Neural Operator.

    Parameters
    ----------
    in_ch    : input channels  (default 3: SDF + σt + albedo)
    out_ch   : output channels (default 1: scalar flux)
    width    : lifting dimension
    modes    : Fourier modes retained per spatial axis
    depth    : number of FNO blocks
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 width: int = 32, modes: int = 12, depth: int = 4):
        super().__init__()
        self.lift   = nn.Conv3d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock3d(width, modes, modes, modes) for _ in range(depth)])
        self.proj1  = nn.Conv3d(width, width * 2, 1)
        self.proj2  = nn.Conv3d(width * 2, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return F.softplus(x)      # non-negative flux output


# ---------------------------------------------------------------------------
# SDF loading and input tensor construction
# ---------------------------------------------------------------------------

def load_sdf_resampled(pt_path: str, N_out: int, r_bb: float = 2.0) -> np.ndarray:
    """
    Load the inner-surface SDF from a geo_sdf .pt file and resample it
    onto the FNO voxel grid  (N_out³ covering [-r_bb, r_bb]³).

    Voxels that fall outside the stored bbox are set to a large positive
    value (clearly "outside the inner geometry" → shell material).

    Returns
    -------
    phi : (N_out, N_out, N_out) float32 ndarray
    """
    d       = torch.load(pt_path, map_location="cpu", weights_only=False)
    phi_src = d["phi"].unsqueeze(0).unsqueeze(0).float()  # (1,1,N,N,N)
    b_min   = d["bbox_min"].numpy()                        # (3,)
    b_max   = d["bbox_max"].numpy()                        # (3,)

    # Build FNO grid in world-space, then normalise to the SDF bbox
    lin = np.linspace(-r_bb, r_bb, N_out, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")  # each (N,N,N)

    def norm(val, lo, hi):
        return 2.0 * (val - lo) / (hi - lo) - 1.0          # maps [lo,hi] → [-1,1]

    gx_n = norm(gx, b_min[0], b_max[0])
    gy_n = norm(gy, b_min[1], b_max[1])
    gz_n = norm(gz, b_min[2], b_max[2])

    # grid_sample convention: grid[..., 0]=x→W, [..1]=y→H, [..2]=z→D
    grid = torch.from_numpy(
        np.stack([gz_n, gy_n, gx_n], axis=-1)              # (N,N,N,3)
    ).unsqueeze(0)                                           # (1,N,N,N,3)

    phi_out = F.grid_sample(
        phi_src, grid,
        mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze().numpy()                                      # (N,N,N)

    # Voxels outside the stored bbox → clearly in the shell (large positive SDF)
    outside = (gx_n < -1) | (gx_n > 1) | \
              (gy_n < -1) | (gy_n > 1) | \
              (gz_n < -1) | (gz_n > 1)
    phi_out[outside] = max(abs(float(phi_src.min())), 0.1) + 1.0

    return phi_out.astype(np.float32)


def build_input_tensor(phi:          np.ndarray,
                       sig_t_inner:  float,
                       albedo_inner: float,
                       sig_t_shell:  float,
                       albedo_shell: float) -> torch.Tensor:
    """
    Build the (3, N, N, N) FNO input tensor.

    Channel 0 – SDF φ (raw signed-distance values; negative = inside OBJ)
    Channel 1 – σt occupation field
    Channel 2 – albedo occupation field

    For the occupation channels:
        value = param_inner  where φ < 0  (inside the inner geometry)
        value = param_shell  where φ ≥ 0  (shell or void region)
    """
    inside        = (phi < 0.0)
    sig_t_field   = np.where(inside, sig_t_inner,  sig_t_shell ).astype(np.float32)
    albedo_field  = np.where(inside, albedo_inner, albedo_shell).astype(np.float32)
    return torch.from_numpy(np.stack([phi, sig_t_field, albedo_field], axis=0))


def sdf_path_for_obj(obj_path: str, sdf_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(sdf_dir, f"{stem}.pt")


# ---------------------------------------------------------------------------
# Material parameter sampling
# ---------------------------------------------------------------------------

PARAM_RANGES = dict(
    sig_t_inner  = (0.1, 2.0),
    albedo_inner = (0.5, 0.99),
    sig_t_shell  = (0.05, 1.0),
    albedo_shell = (0.5, 0.99),
)


def sample_params(rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(*v)) for k, v in PARAM_RANGES.items()}


# ---------------------------------------------------------------------------
# Dataset  (online ground-truth rendering)
# ---------------------------------------------------------------------------

class GeoFluxDataset(Dataset):
    """
    Each sample:
      1. Picks a geometry (OBJ + SDF pair) by index.
      2. Samples random material parameters.
      3. Builds the (3, N, N, N) FNO input from the SDF + occupation tensor.
      4. Calls render_flux_voxels(num_passes=1) for the ground-truth flux.

    Ground-truth results are cached to disk under `cache_dir` so each
    (geometry, param-seed) combination is only rendered once.

    Parameters
    ----------
    obj_paths    : list of str  – full paths to inner OBJ files
    sdf_dir      : str          – directory with <stem>.pt SDF files
    N            : int          – voxel grid resolution
    max_bounce   : int
    num_neutrons : int
    seed         : int          – base RNG seed (each item uses seed + idx)
    cache_dir    : str or None  – directory to cache ground-truth .npy files
    """

    def __init__(self, obj_paths: list[str], sdf_dir: str,
                 N: int = 64, max_bounce: int = 2, num_neutrons: int = 100_000,
                 seed: int = 0, cache_dir: str | None = None):
        self.obj_paths    = list(obj_paths)
        self.sdf_dir      = sdf_dir
        self.N            = N
        self.max_bounce   = max_bounce
        self.num_neutrons = num_neutrons
        self.seed         = seed
        self.cache_dir    = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.obj_paths)

    def __getitem__(self, idx: int):
        # Retry up to len(dataset) times to find a valid sample
        for attempt in range(len(self.obj_paths)):
            try:
                return self._load_one((idx + attempt) % len(self.obj_paths))
            except Exception as e:
                print(f"  [dataset] skipping idx={(idx+attempt)%len(self.obj_paths)}: {e}")
        raise RuntimeError("No valid sample found after full dataset scan.")

    def _load_one(self, idx: int):
        obj_path    = self.obj_paths[idx]
        rng         = np.random.default_rng(self.seed + idx)
        params      = sample_params(rng)
        render_seed = int(rng.integers(0, 2**31))

        # ── SDF → FNO input ───────────────────────────────────────────────
        pt_path = sdf_path_for_obj(obj_path, self.sdf_dir)
        phi     = load_sdf_resampled(pt_path, self.N)
        x_in    = build_input_tensor(phi, **params)           # (3, N, N, N)

        # ── ground-truth flux  (cached if requested) ──────────────────────
        y_out = self._get_flux(obj_path, params, render_seed)  # (1, N, N, N)

        return x_in, y_out, params

    def _get_flux(self, obj_path: str, params: dict,
                  render_seed: int) -> torch.Tensor:
        """Return (1, N, N, N) float32 flux tensor, using disk cache if set."""
        if self.cache_dir:
            stem     = os.path.splitext(os.path.basename(obj_path))[0]
            tag      = (f"st{params['sig_t_inner']:.4f}_ai{params['albedo_inner']:.4f}_"
                        f"ss{params['sig_t_shell']:.4f}_as{params['albedo_shell']:.4f}_"
                        f"s{render_seed}")
            cache_fp = os.path.join(self.cache_dir, f"{stem}_{tag}.npy")
            if os.path.exists(cache_fp):
                arr = np.load(cache_fp).reshape(self.N, self.N, self.N)
                return torch.from_numpy(arr).unsqueeze(0)

        flux_dr = rgv.render_flux_voxels(
            obj_path,
            N            = self.N,
            max_bounce   = self.max_bounce,
            sig_t_inner  = params["sig_t_inner"],
            albedo_inner = params["albedo_inner"],
            sig_t_shell  = params["sig_t_shell"],
            albedo_shell = params["albedo_shell"],
            num_neutrons = self.num_neutrons,
            seed         = render_seed,
            num_passes   = 1,
        )
        arr = flux_dr.numpy().reshape(self.N, self.N, self.N).astype(np.float32)

        if self.cache_dir:
            np.save(cache_fp, arr)

        return torch.from_numpy(arr).unsqueeze(0)             # (1, N, N, N)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Normalised L² loss averaged over the batch."""
    diff   = torch.norm(pred - target, p=2, dim=(-3, -2, -1))
    norm_t = torch.norm(target,        p=2, dim=(-3, -2, -1)).clamp_min(eps)
    return (diff / norm_t).mean()


# ---------------------------------------------------------------------------
# wandb slice visualisation helper
# ---------------------------------------------------------------------------

def _slice_comparison_image(pred: np.ndarray, gt: np.ndarray,
                             cmap: str = "plasma") -> "wandb.Image":
    """
    Build a 3-row × 3-col figure comparing predicted vs ground-truth flux
    across the three centre-plane slices (XY, XZ, YZ).

    Row 0 – ground truth
    Row 1 – prediction
    Row 2 – absolute error

    Returns a wandb.Image ready to be logged.
    """
    import wandb

    N = pred.shape[0]
    iz, iy, ix = N // 2, N // 2, N // 2

    slices_gt   = [gt[iz, :, :],   gt[:, iy, :],   gt[:, :, ix]]
    slices_pred = [pred[iz, :, :], pred[:, iy, :], pred[:, :, ix]]
    titles      = ["XY (z-mid)", "XZ (y-mid)", "YZ (x-mid)"]

    vmax = max(gt.max(), pred.max()) + 1e-8
    vmin = 0.0

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    row_labels = ["GT", "Pred", "|Error|"]

    for col in range(3):
        err = np.abs(slices_pred[col] - slices_gt[col])

        for row, (data, label) in enumerate(zip(
                [slices_gt[col], slices_pred[col], err],
                row_labels)):
            ax = axes[row, col]
            vmax_row = vmax if row < 2 else err.max() + 1e-8
            im = ax.imshow(data, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax_row, aspect="equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    import wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── geometry files ────────────────────────────────────────────────────
    obj_files = sorted(glob.glob(os.path.join(args.obj_dir, "*.obj")))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files in {args.obj_dir}")

    pairs = [(o, sdf_path_for_obj(o, args.sdf_dir))
             for o in obj_files if os.path.exists(sdf_path_for_obj(o, args.sdf_dir))]
    if not pairs:
        raise FileNotFoundError(
            f"No matching SDF .pt files found in {args.sdf_dir}\n"
            f"Expected: <stem>.pt matching <stem>.obj in {args.obj_dir}")
    obj_files = [p[0] for p in pairs]
    print(f"  geometry / SDF pairs found: {len(obj_files)}")

    # train / val split (90 / 10)
    n_val       = max(1, len(obj_files) // 10)
    val_paths   = obj_files[-n_val:]
    train_paths = obj_files[:-n_val] if len(obj_files) > n_val else obj_files

    cache_dir = f"{args.out_prefix}_gt_cache" if args.cache else None

    train_ds = GeoFluxDataset(train_paths, args.sdf_dir,
                              N=args.N, max_bounce=args.max_bounce,
                              num_neutrons=args.num_neutrons,
                              seed=args.seed, cache_dir=cache_dir)
    val_ds   = GeoFluxDataset(val_paths,   args.sdf_dir,
                              N=args.N, max_bounce=args.max_bounce,
                              num_neutrons=args.num_neutrons,
                              seed=args.seed + 99999, cache_dir=cache_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=0)

    # ── model ──────────────────────────────────────────────────────────────
    model = FNO3d(in_ch=3, out_ch=1,
                  width=args.width, modes=args.modes, depth=args.depth
                  ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FNO3d parameters: {n_params:,}  "
          f"(width={args.width}, modes={args.modes}, depth={args.depth})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    # ── wandb init ─────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run or args.out_prefix,
        config  = vars(args),
        mode    = "disabled" if args.no_wandb else "online",
    )
    wandb.watch(model, log="gradients", log_freq=100)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x_in, y_out, _ in train_loader:
            x_in  = x_in.to(device)
            y_out = y_out.to(device)
            pred  = model(x_in)
            loss  = relative_l2(pred, y_out)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ── validate + collect one sample for visualisation ───────────────
        model.eval()
        val_loss   = 0.0
        vis_pred   = None     # first val sample kept for image logging
        vis_gt     = None
        vis_params = None

        with torch.no_grad():
            for i, (x_in, y_out, params) in enumerate(val_loader):
                x_in  = x_in.to(device)
                y_out = y_out.to(device)
                p     = model(x_in)
                val_loss += relative_l2(p, y_out).item()
                if i == 0:
                    vis_pred   = p.squeeze().cpu().numpy()       # (N,N,N)
                    vis_gt     = y_out.squeeze().cpu().numpy()   # (N,N,N)
                    vis_params = params
        val_loss /= max(len(val_loader), 1)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        # ── wandb scalar logging ───────────────────────────────────────────
        log_dict = {
            "epoch":       epoch,
            "train/loss":  train_loss,
            "val/loss":    val_loss,
            "lr":          lr_now,
        }

        # ── slice visualisation (every vis_every epochs) ──────────────────
        if epoch % args.vis_every == 0 and vis_pred is not None:
            log_dict["val/flux_slices"] = _slice_comparison_image(
                vis_pred, vis_gt)
            # also log the SDF channel of the input for reference
            vis_sdf = x_in[0, 0].cpu().numpy()   # (N,N,N)
            N = vis_sdf.shape[0]
            fig_sdf, axes_sdf = plt.subplots(1, 3, figsize=(12, 4))
            for ax, (sl, title) in zip(axes_sdf, [
                    (vis_sdf[N//2, :, :], "XY"),
                    (vis_sdf[:, N//2, :], "XZ"),
                    (vis_sdf[:, :, N//2], "YZ")]):
                im = ax.imshow(sl, origin="lower", cmap="RdBu_r", aspect="equal")
                ax.set_title(f"SDF {title}"); ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig_sdf.suptitle(
                f"sig_t={vis_params['sig_t_inner'][0]:.2f}/{vis_params['sig_t_shell'][0]:.2f}  "
                f"albedo={vis_params['albedo_inner'][0]:.2f}/{vis_params['albedo_shell'][0]:.2f}",
                fontsize=10)
            fig_sdf.tight_layout()
            log_dict["val/sdf_input"] = wandb.Image(fig_sdf)
            plt.close(fig_sdf)

        wandb.log(log_dict, step=epoch)

        # ── checkpoint ────────────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = f"{args.out_prefix}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → saved best (val={best_val:.4f})")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"]    = epoch

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       f"{args.out_prefix}_ep{epoch:04d}.pt")

    wandb.finish()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(ckpt_path: str, obj_path: str, sdf_dir: str,
            sig_t_inner: float = 0.5, albedo_inner: float = 0.8,
            sig_t_shell: float = 0.1, albedo_shell: float = 0.9,
            N: int = 64, r_bb: float = 2.0,
            device: torch.device | None = None) -> np.ndarray:
    """Load a checkpoint and predict flux for one geometry + material config.

    Returns
    -------
    flux : (N, N, N) float32 numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("args", {})

    model = FNO3d(in_ch=3, out_ch=1,
                  width=cfg.get("width", 32),
                  modes=cfg.get("modes", 12),
                  depth=cfg.get("depth", 4)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pt_path = sdf_path_for_obj(obj_path, sdf_dir)
    phi     = load_sdf_resampled(pt_path, N, r_bb)
    x_in    = build_input_tensor(phi, sig_t_inner, albedo_inner,
                                 sig_t_shell, albedo_shell)
    with torch.no_grad():
        pred = model(x_in.unsqueeze(0).to(device))   # (1,1,N,N,N)
    return pred.squeeze().cpu().numpy()               # (N,N,N)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / infer FNO-3D neutron flux from SDF + material params")

    # mode
    parser.add_argument("--predict",  action="store_true",
                        help="Run inference instead of training")
    parser.add_argument("--ckpt",     type=str, default=None,
                        help="Checkpoint path for --predict")

    # data
    parser.add_argument("--obj_dir",       type=str,   default=GEO_OBJ_DIR)
    parser.add_argument("--sdf_dir",       type=str,   default=GEO_SDF_DIR)
    parser.add_argument("--obj",           type=str,   default=None,
                        help="Single OBJ for --predict")
    parser.add_argument("--N",             type=int,   default=64)
    parser.add_argument("--max_bounce",    type=int,   default=2)
    parser.add_argument("--num_neutrons",  type=int,   default=100_000)
    parser.add_argument("--cache",         action="store_true",
                        help="Cache rendered ground-truth to disk")

    # model
    parser.add_argument("--width",     type=int,   default=32,
                        help="Channel width after lifting (default 32)")
    parser.add_argument("--modes",     type=int,   default=12,
                        help="Fourier modes per axis (default 12 = N/64*12)")
    parser.add_argument("--depth",     type=int,   default=4,
                        help="Number of FNO blocks (default 4)")

    # training
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=1,
                        help="Batch size (keep at 1; each sample needs a render)")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=1994)
    parser.add_argument("--save_every",  type=int,   default=20)
    parser.add_argument("--out_prefix",  type=str,   default="fno_flux")

    # wandb
    parser.add_argument("--wandb_project", type=str, default="fno-geo-flux",
                        help="W&B project name")
    parser.add_argument("--wandb_run",     type=str, default=None,
                        help="W&B run name (defaults to out_prefix)")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Disable W&B logging (offline/dry-run)")
    parser.add_argument("--vis_every",     type=int, default=5,
                        help="Log slice comparison images every N epochs (default 5)")

    # material params for --predict
    parser.add_argument("--sig_t_inner",   type=float, default=0.5)
    parser.add_argument("--albedo_inner",  type=float, default=0.8)
    parser.add_argument("--sig_t_shell",   type=float, default=0.1)
    parser.add_argument("--albedo_shell",  type=float, default=0.9)

    args = parser.parse_args()

    if args.predict:
        if not args.ckpt:
            parser.error("--predict requires --ckpt")
        obj_path = args.obj or sorted(glob.glob(
            os.path.join(args.obj_dir, "*.obj")))[0]
        flux = predict(
            ckpt_path    = args.ckpt,
            obj_path     = obj_path,
            sdf_dir      = args.sdf_dir,
            sig_t_inner  = args.sig_t_inner,
            albedo_inner = args.albedo_inner,
            sig_t_shell  = args.sig_t_shell,
            albedo_shell = args.albedo_shell,
            N            = args.N,
        )
        out = f"{args.out_prefix}_pred.npy"
        np.save(out, flux)
        print(f"Saved predicted flux: {out}  shape={flux.shape}  "
              f"max={flux.max():.4e}")
    else:
        train(args)
