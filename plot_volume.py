import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

    

def visualize_slice(id):
    # --- Load or create your 3D volume ---
    # Replace this with your own data, e.g.: volume = np.load("volume.npy")
    # Shape convention: (Z, Y, X)
    # volume = np.random.rand(128, 128, 128)
    volume = np.load("{:02d}_two_sphere_collision_density_gradients_fd_numpy.npy".format(id))
    

    # --- Pick a slice and orientation ---
    z_index = volume.shape[0] // 2   # middle axial slice
    slice_2d = volume[z_index, :, :] # axial: Z fixed


    plt.figure(figsize=(5, 5))
    plt.imshow(slice_2d, cmap='RdBu', origin='lower')
    plt.title(f'Axial slice z={z_index}')
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.show()

def visualize_two(id):
    field = np.load("{}_two_sphere_collision_density_numpy.npy".format(id))
    vol1 = np.load("{:02d}_two_sphere_collision_density_gradients_fd_numpy.npy".format(id))
    vol2 = np.load("{:02d}_two_sphere_collision_density_gradients_ad_numpy.npy".format(id))

    # --- Choose slice index and orientation ---
    z_index = vol1.shape[0] // 2 # middle slice along z-axis
    slice1 = vol1[z_index, :, :]
    slice2 = vol2[z_index, :, :]
    field_slice = field[z_index, :, :]
    print("average value", "finite element: ", np.mean(slice1), "auto differentiation: ", np.mean(slice2))
    # print(slice2.shape, field_slice.shape)

    # --- Compute common color scale ---
    vmin = min(slice1.min(), slice2.min())
    vmax = max(slice1.max(), slice2.max())
    vb = max(abs(vmin), abs(vmax))

    # --- Set up figure ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Plot first volume ---
    im1 = axes[0].imshow(slice1, cmap='RdBu', origin='lower', vmin=-vb, vmax=vb)
    axes[0].set_title(f'Finite difference – Slice {z_index}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Plot second volume ---
    im2 = axes[1].imshow(slice2, cmap='RdBu', origin='lower', vmin=-vb, vmax=vb)
    axes[1].set_title(f'Automatic differentiation – Slice {z_index}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # --- Compute error map (absolute difference) ---
    error_map = np.sqrt(np.power(slice2 - slice1, 2.0))

    im3 = axes[2].imshow(error_map, cmap='viridis', origin='lower')
    axes[2].set_title('Error Map |A − B|')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    im4 = axes[3].imshow(field_slice, cmap='plasma', origin='lower', vmin=0.0)
    axes[3].set_title('fluence')
    axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("bounce1_fdvsad_0307_50.png")
    # plt.show()

def visualize_test():
    field1 = np.load("test_beam_hat_numpy.npy")
    z_index = field1.shape[0] // 2  # middle slice along z-axis
    slice1 = field1[z_index, :, :]
    slice2 = field1[z_index+1, :, :]
    slice3 = field1[z_index-1, :, :]
    slice4 = field1[z_index-2, :, :]
    slice5 = field1[z_index+2, :, :]
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    v_max = np.max(field1)
    v_min = np.min(field1)

    print(v_min, v_max)
    print(np.min(slice2), np.min(slice3),np.min(slice4))

    # --- Plot first volume ---
    im1 = axes[0].imshow(slice1, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
    axes[0].set_title(f'Slice {z_index}')
    axes[0].axis('off')

    im2 = axes[1].imshow(slice2, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
    axes[1].set_title(f'Slice {z_index+1}')
    axes[1].axis('off')

    im3 = axes[2].imshow(slice3, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
    axes[2].set_title(f'Slice {z_index-1}')
    axes[2].axis('off')

    im4 = axes[3].imshow(slice4, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
    axes[3].set_title(f'Slice {z_index-2}')
    axes[3].axis('off')

    im5 = axes[4].imshow(slice5, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
    axes[4].set_title(f'Slice {z_index+2}')
    axes[4].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def visualize_one(id):
    field1 = np.load("{}+_two_sphere_collision_density_numpy.npy".format(id))
    field2 = np.load("{}-_two_sphere_collision_density_numpy.npy".format(id))
    vol1 = np.load("{:02d}_two_sphere_collision_density_gradients_fd_numpy.npy".format(id))

    # --- Choose slice index and orientation ---
    z_index = vol1.shape[0] // 2 - 1 # middle slice along z-axis
    slice1 = field1[z_index, :, :]
    slice2 = field2[z_index, :, :]
    field_slice = vol1[z_index, :, :]
    print("average value", "finite element: ", np.mean(slice1), "auto differentiation: ", np.mean(slice2))
    # print(slice2.shape, field_slice.shape)

    cv2.imwrite("slice1.hdr", slice1)
    cv2.imwrite("slice2.hdr", slice2)

    # --- Compute common color scale ---
    vmin = min(slice1.min(), slice2.min())
    vmax = max(slice1.max(), slice2.max())
    vb = max(abs(vmin), abs(vmax))

    # --- Set up figure ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Plot first volume ---
    im1 = axes[0].imshow(slice1, cmap='RdBu', origin='lower')
    axes[0].set_title(f'+ – Slice {z_index}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Plot second volume ---
    im2 = axes[1].imshow(slice2, cmap='RdBu', origin='lower')
    axes[1].set_title(f'- – Slice {z_index}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    print("slice 1 - slice 2: ", np.sum(np.abs(slice1 - slice2)))
    # --- Compute error map (absolute difference) ---
    error_map = np.sqrt(np.power(slice2 - slice1, 2.0))
    cv2.imwrite("error_map.hdr", slice2 - slice1)

    im3 = axes[2].imshow(error_map, cmap='viridis', origin='lower')
    axes[2].set_title('Error Map |A − B|')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    im4 = axes[3].imshow(field_slice, cmap='plasma', origin='lower')
    axes[3].set_title('fluence')
    axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def debug_beam(file_id):
    for i in range(2):
        vid_p = np.load("{}+_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))
        vid_m = np.load("{}-_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))

        start_p = np.load("start_{}+_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))
        start_m = np.load("start_{}-_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))
        
        end_p = np.load("end_{}+_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))
        end_m = np.load("end_{}-_{:02d}_two_sphere_collision_density_vertices.npy".format(file_id, i))

        only_in_a = np.setdiff1d(vid_p, vid_m)
        only_in_b = np.setdiff1d(vid_m, vid_p)
        print(only_in_a)
        print(only_in_b)
        idx = 407
        print("plus start: ", start_p[:, idx], " end: ", end_p[:, idx])
        print("min start: ", start_m[:, idx], " end: ", end_m[:, idx])



        # print("number of paths plus: ", start_p.shape)
        # print("number of paths min: ", start_m.shape)
        # print("vid_p: ", vid_p.shape)
        # print("vid_m: ", vid_m.shape)

def load_data_fd(name):
    values = np.load(f"{name}_openmc_fd_value.npy")
    gradient = np.load(f"{name}_openmc_fd_gradient.npy")
    errors = np.load(f"{name}_openmc_fd_gradient_errors.npy")
    val_errors = np.load(f"{name}_openmc_fd_value_errors.npy")
    xs     = np.load(f"{name}_openmc_fd_xs.npy")

    return xs, values, gradient, errors, val_errors

def load_data_ad(name):
    gradient = np.load(f"{name}_ad_gradients.npy")
    values = np.load(f"{name}_ad_values.npy")
    gradient_error = np.load(f"{name}_ad_gradients_errors.npy")
    values_error = np.load(f"{name}_ad_values_errors.npy")
    return values, gradient, values_error, gradient_error


def plot_with_errors(name):
    xs, values, gradient, gradient_errors, val_errors = load_data_fd(name)
    values_ad, gradient_ad, values_error, gradient_errors = load_data_ad(name)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=True, figsize=(12, 6)
    )

    # Color-blind friendly colormap
    cmap = plt.get_cmap("RdBu")
    color_value = cmap(0.15)
    mts_value = cmap(0.75)
    color_grad  = cmap(0.15)
    mts_ad    = cmap(0.75)   # new color for AD line

    print(xs)

    print(val_errors)

    # --- Value plot ---
    ax1.errorbar(
        xs,
        values,
        yerr=val_errors,
        fmt="o-",
        linewidth=2,
        markersize=5,
        capsize=4,
        color=color_grad,
        alpha=0.5,
        label="OpenMC (FD)",
    )

   # --- Value plot ---
    ax1.errorbar(
        xs,
        values_ad,
        yerr=values_error,
        fmt="o-",
        linewidth=2,
        markersize=5,
        capsize=4,
        color=mts_ad,
        alpha=0.5,
        label="Ours (AD)",
    )


    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    # --- Gradient plot with error bars (MC) ---
    ax2.errorbar(
        xs,
        gradient,
        yerr=gradient_errors,
        fmt="o-",
        linewidth=2,
        markersize=5,
        capsize=4,
        color=color_grad,
        alpha=0.5,
        label="OpenMC (FD)",
    )

    # --- AD gradient line (new) ---
    ax2.plot(
        xs,
        gradient_ad,
        "--s",
        linewidth=2,
        markersize=4,
        color=mts_ad,
        label="Ours (AD)",
    )

    ax2.set_xlabel("Verticle offset of inner sphere")
    ax2.set_ylabel("Gradient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name}_gradient.png", bbox_inches="tight")
    plt.close(fig)

def plot_with_errors_multiple_energy(name):
    xs, values, gradient, gradient_errors_fd, values_error_fd = load_data_fd(name)
    values_ad, gradient_ad, values_error, gradient_errors_ad = load_data_ad(name)

    xs = np.asarray(xs)
    values = np.asarray(values)
    gradient = np.asarray(gradient)
    gradient_errors = np.asarray(gradient_errors_fd)

    print("gradient error shape", gradient.shape)

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

    # plt.rcParams.update({
    #     "font.size": 18,          # base
    #     "axes.labelsize": 20,
    #     "xtick.labelsize": 16,
    #     "ytick.labelsize": 16,
    #     "legend.fontsize": 16,
    #     "lines.linewidth": 2.5,
    #     "lines.markersize": 8,
    # })

    mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
    "font.size": 18,              # SIGGRAPH body text ≈ 9pt
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "pdf.fonttype": 42,           # embed TrueType (required for submissions)
    "ps.fonttype": 42,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 6))

    # Use a color-blind friendly colormap; pick distinct colors for each group
    cmap = plt.get_cmap("inferno")
    if 2 * G == 1:
        colors = [cmap(0.6)]
    else:
        colors = [cmap(t) for t in np.linspace(0.1, 0.9, 2 * G)]

    # --- Value plot: one curve per energy group ---
    for g in range(G):
        ax1.plot(
            xs,
            values[:, g],
            "o-",
            linewidth=2,
            markersize=5,
            color=colors[g * 2],
            alpha=0.6,
            label=f"Ref g-{g+1}",
        )

        ax1.plot(
            xs,
            values_ad[:, G-g-1],
            "s--",
            markerfacecolor='none',
            linewidth=2,
            markersize=7,
            color=colors[g * 2+1],
            alpha=0.6,
            label=f"Ours g-{g+1}"
        )
    ax1.set_ylabel("Value of Energy Leakage")
    ax1.set_xlabel("Radius of Inner Sphere")
    # ax1.legend(title="Energy group", ncol=min(G, 3))
    ax1.grid(True, alpha=0.3)

    # --- Gradient plot with error bars: one curve per energy group ---
    for g in range(G):
        ax2.errorbar(
            xs,
            gradient[:, g],
            yerr=gradient_errors_fd[:, g],
            fmt="o-",
            linewidth=2,
            markersize=5,
            capsize=4,
            color=colors[g * 2],
            alpha=0.6,
            label=f"Ref g-{g+1}",
        )

        ax2.errorbar(
            xs,
            gradient_ad[:, G-g-1],
            yerr=gradient_errors_ad[:, g],
            fmt="s--",
            linewidth=2,
            markersize=7,
            markerfacecolor='none',
            capsize=4,
            color=colors[g * 2+1],
            alpha=0.6,
            label=f"Ours g-{g+1}",
        )
    ax2.set_xlabel("Sig_t of Inner Sphere")
    ax2.set_ylabel("Gradient")
    # ax2.legend(title="Energy group", ncol=min(G, 3))
    ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        # loc="upper center",
        # bbox_to_anchor=(0.5, 0.98),   # inside, slightly lowered
        ncol=1,                       # compact: 3 × 2 rows
        frameon=True,
        borderaxespad=0.2,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    plt.tight_layout()
    plt.savefig(f"{name}_gradient.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    # visualize_two(2)
    # visualize_test()
    # debug_beam(2)
    plot_with_errors_multiple_energy("sphere_sig_t_multi_3")