import matplotlib.pyplot as plt
import numpy as np

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
    z_index = vol1.shape[0] // 2  # middle slice along z-axis
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
    plt.savefig("finite_difference_vs_autodiff_1b_plane_emission_inner_sphere_100000000co_dense.png")
    # plt.show()


def visualize_one(id):
    field1 = np.load("{}+_two_sphere_collision_density_numpy.npy".format(id))
    field2 = np.load("{}-_two_sphere_collision_density_numpy.npy".format(id))
    vol1 = np.load("{:02d}_two_sphere_collision_density_gradients_fd_numpy.npy".format(id))

    # --- Choose slice index and orientation ---
    z_index = vol1.shape[0] // 2  # middle slice along z-axis
    slice1 = field1[z_index, :, :]
    slice2 = field2[z_index, :, :]
    field_slice = vol1[z_index, :, :]
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
    axes[0].set_title(f'+ – Slice {z_index}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Plot second volume ---
    im2 = axes[1].imshow(slice2, cmap='RdBu', origin='lower', vmin=-vb, vmax=vb)
    axes[1].set_title(f'- – Slice {z_index}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # --- Compute error map (absolute difference) ---
    error_map = np.sqrt(np.power(slice2 - slice1, 2.0))

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

if __name__ == "__main__":
    visualize_two(2)