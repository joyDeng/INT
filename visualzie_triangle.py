import numpy as np
import matplotlib.pyplot as plt

# p0:  1.390714406967163 -0.511243999004364 0.021149249747395515
# p1:  1.3931879997253418 -0.5020279884338379 -0.037429798394441605
# p2:  1.4090640544891357 -0.45886850357055664 7.78602043283172e-06
# ray:  -1.3916730880737305 -0.5079357624053955 0.0
# ray:  1.0 0.0 0.0

# Triangle vertices
p0 = np.array([1.390714406967163, -0.511243999004364, 0.021149249747395515])
p1 = np.array([1.3931879997253418, -0.5020279884338379, -0.037429798394441605])
p2 = np.array([1.4090640544891357, -0.45886850357055664, 7.78602043283172e-06])

# Ray
ray_o = np.array([-1.3916730880737305, -0.5079357624053955, 0.0])
ray_d = np.array([1.0, 0.0, 0.0])
ray_d = ray_d / np.linalg.norm(ray_d)  # normalize

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# --- Triangle ---
triangle_x = [p0[0], p1[0], p2[0], p0[0]]
triangle_y = [p0[1], p1[1], p2[1], p0[1]]
triangle_z = [p0[2], p1[2], p2[2], p0[2]]

ax.plot(triangle_x, triangle_y, triangle_z, color='blue', linewidth=2)
ax.scatter(*p0, color='blue')
ax.scatter(*p1, color='blue')
ax.scatter(*p2, color='blue')

# --- Ray ---
t = np.linspace(0, 4.0, 10)
ray_pts = ray_o[None, :] + t[:, None] * ray_d[None, :]

ax.plot(ray_pts[:, 0], ray_pts[:, 1], ray_pts[:, 2], color='red', linewidth=2)
ax.scatter(*ray_o, color='red')

# --- Labels ---
ax.text(*p0, "p0")
ax.text(*p1, "p1")
ax.text(*p2, "p2")
ax.text(*ray_o, "ray origin")

# --- Axis settings ---
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Ray – Triangle Visualization")
ax.set_box_aspect([1, 1, 1])

plt.show()
