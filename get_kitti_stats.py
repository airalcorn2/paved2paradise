import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

from env_vars import MIN_PROP, PRISMS
from prepare_kitti_data import LABELS_PATH, NPYS_PATH
from scipy.optimize import curve_fit

(MIN_X, MAX_X) = PRISMS["kitti"]["x"]
(MIN_Y, MAX_Y) = PRISMS["kitti"]["y"]


def func(x, a, b, c):
    return a * np.exp(b * x) + c


all_points = []
counts = []
centers = []
for npy_f in os.listdir(NPYS_PATH):
    if "_" not in npy_f:
        continue

    center = np.load(f"{LABELS_PATH}/{npy_f}")[:2]
    if not ((MIN_Y < center[0] < MAX_X) and (MIN_Y < center[1] < MAX_Y)):
        continue

    points = np.load(f"{NPYS_PATH}/{npy_f}")
    all_points.append(points)
    counts.append(len(points))
    centers.append(center)

counts = np.array(counts)
centers = np.stack(centers)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(all_points)))
o3d.visualization.draw_geometries([pcd])

dists = np.linalg.norm(centers, axis=1)
popt, pcov = curve_fit(func, -dists, counts)
plt.scatter(dists, counts)
plt.scatter(dists, func(-dists, *popt), label="Fitted Curve")
plt.scatter(dists, MIN_PROP * func(-dists, *popt), label="Reduced Fitted Curve")
plt.show()
print(np.quantile(dists, [0.9]))
plt.hist(dists, bins=100)
plt.show()
plt.hist(centers[:, 0], bins=100)
plt.show()
print(np.quantile(np.abs(centers[:, 1]), [0.9]))
plt.hist(centers[:, 1], bins=100)
plt.show()

keep_x = centers[:, 0] < 20
keep_y = np.abs(centers[:, 1]) < 10
n_keep = (keep_x & keep_y).sum()
print(f"{100 * n_keep / len(centers)}%")

count_idxs = counts.argsort()
idx = count_idxs[400]
print(counts[idx])
print(centers[idx])
small_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points[idx]))
o3d.visualization.draw_geometries([small_pcd])

print(np.quantile(counts, [0.1, 0.25, 0.5]))

height = MAX_X
width = 2 * MAX_Y
grid = np.zeros((height, width))
rows = (height - centers[:, 0]).astype("int")
cols = (centers[:, 1] + width / 2).astype("int")
row_cols = np.stack([rows, cols]).T
(row_cols, counts) = np.unique(row_cols, axis=0, return_counts=True)
grid[row_cols[:, 0], row_cols[:, 1]] = counts
grid = grid / grid.sum()
plt.imshow(grid, cmap="hot", interpolation="nearest")
plt.show()

centers_mirror = np.copy(centers)
centers_mirror[:, 1] = -centers_mirror[:, 1]
centers = np.concatenate([centers, centers_mirror])
rows = (height - centers[:, 0]).astype("int")
cols = (centers[:, 1] + width / 2).astype("int")

row_cols = np.stack([rows, cols]).T
(row_cols, counts) = np.unique(row_cols, axis=0, return_counts=True)
grid[row_cols[:, 0], row_cols[:, 1]] = counts
grid = grid / grid.sum()
plt.imshow(grid, cmap="hot", interpolation="nearest")
plt.show()
uniform_grid = (grid > 0).astype("float")
uniform_grid /= uniform_grid.sum()
plt.imshow(uniform_grid, cmap="hot", interpolation="nearest")
plt.show()
