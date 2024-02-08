import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

from kitti_env import KITTIEnv
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(b * x) + c


def main():
    all_points = []
    counts = []
    centers = []
    npys_path = KITTIEnv.npys_path
    labels_path = KITTIEnv.labels_path
    (min_x, max_x) = KITTIEnv.x_range
    (min_y, max_y) = KITTIEnv.y_range
    scene2humans = {}
    for npy_f in os.listdir(npys_path):
        if "_" not in npy_f:
            continue

        center = np.load(f"{labels_path}/{npy_f}")[:2]
        if not ((min_x < center[0] < max_x) and (min_y < center[1] < max_y)):
            continue

        points = np.load(f"{npys_path}/{npy_f}")
        all_points.append(points)
        counts.append(len(points))
        centers.append(center)
        scene = npy_f.split("_")[0]
        if scene not in scene2humans:
            scene2humans[scene] = []

        scene2humans[scene].append(npy_f)

    human_counts = np.array([len(humans) for humans in scene2humans.values()])
    human_count_frequencies = np.unique(human_counts, return_counts=True)
    human_count_props = human_count_frequencies[1][:10]
    human_count_props = human_count_props / human_count_props.sum()
    np.save("human_count_props.npy", human_count_props)

    counts = np.array(counts)
    centers = np.stack(centers)
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.concatenate(all_points))
    )
    o3d.visualization.draw_geometries([pcd])

    dists = np.linalg.norm(centers, axis=1)
    popt, pcov = curve_fit(func, -dists, counts)
    plt.scatter(dists, counts)
    plt.scatter(dists, func(-dists, *popt), label="Fitted Curve")
    plt.scatter(
        dists, KITTIEnv.min_prop * func(-dists, *popt), label="Reduced Fitted Curve"
    )
    plt.show()
    print(np.quantile(dists, [0.9]))
    plt.hist(dists, bins=100)
    plt.show()
    plt.hist(centers[:, 0], bins=100)
    plt.show()
    print(np.quantile(np.abs(centers[:, 1]), [0.9]))
    plt.hist(centers[:, 1], bins=100)
    plt.show()

    count_idxs = counts.argsort()
    idx = count_idxs[400]
    print(counts[idx])
    print(centers[idx])
    small_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points[idx]))
    o3d.visualization.draw_geometries([small_pcd])

    print(np.quantile(counts, [0.1, 0.25, 0.5]))

    height = max_x
    width = 2 * max_y
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


if __name__ == "__main__":
    main()
