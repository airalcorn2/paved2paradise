import json
import numpy as np
import open3d as o3d
import os

from kitti_env import KITTIEnv

BBOX_COLOR = [1, 0, 0]


def main():
    human_fs = os.listdir(KITTIEnv.final_npys_path)
    human_fs.sort()
    for human_f in human_fs:
        samp_name = human_f.split(".npy")[0]
        human_points = np.load(f"{KITTIEnv.final_npys_path}/{samp_name}.npy")

        with open(f"{KITTIEnv.final_jsons_path}/{samp_name}.json") as f:
            metadata = json.load(f)

        background_f = metadata["background"]
        mirror = "True" in background_f
        background_f = background_f.split("_")[0] + ".npy"

        background_points = np.load(f"{KITTIEnv.npys_path}/{background_f}")
        if mirror:
            background_points[:, 1] = -background_points[:, 1]

        occlude_idxs = np.load(f"{KITTIEnv.final_idxs_path}/{samp_name}.npy")
        background_mask = np.ones(len(background_points), dtype="bool")
        background_mask[occlude_idxs] = False

        labels = np.load(f"{KITTIEnv.final_labels_path}/{samp_name}.npy")
        center = labels[:3]
        extent = labels[3:6]
        bbox_R = labels[6:15].reshape(3, 3)
        bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
        bbox.color = BBOX_COLOR

        all_points = np.concatenate([human_points, background_points[background_mask]])
        all_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))

        print(f"samp_name: {samp_name}")
        print(f"metadata: {metadata}")
        print(f"len(occlude_idxs): {len(occlude_idxs)}\n")
        o3d.visualization.draw_geometries([bbox, all_pcd], samp_name)


if __name__ == "__main__":
    main()
