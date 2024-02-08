import json
import numpy as np
import open3d as o3d
import os
import pickle
import shutil
import sys

from kitti_env import KITTIEnv

VIS_SAMPS = 100
BBOX_COLOR = [1, 0, 0]


def save_render_parts():
    os.makedirs(KITTIEnv.renders_dir)
    obj_fs = os.listdir(KITTIEnv.final_npys_path)
    obj_fs.sort()
    for obj_f in obj_fs[:VIS_SAMPS]:
        samp_name = obj_f.split(".npy")[0]
        obj_points = np.load(f"{KITTIEnv.final_npys_path}/{samp_name}.npy")

        with open(f"{KITTIEnv.final_jsons_path}/{samp_name}.json") as f:
            metadata = json.load(f)

        bg_f = metadata["background"]
        bg_points = np.load(f"{KITTIEnv.unlevel_background_npys_path}/{bg_f}")

        occlude_idxs = np.load(f"{KITTIEnv.final_idxs_path}/{samp_name}.npy")
        bg_mask = np.ones(len(bg_points), dtype="bool")
        bg_mask[occlude_idxs] = False

        labels = np.load(f"{KITTIEnv.final_labels_path}/{samp_name}.npy")
        all_points = np.concatenate([obj_points, bg_points[bg_mask]])
        desc = f"samp_name: {samp_name}\nmetadata: {metadata}\n"
        desc += f"len(occlude_idxs): {len(occlude_idxs)}\n"
        render_parts = {
            "center": labels[:, :3],
            "extent": labels[:, 3:6],
            "bbox_R": labels[:, 6:15].reshape(-1, 3, 3),
            "all_points": all_points,
            "desc": desc,
        }

        with open(f"{KITTIEnv.renders_dir}/{samp_name}.pydict", "wb") as f:
            pickle.dump(render_parts, f)

    shutil.make_archive(KITTIEnv.renders_dir, "zip", KITTIEnv.renders_dir)
    shutil.rmtree(KITTIEnv.renders_dir)


def render_parts():
    shutil.rmtree(KITTIEnv.renders_dir, ignore_errors=True)
    shutil.unpack_archive(f"{KITTIEnv.renders_dir}.zip", KITTIEnv.renders_dir)
    samp_dicts = os.listdir(KITTIEnv.renders_dir)
    samp_dicts.sort()
    (min_x, max_x) = KITTIEnv.x_range
    (min_y, max_y) = KITTIEnv.y_range
    (min_z, max_z) = KITTIEnv.z_range
    for samp_dict in samp_dicts:
        with open(f"{KITTIEnv.renders_dir}/{samp_dict}", "rb") as f:
            render_parts = pickle.load(f)

        print(render_parts["desc"])
        geoms = []
        for i in range(len(render_parts["center"])):
            center = render_parts["center"][i]
            bbox_R = render_parts["bbox_R"][i]
            extent = render_parts["extent"][i]
            bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
            bbox.color = BBOX_COLOR
            geoms.append(bbox)

        all_points = render_parts["all_points"]
        in_x = (min_x < all_points[:, 0]) & (all_points[:, 0] < max_x)
        in_y = (min_y < all_points[:, 1]) & (all_points[:, 1] < max_y)
        in_z = (min_z < all_points[:, 2]) & (all_points[:, 2] < max_z)
        all_points = all_points[in_x & in_y & in_z]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points[:, :3]))
        geoms.append(pcd)
        samp_name = samp_dict.split(".pydict")[0]
        o3d.visualization.draw_geometries(geoms, samp_name)


if __name__ == "__main__":
    task = sys.argv[1]
    assert task in {"save", "show"}
    if task == "save":
        save_render_parts()
    else:
        render_parts()
