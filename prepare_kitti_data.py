import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import open3d as o3d
import os
import sys

from kitti_env import KITTIEnv
from PIL import Image
from scipy.spatial.transform import Rotation

DATA_TYPE = {
    False: {False: "neither", True: "cyclist"},
    True: {True: "both", False: "pedestrian"},
}


def prepare_object_labels(frame_annotations):
    for frame, annotations in frame_annotations:
        # No object labeled.
        if len(annotations["annotations"]) == 0:
            continue

        name = frame["name"]
        if os.path.exists(f"{KITTIEnv.raw_object_labels_path}/{name}.npy"):
            continue

        labels = []
        for bbox in annotations["annotations"]:
            center = []
            extent = []
            for xyz in ["x", "y", "z"]:
                center.append(bbox["position"][xyz])
                extent.append(bbox["dimensions"][xyz])

            center = np.array(center)
            extent = np.array(extent)

            q = []
            for xyzw in ["x", "y", "z", "w"]:
                q.append(bbox["rotation"][f"q{xyzw}"])

            q = np.array(q)
            labels.append(np.concatenate([[1], center, extent, q]))

        labels = np.stack(labels)
        np.save(f"{KITTIEnv.raw_object_labels_path}/{name}.npy", labels)


def prepare_object_labels_parallel(use_all):
    if os.path.exists(KITTIEnv.raw_object_labels_path):
        print(
            f"{KITTIEnv.raw_object_labels_path} already exists. Make sure it doesn't contain out-of-date data."
        )

    os.makedirs(KITTIEnv.raw_object_labels_path, exist_ok=True)
    with open(KITTIEnv.labels_json_f) as f:
        d = json.load(f)

    for sample_dict in d["dataset"]["samples"]:
        label_status = sample_dict["labels"]["ground-truth"]["label_status"]
        if (not use_all) and (label_status != "LABELED"):
            continue

        if "Crouch" in sample_dict["name"]:
            continue

        frames = sample_dict["attributes"]["frames"]
        frame_annotations = sample_dict["labels"]["ground-truth"]["attributes"][
            "frames"
        ]
        all_frame_annotations = list(zip(frames, frame_annotations))

        n_jobs = multiprocessing.cpu_count()
        frames_per_job = int(np.ceil(len(all_frame_annotations) / n_jobs))
        procs = []
        for job in range(n_jobs):
            start = job * frames_per_job
            end = start + frames_per_job
            frame_annotations = all_frame_annotations[start:end]
            proc = multiprocessing.Process(
                target=prepare_object_labels, args=(frame_annotations,)
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()


def worker(frames, show_pcd):
    (min_x, max_x) = KITTIEnv.x_range
    (min_y, max_y) = KITTIEnv.y_range
    (min_z, max_z) = KITTIEnv.z_range
    (samp_min_x, samp_max_x) = KITTIEnv.samp_x_range
    (samp_min_y, samp_max_y) = KITTIEnv.samp_y_range
    for frame in frames:
        frame_name = frame.split(".bin")[0]

        transforms = {}
        with open(f"{KITTIEnv.raw_backgrounds_path}/calib/{frame_name}.txt") as f:
            for line in f:
                # The left RGB camera (P2) is the reference camera. See Figure 3 and
                # Section III.C in Geiger et al. (2013).
                if (
                    line.startswith("R0_rect")
                    or line.startswith("Tr_velo_to_cam")
                    or line.startswith("P2")
                ):
                    (part, numbers) = line.strip().split(":")
                    if part == "R0_rect":
                        mat = np.array([float(x) for x in numbers.split()])
                        mat = mat.reshape(3, 3)
                    else:
                        mat = np.array([float(x) for x in numbers.split()])
                        mat = mat.reshape(3, 4)

                    transforms[part] = mat

        scan = np.fromfile(
            f"{KITTIEnv.samples_path}/{frame_name}.bin", dtype=np.float32
        )
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        in_x = (min_x < points[:, 0]) & (points[:, 0] < max_x)
        in_y = (min_y < points[:, 1]) & (points[:, 1] < max_y)
        in_z = (min_z < points[:, 2]) & (points[:, 2] < max_z)
        points = points[in_x & in_y & in_z]
        points = np.hstack([points, np.ones((len(points), 1))])
        raw_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))

        # See Equation (7) in Geiger et al. (2013).
        P2 = transforms["P2"]
        R0_rect = np.eye(4)
        R0_rect[:3, :3] = transforms["R0_rect"]
        T_v2c = np.eye(4)
        T_v2c[:3, :4] = transforms["Tr_velo_to_cam"]
        proj = (P2 @ R0_rect @ T_v2c @ points.T).T

        img = Image.open(f"{KITTIEnv.raw_backgrounds_path}/image_2/{frame_name}.png")

        in_front = proj[:, 2] > 0
        uvs = proj / proj[:, 2:3]
        uvs[:, 2] = proj[:, 2]
        in_width = (0 < uvs[:, 0]) & (uvs[:, 0] < img.size[0])
        in_height = (0 < uvs[:, 1]) & (uvs[:, 1] < img.size[1])
        keep = in_front & in_width & in_height
        points = points[keep, :3]
        uvs = uvs[keep]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        R0_rect = R0_rect[:3, :3]
        R_v2c = T_v2c[:3, :3]
        t_v2c = T_v2c[:3, 3]
        bboxes = []
        bbox_idxs = []
        bbox_fs = []
        (has_ped, has_cyc) = (False, False)
        with open(f"{KITTIEnv.raw_backgrounds_path}/label_2/{frame_name}.txt") as f:
            ped_idx = 0
            for line in f:
                if line.startswith("Pedestrian") or line.startswith("Cyclist"):
                    parts = [float(v) for v in line.split()[1:]]
                    (h, w, l) = parts[7:10]
                    extent = np.array([w, l, h])
                    cam_center = np.array(parts[10:13])
                    velo_center = R_v2c.T @ (R0_rect[:3, :3].T @ cam_center - t_v2c)
                    # z is given for the bottom of the bounding box, but we want the
                    # middle.
                    center = velo_center + np.array([0, 0, h / 2])

                    in_x = samp_min_x < center[0] < samp_max_x
                    in_y = samp_min_y < center[1] < samp_max_y
                    if not (in_x and in_y):
                        continue

                    rot_angle = parts[13]
                    R = Rotation.from_euler("Z", rot_angle).as_matrix()

                    bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
                    bbox.color = [1, 0, 0]
                    bboxes.append(bbox)

                    bbox_idxs.append(
                        bbox.get_point_indices_within_bounding_box(pcd.points)
                    )

                    if line.startswith("Pedestrian"):
                        has_ped = True
                    else:
                        has_cyc = True
                        continue

                    labels = np.concatenate([center, extent, R.flatten()])
                    bbox_f = f"{frame_name}_{ped_idx}.npy"
                    bbox_fs.append(bbox_f)
                    ped_pcd = pcd.crop(bbox)
                    ped_idx += 1

                    if not show_pcd:
                        np.save(f"{KITTIEnv.labels_path}/{bbox_f}", labels)
                        np.save(
                            f"{KITTIEnv.npys_path}/{bbox_f}", np.array(ped_pcd.points)
                        )

        if show_pcd:
            img.show()

            plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
            plt.imshow(img)

            plt.scatter(
                uvs[:, 0], uvs[:, 1], c=uvs[:, 2], cmap="rainbow_r", alpha=0.5, s=2
            )
            plt.show()

            o3d.visualization.draw_geometries([raw_pcd] + bboxes, frame_name)
            o3d.visualization.draw_geometries([pcd] + bboxes, frame_name)

        else:
            if len(bbox_idxs) > 0:
                bbox_idxs = np.concatenate(bbox_idxs).astype("int")

            non_bbox_pcd = pcd.select_by_index(bbox_idxs, invert=True)
            np.save(
                f"{KITTIEnv.npys_path}/{frame_name}.npy", np.array(non_bbox_pcd.points)
            )

            data_type = DATA_TYPE[has_ped][has_cyc]
            metadata = {"type": data_type, "bboxes": bbox_fs}
            with open(f"{KITTIEnv.jsons_path}/{frame_name}.json", "w") as f:
                json.dump(metadata, f)


def main():
    use_all = sys.argv[1]
    assert use_all in {"yes", "no"}
    use_all = use_all == "yes"
    if use_all:
        print("Using human and machine-labeled samples.")
    else:
        print("Only using human-labeled samples.")

    prepare_object_labels_parallel(use_all)

    show_pcd = False

    if not show_pcd:
        os.makedirs(KITTIEnv.npys_path)
        os.makedirs(KITTIEnv.labels_path)
        os.makedirs(KITTIEnv.jsons_path)

    all_frames = os.listdir(KITTIEnv.samples_path)
    all_frames.sort()

    n_jobs = 1 if show_pcd else multiprocessing.cpu_count()
    frames_per_job = int(np.ceil(len(all_frames) / n_jobs))
    procs = []
    for job in range(n_jobs):
        start = job * frames_per_job
        end = start + frames_per_job
        frames = all_frames[start:end]
        proc = multiprocessing.Process(target=worker, args=(frames, show_pcd))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
