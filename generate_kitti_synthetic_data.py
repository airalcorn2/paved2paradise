import json
import multiprocessing
import os
import random
import sys
import warnings

from kitti_env import KITTIEnv
from PIL import Image
from synthetic_data_functions import *

(A, B, C) = KITTIEnv.abc
BG_IDX = -1


def level_object_scenes(label_fs):
    for label_f in label_fs:
        name = label_f.split(".npy")[0]
        if os.path.exists(f"{KITTIEnv.level_object_npys_path}/{name}_False.npy"):
            continue

        scene_pcd = o3d.io.read_point_cloud(
            f"{KITTIEnv.raw_object_pcds_path}/{name}.pcd"
        )
        scene_points = np.array(scene_pcd.points)

        # Get bounding box info.
        labels = np.load(f"{KITTIEnv.raw_object_labels_path}/{label_f}")[0]
        center = labels[1:4]
        extent = labels[4:7]
        q = labels[7:11]
        bbox_R = Rotation.from_quat(q).as_matrix()

        for mirror in KITTIEnv.mirrors:
            if mirror:
                scene_points[:, 1] = -scene_points[:, 1]
                center[1] = -center[1]
                rotvec = Rotation.from_quat(q).as_rotvec()
                rotvec[-1] = -rotvec[-1]
                bbox_R = Rotation.from_rotvec(rotvec).as_matrix()
                scene_pcd = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(scene_points)
                )

            # The object extraction step has to be before the leveling step because the
            # scene point cloud points get modified otherwise.
            bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
            object_pcd = scene_pcd.crop(bbox)
            object_points = np.array(object_pcd.points)
            assert len(object_points) > 0

            (object_R, object_z) = level_scene(
                scene_points,
                KITTIEnv.x_start_end_object,
                KITTIEnv.y_start_end_object,
                KITTIEnv.n_grid_points,
            )

            # Get object points, bounding box center, and bounding box rotation matrix
            # in leveled parking lot.
            object_t_z = np.array([0, 0, object_z])
            object_points = (object_R @ object_points.T).T - object_t_z
            # Only keep points above the ground.
            object_points = object_points[object_points[:, 2] > 0]
            out_center = object_R @ center - object_t_z
            out_bbox_R = object_R @ bbox_R
            labels = np.concatenate([out_center, extent, out_bbox_R.flatten()])

            np.save(
                f"{KITTIEnv.level_object_npys_path}/{name}_{mirror}.npy", object_points
            )
            np.save(f"{KITTIEnv.level_object_labels_path}/{name}_{mirror}.npy", labels)


def level_object_scenes_parallel():
    for path in [KITTIEnv.level_object_npys_path, KITTIEnv.level_object_labels_path]:
        if os.path.exists(path):
            print(
                f"{path} already exists. Make sure it doesn't contain out-of-date data."
            )

        os.makedirs(path, exist_ok=True)

    all_label_fs = os.listdir(KITTIEnv.raw_object_labels_path)
    all_label_fs.sort()
    n_jobs = multiprocessing.cpu_count()
    frames_per_job = int(np.ceil(len(all_label_fs) / n_jobs))
    procs = []
    for job in range(n_jobs):
        start = job * frames_per_job
        end = start + frames_per_job
        label_fs = all_label_fs[start:end]
        proc = multiprocessing.Process(target=level_object_scenes, args=(label_fs,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def init_data_dict():
    json_fs = os.listdir(KITTIEnv.jsons_path)
    random.shuffle(json_fs)
    train_p = KITTIEnv.train_p
    train_n = int(train_p * len(json_fs))
    train_val_fs = json_fs[:train_n]
    test_fs = json_fs[train_n:]
    train_n = int(train_p * len(train_val_fs))
    train_fs = train_val_fs[:train_n]
    valid_fs = train_val_fs[train_n:]
    data_dict = {"train": train_fs, "valid": valid_fs, "test": test_fs}

    with open(KITTIEnv.kitti_data_dict_f, "w") as f:
        json.dump(data_dict, f)


def level_background_scenes(json_fs):
    for json_f in json_fs:
        with open(f"{KITTIEnv.jsons_path}/{json_f}") as f:
            metadata = json.load(f)

        if (metadata["type"] == "pedestrian") or (metadata["type"] == "both"):
            continue

        name = json_f.split(".json")[0]
        if os.path.exists(f"{KITTIEnv.unlevel_background_npys_path}/{name}_False.npy"):
            continue

        background_points = np.load(f"{KITTIEnv.npys_path}/{name}.npy")
        for mirror in KITTIEnv.mirrors:
            if mirror:
                background_points[:, 1] = -background_points[:, 1]

            (background_R, background_z) = level_scene(
                background_points,
                KITTIEnv.x_start_end_background,
                KITTIEnv.y_start_end_background,
                KITTIEnv.n_grid_points,
            )
            background_R_z = np.array(list(background_R.flatten()) + [background_z])
            np.save(
                f"{KITTIEnv.unlevel_background_npys_path}/{name}_{mirror}.npy",
                background_points,
            )
            np.save(
                f"{KITTIEnv.level_background_transforms_path}/{name}_{mirror}.npy",
                background_R_z,
            )


def level_background_scenes_parallel():
    for path in [
        KITTIEnv.unlevel_background_npys_path,
        KITTIEnv.level_background_transforms_path,
    ]:
        if os.path.exists(path):
            print(
                f"{path} already exists. Make sure it doesn't contain out-of-date data."
            )

        os.makedirs(path, exist_ok=True)

    with open(KITTIEnv.kitti_data_dict_f) as f:
        data_dict = json.load(f)

    all_json_fs = data_dict["train"] + data_dict["valid"]
    all_json_fs.sort()
    n_jobs = multiprocessing.cpu_count()
    frames_per_job = int(np.ceil(len(all_json_fs) / n_jobs))
    procs = []
    for job in range(n_jobs):
        start = job * frames_per_job
        end = start + frames_per_job
        json_fs = all_json_fs[start:end]
        proc = multiprocessing.Process(target=level_background_scenes, args=(json_fs,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def load_transforms(frame_name):
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

    return transforms


def get_min_points_for_dist(dist):
    return KITTIEnv.min_prop * (A * np.exp(-B * dist) + C)


def synthesize_samples(samp_names):
    background_fs = os.listdir(KITTIEnv.unlevel_background_npys_path)
    background_fs.sort()
    object_fs = os.listdir(KITTIEnv.level_object_npys_path)
    object_fs.sort()
    (min_x, max_x) = KITTIEnv.samp_x_range
    (min_y, max_y) = KITTIEnv.samp_y_range
    p2p_env = KITTIEnv()
    human_count_props = np.load("human_count_props.npy")
    human_counts = 1 + np.arange(len(human_count_props))
    for samp_name in samp_names:
        # Randomly sample background and object.
        background_f = np.random.choice(background_fs)
        background_points = np.load(
            f"{KITTIEnv.unlevel_background_npys_path}/{background_f}"
        )
        object_idxs = np.full(len(background_points), BG_IDX)

        frame_name = background_f.split("_")[0]
        transforms = load_transforms(frame_name)
        P2 = transforms["P2"]
        R0_rect = np.eye(4)
        R0_rect[:3, :3] = transforms["R0_rect"]
        T_v2c = np.eye(4)
        T_v2c[:3, :4] = transforms["Tr_velo_to_cam"]
        img = Image.open(f"{KITTIEnv.raw_backgrounds_path}/image_2/{frame_name}.png")

        n_objects = np.random.choice(human_counts, p=human_count_props)
        samp_object_fs = np.random.choice(object_fs, n_objects, replace=False)
        (x, y) = (None, None)
        object_infos = {}
        all_bg_idxs = set(np.arange(len(background_points)))
        remaining_bg_idxs = np.arange(len(background_points))
        obj_idx = 0
        for object_f in samp_object_fs:
            # Randomly sample object position.
            if x is None:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)

            else:
                # See: https://stackoverflow.com/questions/13064912/generate-a-uniformly-random-point-within-an-annulus-ring.
                length_sq = np.random.uniform(
                    KITTIEnv.min_dist**2, KITTIEnv.max_dist**2
                )
                length = length_sq**0.5
                angle = np.random.uniform(0, 2 * np.pi)

                x = x + length * np.cos(angle)
                y = y + length * np.sin(angle)

            while True:
                (
                    occluded_object_points,
                    background_labels,
                    occluded_background_idxs,
                ) = put_object_in_background(
                    object_f, background_points, np.array([x, y]), p2p_env
                )
                # Check if bad ground spot.
                if type(occluded_object_points) == list:
                    x = np.random.uniform(min_x, max_x)
                    y = np.random.uniform(-max_y, max_y)
                    continue

                # Check if object point cloud was too occluded.
                if len(occluded_object_points) == 0:
                    if obj_idx == 0:
                        x = np.random.uniform(min_x, max_x)
                        y = np.random.uniform(min_y, max_y)
                        continue
                    else:
                        break

                # Only keep points that can be seen in the image.
                ones = np.ones((len(occluded_object_points), 1))
                hom_points = np.hstack([occluded_object_points, ones])
                proj = (P2 @ R0_rect @ T_v2c @ hom_points.T).T
                in_front = proj[:, 2] > 0
                uvs = proj / proj[:, 2:3]
                uvs[:, 2] = proj[:, 2]
                in_width = (0 < uvs[:, 0]) & (uvs[:, 0] < img.size[0])
                in_height = (0 < uvs[:, 1]) & (uvs[:, 1] < img.size[1])
                keep = in_front & in_width & in_height
                occluded_object_points = occluded_object_points[keep]

                # Check if object point cloud was too occluded.
                dist = (x**2 + y**2) ** 0.5
                if len(occluded_object_points) < get_min_points_for_dist(dist):
                    if obj_idx == 0:
                        x = np.random.uniform(min_x, max_x)
                        y = np.random.uniform(-max_y, max_y)
                        continue
                    else:
                        break

                bg_mask = np.ones(len(background_points), dtype="bool")
                bg_mask[occluded_background_idxs] = False
                # Update data for previously added objects.
                for prev_obj in list(object_infos):
                    info = object_infos[prev_obj]
                    prev_obj_idx = info["idx"]
                    prev_obj_mask = bg_mask[object_idxs == prev_obj_idx]
                    prev_obj_points = info["points"][prev_obj_mask]
                    if len(prev_obj_points) < get_min_points_for_dist(dist):
                        del object_infos[prev_obj]
                    else:
                        info["points"] = prev_obj_points

                remaining_bg_mask = bg_mask[object_idxs == BG_IDX]
                remaining_bg_idxs = remaining_bg_idxs[remaining_bg_mask]

                background_points = np.concatenate(
                    [background_points[bg_mask], occluded_object_points]
                )
                object_idxs = np.concatenate(
                    [
                        object_idxs[bg_mask],
                        np.full(len(occluded_object_points), obj_idx),
                    ]
                )

                object_infos[object_f] = {
                    "idx": obj_idx,
                    "points": occluded_object_points,
                    "labels": background_labels,
                    "position": (x, y),
                }
                obj_idx += 1

                break

        all_obj_points = []
        all_obj_labels = []
        all_obj_positions = []
        for obj_f, info in object_infos.items():
            all_obj_points.append(info["points"])
            all_obj_labels.append(info["labels"])
            all_obj_positions.append(info["position"])

        all_obj_points = np.concatenate(all_obj_points)
        np.save(f"{KITTIEnv.final_npys_path}/{samp_name}.npy", all_obj_points)
        all_obj_labels = np.stack(all_obj_labels)
        np.save(f"{KITTIEnv.final_labels_path}/{samp_name}.npy", all_obj_labels)
        final_occluded_background_idxs = all_bg_idxs - set(remaining_bg_idxs)
        np.save(
            f"{KITTIEnv.final_idxs_path}/{samp_name}.npy",
            np.array(list(final_occluded_background_idxs), dtype=int),
        )
        metadata = {
            "background": background_f,
            "objects": list(object_infos),
            "object_positions": all_obj_positions,
        }
        with open(f"{KITTIEnv.final_jsons_path}/{samp_name}.json", "w") as f:
            json.dump(metadata, f)


def synthesize_samples_parallel():
    if os.path.exists(KITTIEnv.final_npys_path):
        print(
            f"{KITTIEnv.final_npys_path} already exists. Make sure it doesn't contain out-of-date data."
        )

    os.makedirs(f"{KITTIEnv.final_npys_path}", exist_ok=True)
    os.makedirs(f"{KITTIEnv.final_labels_path}", exist_ok=True)
    os.makedirs(f"{KITTIEnv.final_idxs_path}", exist_ok=True)
    os.makedirs(f"{KITTIEnv.final_jsons_path}", exist_ok=True)

    samples = int(sys.argv[1])
    cpus = multiprocessing.cpu_count()
    try:
        n_jobs = int(sys.argv[2])
        n_jobs = min(cpus, n_jobs)
    except IndexError:
        n_jobs = cpus

    samps_per_job = int(np.ceil(samples / n_jobs))
    samples = n_jobs * samps_per_job
    zfill = len(str(samples - 1))

    # Makes it easy to continue generating samples if the script crashes.
    all_samp_names = []
    for samp in np.arange(samples):
        samp_name = str(samp).zfill(zfill)
        if not os.path.isfile(f"{KITTIEnv.final_jsons_path}/{samp_name}.json"):
            all_samp_names.append(samp_name)

    samps_per_job = int(np.ceil(len(all_samp_names) / n_jobs))
    procs = []
    for job in range(n_jobs):
        start = job * samps_per_job
        end = start + samps_per_job
        samp_names = all_samp_names[start:end]
        proc = multiprocessing.Process(target=synthesize_samples, args=(samp_names,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def main():
    warnings.filterwarnings("ignore")
    os.register_at_fork(after_in_child=np.random.seed)

    level_object_scenes_parallel()
    level_background_scenes_parallel()
    synthesize_samples_parallel()


if __name__ == "__main__":
    main()
