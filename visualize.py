import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import sys
import torch

from kitti_dataset import KITTIDataset
from kitti_env import KITTIEnv
from load_wandb_model import load_wandb_model

BG_COLOR = np.zeros(3)
BBOX_COLOR = [1, 0, 0]
(VMIN, VMAX) = (0.0, 10.0)
COLORMAP = "RdYlGn"
CMAP = plt.get_cmap(COLORMAP)
CMAP_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
CYL_RADIUS = 0.05
SHOW_KITTI_GTS = True


def load_scan(dataset, idx, vis):
    sys.stdout.write("\033[2K\033[1G")

    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.clear_geometries()

    points = dataset.load_points(idx)
    if type(points) == tuple:
        points = points[0]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    dists = np.linalg.norm(points, axis=1)
    colors = CMAP(CMAP_NORM(dists))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    json_f = dataset.json_fs[idx]
    with open(f"{dataset.jsons_path}/{json_f}") as f:
        metadata = json.load(f)

    centers = []
    for bbox_f in metadata["bboxes"]:
        bbox_labels = np.load(f"{dataset.labels_path}/{bbox_f}")
        centers.append(f"({bbox_labels[0]:.2f}, {bbox_labels[1]:.2f})")
        if SHOW_KITTI_GTS:
            cyl = o3d.geometry.TriangleMesh.create_cylinder(CYL_RADIUS)
            cyl.translate(bbox_labels[:3], relative=False)
            cyl.paint_uniform_color([0, 0, 1])
            vis.add_geometry(cyl)

    samp_name = json_f.split(".json")[0]
    n_bboxes = len(metadata["bboxes"])
    centers_str = " - " + "/".join(centers) if len(centers) > 0 else ""
    term_width = os.get_terminal_size().columns
    print(f"\r{samp_name}: {idx}/{n_bboxes}{centers_str}"[:term_width], end="")

    vis.get_view_control().convert_from_pinhole_camera_parameters(param)
    opt = vis.get_render_option()
    opt.background_color = BG_COLOR


def add_detections(vis, centers, z):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    centers = centers.cpu().numpy()
    centers = np.hstack([centers, np.full((len(centers), 1), z)])
    for center in centers:
        cyl = o3d.geometry.TriangleMesh.create_cylinder(CYL_RADIUS)
        cyl.translate(center, relative=False)
        cyl.paint_uniform_color(BBOX_COLOR)
        vis.add_geometry(cyl)

    vis.get_view_control().convert_from_pinhole_camera_parameters(param)


def vis_seq():
    run_id = sys.argv[1]
    device = torch.device("cuda:0")

    (model, config, root) = load_wandb_model(KITTIEnv, run_id, device)

    with open(KITTIEnv.kitti_data_dict_f) as f:
        data_dict = json.load(f)

    dataset_args = {
        "dataset": "baseline",
        "jsons_path": KITTIEnv.jsons_path,
        "npys_path": KITTIEnv.npys_path,
        "labels_path": KITTIEnv.labels_path,
        "idxs_path": None,
        "backgrounds_path": None,
        "json_fs": data_dict["test"],
        "prepare_pillars": model.prepare_pillars,
        "augment": False,
        "max_drop_p": 0.0,
    }
    eval_dataset = KITTIDataset(**dataset_args)

    min_prob = 0.3
    min_logit = torch.Tensor([-np.log(1 / min_prob - 1)]).to(device)
    z = -0.6

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(eval_dataset.scans_name)
    idx = 0
    max_idx = len(eval_dataset) - 1
    load_scan(eval_dataset, idx, vis)
    vis.reset_view_point(True)

    def next_scan(vis):
        nonlocal idx
        idx = min(max_idx, idx + 1)
        load_scan(eval_dataset, idx, vis)

    def prev_scan(vis):
        nonlocal idx
        idx = max(0, idx - 1)
        load_scan(eval_dataset, idx, vis)

    def go_to_scan(vis):
        nonlocal idx
        try:
            sys.stdout.write("\033[2K\033[1G")
            resp = input(f"Enter a scan index from 0 to {max_idx}: ")
            cand_idx = int(resp)
            if not (0 <= cand_idx <= max_idx):
                raise IndexError
            else:
                idx = cand_idx

            sys.stdout.write("\033[1A\033[2K\033[1G")
            load_scan(eval_dataset, idx, vis)

        except ValueError:
            sys.stdout.write("\033[1A\033[2K\033[1G")
            print("\rInput was not an integer.", end="")
        except IndexError:
            sys.stdout.write("\033[1A\033[2K\033[1G")
            print(f"\rInput was not in the range 0 to {max_idx}.", end="")

    def change_threshold(vis):
        nonlocal min_logit
        try:
            sys.stdout.write("\033[2K\033[1G")
            resp = input(f"Enter a detection threshold between 0.0 and 1.0: ")
            cand_prob = float(resp)
            if not (0 < cand_prob < 1.0):
                raise ValueError
            else:
                min_logit = -np.log(1 / cand_prob - 1)
                min_logit = torch.Tensor([min_logit]).to(device)

            sys.stdout.write("\033[1A\033[2K\033[1G")
            load_scan(eval_dataset, idx, vis)

        except ValueError:
            sys.stdout.write("\033[1A\033[2K\033[1G")
            print("\rInput was not a number between 0.0 and 1.0.", end="")

    def toggle_show(vis):
        global SHOW_KITTI_GTS
        SHOW_KITTI_GTS = not SHOW_KITTI_GTS
        load_scan(eval_dataset, idx, vis)

    def make_prediction(vis):
        load_scan(eval_dataset, idx, vis)
        points = eval_dataset.load_points(idx)[0]
        with torch.no_grad():
            if config["use_amp"]:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    detections = model.get_detections(points, device, min_logit)
            else:
                detections = model.get_detections(points, device, min_logit)

            add_detections(vis, detections, z)

            centers = []
            for detection in detections:
                centers.append(f"({detection[0]:.2f}, {detection[1]:.2f})")

            sys.stdout.write("\033[2K\033[1G")
            centers_str = " - " + "/".join(centers) if len(centers) > 0 else ""
            out_str = f"{len(centers)}{centers_str}"
            term_width = os.get_terminal_size().columns
            print(f"\rDetections: {out_str}"[:term_width], end="")

    # See: https://www.glfw.org/docs/latest/group__keys.html.
    # Left arrow.
    vis.register_key_callback(262, next_scan)
    # Right arrow.
    vis.register_key_callback(263, prev_scan)
    # Up arrow.
    vis.register_key_callback(265, make_prediction)
    # "G" key.
    vis.register_key_callback(71, go_to_scan)
    # "C" key.
    vis.register_key_callback(67, change_threshold)
    # "S" key.
    vis.register_key_callback(83, toggle_show)
    vis.poll_events()
    vis.run()


def main():
    vis_seq()
    sys.stdout.write("\033[2K\033[1G")


if __name__ == "__main__":
    print("Right arrow: Next frame.")
    print("Left arrow: Previous frame.")
    print("Up arrow: Show pedestrian detections.")
    print("G: Go to a specific frame.")
    print("C: Change the detection threshold.")
    print("S: Toggle showing the KITTI ground truth labels.")
    main()
