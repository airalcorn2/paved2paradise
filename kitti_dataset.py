import json
import numpy as np

from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch import LongTensor, Tensor

# See Section 4.3 in Lang et al. (2019) and Section 3.2.2 in Yan, Mao, and Li (2018).
ROT_BOX = np.pi / 20
T_MU_BOX = 0
T_VAR_BOX = 0.25

ROT_GLOBAL = np.pi / 4
T_MU_GLOBAL = 0
T_VAR_GLOBAL = 0.2
SCALE_GLOBAL = (0.95, 1.05)


class KITTIDataset(Dataset):
    def __init__(
        self,
        dataset,
        jsons_path,
        npys_path,
        labels_path,
        idxs_path,
        backgrounds_path,
        json_fs,
        prepare_pillars,
        augment,
        max_drop_p,
    ):
        super().__init__()
        self.dataset = dataset
        self.jsons_path = jsons_path
        self.npys_path = npys_path
        self.labels_path = labels_path
        self.idxs_path = idxs_path
        self.backgrounds_path = backgrounds_path
        self.json_fs = json_fs
        self.prepare_pillars = prepare_pillars
        self.augment = augment
        self.max_drop_p = max_drop_p
        self.scans_name = "kitti"

    def __len__(self):
        return len(self.json_fs)

    def load_points(self, idx):
        json_f = self.json_fs[idx]
        with open(f"{self.jsons_path}/{json_f}") as f:
            metadata = json.load(f)

        frame_name = json_f.split(".json")[0]
        bbox_points = []
        labels = []
        if self.dataset == "baseline":
            bbox_fs = metadata["bboxes"]
        else:
            bbox_fs = [f"{frame_name}.npy"]

        if self.augment:
            glob_angle = np.random.uniform(-ROT_GLOBAL, ROT_GLOBAL)
            glob_R = Rotation.from_euler("Z", glob_angle).as_matrix()
            glob_t = np.random.normal(T_MU_GLOBAL, T_VAR_GLOBAL**0.5, size=3)
            glob_scale = np.random.uniform(*SCALE_GLOBAL)
            mirror = np.random.random() > 0.5

        for bbox_f in bbox_fs:
            points = np.load(f"{self.npys_path}/{bbox_f}")
            bbox_labels = np.load(f"{self.labels_path}/{bbox_f}")
            if self.augment:
                center = bbox_labels[:3]
                extent = bbox_labels[3:6]
                bbox_R = bbox_labels[6:].reshape(3, 3)

                # Apply object augmentations.
                if self.max_drop_p > 0.0:
                    drop_p = np.random.uniform(0.0, self.max_drop_p)
                    ps = np.random.random(len(points))
                    drop = ps < drop_p
                    points = points[~drop]
                    if len(points) == 0:
                        continue

                angle = np.random.uniform(-ROT_BOX, ROT_BOX)
                R = Rotation.from_euler("Z", angle).as_matrix()
                points = (R @ (points - center).T).T + center
                bbox_R = R @ bbox_R

                t = np.random.normal(T_MU_BOX, T_VAR_BOX**0.5, size=3)
                points = points + t
                center = center + t

                # Apply global augmentations.
                points = glob_scale * ((glob_R @ points.T).T + glob_t)
                center = glob_scale * (glob_R @ center + glob_t)
                extent = glob_scale * extent
                bbox_R = glob_R @ bbox_R
                if mirror:
                    points[:, 1] = -points[:, 1]
                    center[1] = -center[1]
                    rotvec = Rotation.from_matrix(bbox_R).as_rotvec()
                    rotvec[-1] = -rotvec[-1]
                    bbox_R = Rotation.from_rotvec(rotvec).as_matrix()

                bbox_labels = np.concatenate([center, extent, bbox_R.flatten()])

            bbox_points.append(points)
            labels.append(bbox_labels)

        if self.dataset == "baseline":
            bg_points = np.load(f"{self.npys_path}/{frame_name}.npy")

        else:
            bg_f = metadata["background"]
            bg_points = np.load(f"{self.backgrounds_path}/{bg_f}")
            occlude_idxs = np.load(f"{self.idxs_path}/{frame_name}.npy")
            bg_mask = np.ones(len(bg_points), dtype="bool")
            bg_mask[occlude_idxs] = False
            bg_points = bg_points[bg_mask]

        if self.augment:
            # Apply global augmentations.
            bg_points = glob_scale * ((glob_R @ bg_points.T).T + glob_t)
            if mirror:
                bg_points[:, 1] = -bg_points[:, 1]

        if len(bbox_points) > 0:
            bbox_points = np.concatenate(bbox_points)
            points = np.concatenate([bbox_points, bg_points])
            labels = np.stack(labels)
        else:
            points = bg_points
            labels = np.full(15, -500)

        return (points, labels)

    def __getitem__(self, idx):
        (points, labels) = self.load_points(idx)

        (pillar_pieces, tgt) = self.prepare_pillars(points, labels)
        pillars_buffer = Tensor(pillar_pieces[0])
        pillar_pixels = LongTensor(pillar_pieces[1])
        pillar_avgs = Tensor(pillar_pieces[2])

        return {
            "pillar_buffers": pillars_buffer,
            "pillar_pixels": pillar_pixels,
            "pillar_avgs": pillar_avgs,
            "tgt": tgt,
        }


class BackgroundsDataset(Dataset):
    def __init__(self, backgrounds_path, npy_fs, prepare_pillars):
        super().__init__()
        self.backgrounds_path = backgrounds_path
        self.npy_fs = npy_fs
        self.prepare_pillars = prepare_pillars

    def __len__(self):
        return len(self.npy_fs)

    def load_points(self, idx):
        npy_f = self.npy_fs[idx]
        return (np.load(f"{self.backgrounds_path}/{npy_f}"), np.full(15, -500))

    def __getitem__(self, idx):
        (points, labels) = self.load_points(idx)

        (pillar_pieces, tgt) = self.prepare_pillars(points, labels)
        pillars_buffer = Tensor(pillar_pieces[0])
        pillar_pixels = LongTensor(pillar_pieces[1])
        pillar_avgs = Tensor(pillar_pieces[2])

        return {
            "pillar_buffers": pillars_buffer,
            "pillar_pixels": pillar_pixels,
            "pillar_avgs": pillar_avgs,
            "tgt": tgt,
        }
