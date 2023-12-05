import os
import sys

# See: https://github.com/pytorch/pytorch/issues/9158#issuecomment-402358096.
if len(sys.argv) > 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

import json
import numpy as np
import torch

from kitti_dataset import BackgroundsDataset, KITTIDataset
from kitti_env import KITTIEnv
from load_wandb_model import load_wandb_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader


def get_logits_labels(data_loader, device, config, model):
    max_detect_x = KITTIEnv.samp_x_range[1]
    max_detect_row = int(model.side_cells.item() - max_detect_x / model.cell_length)
    max_y = KITTIEnv.y_range[1]
    max_detect_y = KITTIEnv.samp_y_range[1]
    diff_y = max_y - max_detect_y
    min_detect_col = int(diff_y / model.cell_width)
    max_detect_col = int((2 * max_y - diff_y) / model.cell_width)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for tensors_dict in data_loader:
            pillar_buffers = tensors_dict["pillar_buffers"].to(device)
            pillar_pixels = tensors_dict["pillar_pixels"].to(device)
            pillar_avgs = tensors_dict["pillar_avgs"].to(device)
            if config["use_amp"]:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = model(pillar_buffers, pillar_avgs, pillar_pixels)
            else:
                preds = model(pillar_buffers, pillar_avgs, pillar_pixels)

            preds = preds[:, max_detect_row:, min_detect_col:max_detect_col]
            all_logits.append(preds.reshape(len(preds), -1).cpu().numpy())
            if "tgt" in tensors_dict:
                tgts = tensors_dict["tgt"]
                tgts = tgts[:, max_detect_row:, min_detect_col:max_detect_col]
                tgts = tgts.reshape(len(preds), -1).numpy().astype("int")
            else:
                tgts = np.zeros_like(all_logits[-1])

            all_labels.append(tgts)

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    return (all_logits, all_labels)


def calc_logit_for_prob(b0, b1, p):
    return -(b0 + np.log(1 / p - 1)) / b1


def print_metrics(clf, logits, labels, samp_idxs):
    preds = clf.predict(logits[:, None])
    tps = ((preds == 1) & (labels == 1)).sum()
    tns = ((preds == 0) & (labels == 0)).sum()
    fps = ((preds == 1) & (labels == 0)).sum()
    fns = ((preds == 0) & (labels == 1)).sum()

    print(f"FPs: {samp_idxs[(preds == 1) & (labels == 0)]}")
    print(f"FNs: {samp_idxs[(preds == 0) & (labels == 1)]}")

    print(f"TPs/FPs/TNs/FNs: {tps}/{fps}/{tns}/{fns}")

    fpr = 100 * fps / (fps + tns)
    print(f"False Positive Rate: {fpr:.2f}% ({fps}/{fps + tns})")
    fnr = 100 * fns / (fns + tps)
    print(f"False Negative Rate: {fnr:.2f}% ({fns}/{fns + tps})")

    recall = 100 * tps / (tps + fns)
    print(f"Recall: {recall:.2f}% ({tps}/{tps + fns})")
    precision = 100 * tps / (tps + fps)
    print(f"Precision: {precision:.2f}% ({tps}/{tps + fps})")

    average_precision = average_precision_score(labels, logits)
    print(f"AUC-PR: {average_precision}")

    b0 = clf.intercept_[0]
    b1 = clf.coef_[0, 0]
    print(f"Threshold: {calc_logit_for_prob(b0, b1, 0.5)}")


def run_test(valid_loader, device, config, model, test_loader, other_loader):
    (val_logits, val_labels) = get_logits_labels(valid_loader, device, config, model)
    (test_logits, test_labels) = get_logits_labels(test_loader, device, config, model)

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    # Grid classifier stats.
    flat_logits = val_logits.flatten()
    flat_labels = val_labels.flatten()
    clf = LogisticRegression(penalty=None)
    clf.fit(flat_logits[:, None], flat_labels)

    flat_logits = test_logits.flatten()
    flat_labels = test_labels.flatten()
    samp_idxs = np.arange(len(flat_logits))
    print_metrics(clf, flat_logits, flat_labels, samp_idxs)

    total_test_loss = criterion(torch.Tensor(flat_logits), torch.Tensor(flat_labels))
    test_nll = total_test_loss / len(test_loader.dataset)
    print(f"Average NLL: {test_nll}\n")

    # Not crowded grid classifier stats.
    val_ped_counts = val_labels.sum(1)
    flat_logits = val_logits[val_ped_counts < 2].flatten()
    flat_labels = val_labels[val_ped_counts < 2].flatten()
    clf = LogisticRegression(penalty=None)
    clf.fit(flat_logits[:, None], flat_labels)

    test_ped_counts = test_labels.sum(1)
    flat_logits = test_logits[test_ped_counts < 2].flatten()
    flat_labels = test_labels[test_ped_counts < 2].flatten()
    samp_idxs = samp_idxs.reshape(len(test_ped_counts), -1)[test_ped_counts < 2]
    print_metrics(clf, flat_logits, flat_labels, samp_idxs.flatten())

    total_test_loss = criterion(torch.Tensor(flat_logits), torch.Tensor(flat_labels))
    test_nll = total_test_loss / len(test_loader.dataset)
    print(f"Average NLL: {test_nll}\n")

    # Scene classifier stats.
    max_logits = val_logits.max(1)
    max_labels = val_labels.max(1)
    if other_loader != valid_loader:
        (other_logits, other_labels) = get_logits_labels(
            other_loader, device, config, model
        )
        max_logits = np.concatenate([max_logits, other_logits.max(1)])
        max_labels = np.concatenate([max_labels, other_labels.max(1)])

    clf = LogisticRegression(penalty=None)
    clf.fit(max_logits[:, None], max_labels)

    max_logits = test_logits.max(1)
    max_labels = test_labels.max(1)
    samp_idxs = np.arange(len(max_logits))
    print_metrics(clf, max_logits, max_labels, samp_idxs)

    total_test_loss = criterion(torch.Tensor(max_logits), torch.Tensor(max_labels))
    test_nll = total_test_loss / len(test_loader.dataset)
    print(f"Average NLL: {test_nll}\n")

    # Not crowded scene classifier stats.
    max_logits = val_logits[val_ped_counts < 2].max(1)
    max_labels = val_labels[val_ped_counts < 2].max(1)
    if other_loader != valid_loader:
        max_logits = np.concatenate([max_logits, other_logits.max(1)])
        max_labels = np.concatenate([max_labels, other_labels.max(1)])

    clf = LogisticRegression(penalty=None)
    clf.fit(max_logits[:, None], max_labels)

    max_logits = test_logits[test_ped_counts < 2].max(1)
    max_labels = test_labels[test_ped_counts < 2].max(1)
    samp_idxs = samp_idxs[test_ped_counts < 2]
    print_metrics(clf, max_logits, max_labels, samp_idxs)

    total_test_loss = criterion(torch.Tensor(max_logits), torch.Tensor(max_labels))
    test_nll = total_test_loss / len(test_loader.dataset)
    print(f"Average NLL: {test_nll}")


def main():
    device = torch.device("cuda:0")
    run_id = sys.argv[1]
    (model, config, root) = load_wandb_model(KITTIEnv(), run_id, device)

    if config["dataset"] == "paved2paradise":
        jsons_path = KITTIEnv.final_jsons_path
        npys_path = KITTIEnv.final_npys_path
        labels_path = KITTIEnv.final_labels_path
        idxs_path = KITTIEnv.final_idxs_path
        backgrounds_path = KITTIEnv.unlevel_background_npys_path

    else:
        jsons_path = KITTIEnv.jsons_path
        npys_path = KITTIEnv.npys_path
        labels_path = KITTIEnv.labels_path
        idxs_path = None
        backgrounds_path = None

    dataset_args = {
        "dataset": config["dataset"],
        "jsons_path": jsons_path,
        "npys_path": npys_path,
        "labels_path": labels_path,
        "idxs_path": idxs_path,
        "backgrounds_path": backgrounds_path,
        "json_fs": config["data_dict"]["valid"],
        "prepare_pillars": model.prepare_pillars,
        "augment": False,
        "max_drop_p": config["max_drop_p"],
    }
    valid_dataset = KITTIDataset(**dataset_args)
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers
    )

    with open(KITTIEnv.kitti_data_dict_f) as f:
        data_dict = json.load(f)

    dataset_args["dataset"] = "baseline"
    dataset_args["jsons_path"] = KITTIEnv.jsons_path
    dataset_args["npys_path"] = KITTIEnv.npys_path
    dataset_args["labels_path"] = KITTIEnv.labels_path
    dataset_args["json_fs"] = data_dict["test"]
    test_dataset = KITTIDataset(**dataset_args)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    if config["dataset"] == "baseline":
        other_loader = valid_loader

    else:
        # Set up pedestrian-less dataset.
        valid_background_fs = set()
        for json_f in config["data_dict"]["train"]:
            with open(f"{jsons_path}/{json_f}") as f:
                metadata = json.load(f)
                valid_background_fs.add(metadata["background"])

        dataset_args = {
            "backgrounds_path": backgrounds_path,
            "npy_fs": list(valid_background_fs),
            "prepare_pillars": model.prepare_pillars,
        }
        other_dataset = BackgroundsDataset(**dataset_args)
        other_loader = DataLoader(
            other_dataset, batch_size=batch_size, num_workers=num_workers
        )

    run_test(valid_loader, device, config, model, test_loader, other_loader)


if __name__ == "__main__":
    main()
