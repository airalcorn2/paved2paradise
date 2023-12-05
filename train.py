import os
import sys

# See: https://github.com/pytorch/pytorch/issues/9158#issuecomment-402358096.
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import json
import random
import shutil
import torch
import wandb

from config import config
from kitti_dataset import KITTIDataset
from kitti_env import KITTIEnv
from pointpillars import PointPillars
from torch import nn, optim
from torch.utils.data import DataLoader


def init_data_dict(jsons_path, data_dict_f):
    json_fs = os.listdir(jsons_path)
    random.shuffle(json_fs)
    train_p = config["train_p"]
    train_n = int(train_p * len(json_fs))
    train_val_fs = json_fs[:train_n]
    test_fs = json_fs[train_n:]
    data_dict = {"train": train_val_fs, "valid": test_fs}
    with open(data_dict_f, "w") as f:
        json.dump(data_dict, f)


def get_loss(model, tensors_dict, device, criterion):
    pillar_buffers = tensors_dict["pillar_buffers"].to(device)
    pillar_pixels = tensors_dict["pillar_pixels"].to(device)
    pillar_avgs = tensors_dict["pillar_avgs"].to(device)
    preds = model(pillar_buffers, pillar_avgs, pillar_pixels)
    labels = tensors_dict["tgt"].to(device)
    loss = criterion(preds, labels)

    return loss


def validate(model, valid_loader, device, criterion):
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for tensors_dict in valid_loader:
            loss = get_loss(model, tensors_dict, device, criterion)
            total_valid_loss += loss.item()

    return total_valid_loss


def train(config, model, valid_loader, train_loader, device):
    lr = config["lr"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    scaler = torch.cuda.amp.GradScaler()
    # See: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html,
    # and: https://pytorch.org/docs/stable/notes/amp_examples.html,
    # and: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/.
    use_amp = config["use_amp"]

    best_valid_loss = float("inf")
    patience = config["patience"]
    max_lr_drops = config["max_lr_drops"]
    lr_reducer = config["lr_reducer"]
    eval_every = config["eval_every"]
    no_improvement = 0
    lr_drops = 0
    lr_reductions = 0
    total_train_loss = float("inf")
    n_valid = len(valid_loader.dataset)
    n_train = 0
    for epoch in range(config["epochs"]):
        model.train()
        for idx, tensors_dict in enumerate(train_loader):
            if idx % eval_every == 0:
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        total_valid_loss = validate(
                            model, valid_loader, device, criterion
                        )
                else:
                    total_valid_loss = validate(model, valid_loader, device, criterion)

                if total_valid_loss < best_valid_loss:
                    best_valid_loss = total_valid_loss
                    no_improvement = 0
                    lr_drops = 0
                    torch.save(
                        model.state_dict(), f"{wandb.run.dir}/{KITTIEnv.best_params_f}"
                    )

                else:
                    no_improvement += 1
                    if no_improvement == patience:
                        lr_drops += 1
                        if lr_drops == max_lr_drops:
                            sys.exit()

                        no_improvement = 0
                        lr_reductions += 1
                        for g in optimizer.param_groups:
                            g["lr"] *= lr_reducer

                if n_train > 0:
                    average_train_loss = total_train_loss / n_train
                else:
                    average_train_loss = total_train_loss

                wandb.log(
                    {
                        "average_train_loss": average_train_loss,
                        "average_valid_loss": total_valid_loss / n_valid,
                        "lr_reductions": lr_reductions,
                    }
                )

                total_train_loss = 0.0
                n_train = 0
                model.train()

            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = get_loss(model, tensors_dict, device, criterion)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss = get_loss(model, tensors_dict, device, criterion)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            n_train += len(tensors_dict["tgt"])


def main():
    dataset = config["dataset"]
    assert dataset in {"baseline", "paved2paradise"}

    if dataset == "paved2paradise":
        jsons_path = KITTIEnv.final_jsons_path
        npys_path = KITTIEnv.final_npys_path
        labels_path = KITTIEnv.final_labels_path
        idxs_path = KITTIEnv.final_idxs_path
        backgrounds_path = KITTIEnv.unlevel_background_npys_path
        data_dict_f = KITTIEnv.data_dict_f

    else:
        jsons_path = KITTIEnv.jsons_path
        npys_path = KITTIEnv.npys_path
        labels_path = KITTIEnv.labels_path
        idxs_path = None
        backgrounds_path = None
        data_dict_f = KITTIEnv.kitti_data_dict_f
        # This should have been created when preparing the KITTI data.
        assert os.path.isfile(data_dict_f)

    if not os.path.isfile(data_dict_f):
        init_data_dict(jsons_path, data_dict_f)

    with open(data_dict_f) as f:
        data_dict = json.load(f)

    config["data_dict"] = data_dict
    config["model_args"]["x_range"] = KITTIEnv.x_range
    config["model_args"]["y_range"] = KITTIEnv.y_range
    config["model_args"]["z_range"] = KITTIEnv.z_range
    wandb.init(project=KITTIEnv.wandb_project, entity=KITTIEnv.entity, config=config)

    shutil.copyfile(KITTIEnv.model_f, f"{wandb.run.dir}/{KITTIEnv.model_f}")

    device = torch.device("cuda:0")
    model = PointPillars(**config["model_args"]).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")
    print(repr(model))

    dataset_args = {
        "dataset": dataset,
        "jsons_path": jsons_path,
        "npys_path": npys_path,
        "labels_path": labels_path,
        "idxs_path": idxs_path,
        "backgrounds_path": backgrounds_path,
        "json_fs": data_dict["train"],
        "prepare_pillars": model.prepare_pillars,
        "augment": True,
        "max_drop_p": config["max_drop_p"],
    }
    train_dataset = KITTIDataset(**dataset_args)
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size, True, num_workers=num_workers)
    dataset_args["json_fs"] = data_dict["valid"]
    dataset_args["augment"] = False
    valid_dataset = KITTIDataset(**dataset_args)
    valid_loader = DataLoader(valid_dataset, batch_size, num_workers=num_workers)

    train(config, model, valid_loader, train_loader, device)


if __name__ == "__main__":
    main()
