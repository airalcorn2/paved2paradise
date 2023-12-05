import importlib
import os
import pickle
import torch
import wandb

from p2p_env import P2PEnv


def load_wandb_model(env_vars: P2PEnv, run_id, device):
    api = wandb.Api()
    # See: https://github.com/wandb/wandb/issues/3678.
    run_path = f"{env_vars.entity}/{env_vars.wandb_project}/{run_id}"
    run = api.run(run_path)
    config = run.config
    root = f"{env_vars.wandb_runs}/{run_id}"
    os.makedirs(root, exist_ok=True)
    with open(f"{root}/{env_vars.config_f}", "wb") as f:
        pickle.dump(config, f)

    _ = wandb.restore(env_vars.model_f, run_path, root=root)
    module = env_vars.model_f.split(".")[0]
    pointpillars = importlib.import_module(".".join(root.split("/")) + f".{module}")
    model = pointpillars.PointPillars(**config["model_args"]).to(device)

    # See: https://wandb.ai/lavanyashukla/save_and_restore/reports/Saving-and-Restoring-Machine-Learning-Models-with-W-B--Vmlldzo3MDQ3Mw
    # and: https://docs.wandb.ai/guides/track/save-restore.
    weights_f = wandb.restore(env_vars.best_params_f, run_path, root=root)
    model.load_state_dict(torch.load(weights_f.name))
    model.eval()

    return (model, config, root)
