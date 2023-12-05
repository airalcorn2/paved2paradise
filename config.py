# See Sections 3.2 and 4.2 for most of the hyperparameters. The car, pedestrian, and
# cyclist each only had a single anchor.
(C, S) = (64, 1)
config = {
    "dataset": "paved2paradise",
    "train_p": 0.995,
    "num_workers": 8,
    "batch_size": 8,
    "max_drop_p": 0.2,
    "model_args": {
        "position_encodings": 0,
        "mlp_layers": [C],
        "block_layers": [
            (1, 4, C, 2 * C),
            (2, 6, 2 * C, 2 * C),
            (2, 6, 4 * C, 2 * C),
        ],
        "side_cells": 200,
        "max_pillars": 12000,
        "max_points": 100,
    },
    "lr": 1e-3,
    "patience": 5,
    "max_lr_drops": 2,
    "lr_reducer": 0.5,
    "epochs": 650,
    "use_amp": True,
    "eval_every": 50000 // 8,
}
