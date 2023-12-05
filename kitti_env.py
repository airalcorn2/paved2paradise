import numpy as np

from p2p_env import P2PEnv


class KITTIEnv(P2PEnv):
    dataset = "kitti"

    # Defines the bounding prism (in meters) for the training data. The detection prism
    # is one meter shorter in the x and y dimensions to avoid cropping.
    # See Section 4.2 in Lang et al. (2019).
    x_range = (0, 20)
    y_range = (-10, 10)
    z_range = (-2.5, 2)

    # LiDAR sensor configurations.
    # Velodyne settings. See: https://hypertech.co.il/wp-content/uploads/2015/12/HDL-64E-Data-Sheet.pdf.
    # See also page 37 here: https://www.termocam.it/pdf/manuale-HDL-64E.pdf. The KITTI
    # dataset was collected at 10 Hz. See: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf.
    min_elev = np.deg2rad(-24.8)
    max_elev = np.deg2rad(2)
    vert_res = 64
    elevs = np.linspace(min_elev, max_elev, vert_res)
    horiz_res = 2083
    azims = np.linspace(-np.pi, np.pi, horiz_res, False)
    azim_pm = 2 * np.pi / horiz_res
    # For some lidar sensors, like the Ouster, the sensor frame is shifted relative to
    # the lidar frame. See Section 3.1 in:
    # https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2p0.pdf.
    sensor2lidar = -np.array([0, 0, 0.0])

    # Data-generating variables.
    # Whether to randomly flip the point clouds along the x-axis.
    mirror = True
    mirrors = [False, True] if mirror else [False]
    # Whether to randomly move the human point cloud in the region of interest.
    move = True
    # The number of points along one side of the grid used to estimate the ground plane.
    n_grid_points = 100
    # The range of the space used for building the leveling grid of points for object
    # scenes.
    x_start_end_object = (0, 7)
    y_start_end_object = (-3.0, 3.0)
    # The range of the space used for building the leveling grid of points for
    # background scenes.
    x_start_end_background = (6, 13)
    y_start_end_background = (-1.7, 1.7)
    # The range for randomly placing the object. Minus one to avoid cropped objects.
    # See get_kitti_stats.py for x minimum.
    samp_x_range = (2.25, x_range[1] - 1)
    samp_y_range = (y_range[0] + 1, y_range[1] - 1)
    # If an object point is within this distance of a lidar beam, the lidar beam
    # potentially hits the object.
    hit_thresh = 0.04
    # Points that are within this distance of a ray cause the point associated with the
    # ray to be occluded.
    occlude_object_thresh = 0.08
    occlude_background_thresh = 0.03
    # Numbers come from fitting an exponential function to KITTI pedestrian data.
    # See: get_kitti_stats.py.
    abc = (1445.4513596044249, 0.16880749180490645, 17.230571487821805)
    # The minimum proportion of the object that must be visible in the final scene.
    min_prop = 0.15
    # The proportion of KITTI training data that can potentially be used for
    # backgrounds.
    train_p = 0.9
    # The forward direction for the LiDAR sensor.
    fwd = np.array([1, 0, 0])
    # The maximum number of close object points to a ray that are used to estimate the
    # surface point of an object.
    neighbors = 2

    # Directory names.
    parking_lot_dir = "parking_lot"
    backgrounds_dir = "backgrounds"
    objects_dir = "objects"
    raw_dir = "raw"
    synthetic_dir = "synthetic"
    level_dir = "level"
    final_dir = "final"
    pcds_dir = "pcds"
    npys_dir = "npys"
    transforms_dir = "levels"
    raw_labels_dir = "raw_labels"
    labels_dir = "labels"
    idxs_dir = "idxs"
    jsons_dir = "jsons"

    # Dataset paths.
    datasets_root = "/home/michael/datasets"
    raw_backgrounds_path = f"{datasets_root}/KITTI/training"

    raw_objects_path = f"{datasets_root}/Paved2Paradise/{raw_dir}/parking_lot"
    raw_object_pcds_path = f"{raw_objects_path}/{pcds_dir}"

    samples_path = f"{raw_backgrounds_path}/velodyne"
    paved2paradise_path = f"{raw_backgrounds_path}/paved2paradise"
    npys_path = f"{paved2paradise_path}/{npys_dir}"
    labels_path = f"{paved2paradise_path}/{labels_dir}"
    jsons_path = f"{paved2paradise_path}/{jsons_dir}"

    # This is the root for where all of the synthetic data will be stored.
    synthetic_path = f"{raw_backgrounds_path}/{synthetic_dir}"

    raw_object_labels_path = f"{synthetic_path}/{raw_labels_dir}"

    level_path = f"{synthetic_path}/{level_dir}"

    unlevel_backgrounds_path = f"{level_path}/{backgrounds_dir}"
    unlevel_background_npys_path = f"{unlevel_backgrounds_path}/{npys_dir}"
    level_background_transforms_path = f"{unlevel_backgrounds_path}/{transforms_dir}"

    level_objects_path = f"{level_path}/{objects_dir}"
    level_object_npys_path = f"{level_objects_path}/{npys_dir}"
    level_object_labels_path = f"{level_objects_path}/{labels_dir}"

    final_path = f"{synthetic_path}/{final_dir}"
    final_npys_path = f"{final_path}/{npys_dir}"
    final_labels_path = f"{final_path}/{labels_dir}"
    final_idxs_path = f"{final_path}/{idxs_dir}"
    final_jsons_path = f"{final_path}/{jsons_dir}"

    # The Segments.ai object bounding boxes.
    labels_json_f = "AAFieldDay.json"

    # Weights & Biases stuff.
    wandb_project = "paved2paradise_kitti"
    entity = "airalcorn2"
    data_dict_f = "data_dict.json"
    kitti_data_dict_f = "data_dict_kitti.json"
    model_f = "pointpillars.py"
    config_f = "config.pydict"
    best_params_f = "best_params.pth"
    wandb_runs = "wandb_runs"
