import numpy as np
import open3d as o3d

from p2p_env import P2PEnv
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


def level_scene(points, x_start_end, y_start_end, n_grid_points):
    (x_start, x_end) = x_start_end
    in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
    (y_start, y_end) = y_start_end
    in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
    region_points = points[in_x & in_y]
    # No ground points. Bad spot.
    if len(region_points) == 0:
        return (False, False)

    # Create a grid of points.
    (x_start, x_end) = x_start_end
    xs = np.linspace(x_start, x_end, n_grid_points)
    (y_start, y_end) = y_start_end
    ys = np.linspace(y_start, y_end, n_grid_points)
    grid = np.stack(np.meshgrid(xs, ys)).transpose(1, 2, 0)
    grid_points = grid.reshape(-1, 2)
    grid_points = np.hstack([grid_points, np.zeros(len(grid_points))[:, None]])
    grid_points += np.array([0, 0, region_points[:, 2].min()])

    # For each grid point, find nearest neighbor in scene point cloud. These are our
    # "ground" points.
    nbrs = NearestNeighbors(n_neighbors=1).fit(region_points)
    (_, pcd_idxs) = nbrs.kneighbors(grid_points)
    pcd_idxs = np.unique(pcd_idxs)
    ground_points = region_points[pcd_idxs]

    # Estimate the plane for the ground points.
    X = np.hstack([ground_points[:, :2], np.ones(len(ground_points))[:, None]])
    y = ground_points[:, 2]
    coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calculate the coordinate rotation matrix for the ground plane.
    # See: https://math.stackexchange.com/a/476311/614328.
    z_axis = -np.array([coefs[0], coefs[1], -1])
    a = z_axis / np.linalg.norm(z_axis)
    b = np.array([0, 0, 1])
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + skew + skew @ skew * 1 / (1 + a[2])

    return (R, coefs[2])


def move_object_to_position(points, bbox, xy):
    object_angle = np.arctan2(bbox.center[1], bbox.center[0])
    new_angle = np.arctan2(xy[1], xy[0])
    rot_angle = new_angle - object_angle
    R = Rotation.from_euler("Z", rot_angle).as_matrix()

    xy_dist = np.linalg.norm(xy)
    bbox_dir = xy / xy_dist
    bbox_dir = np.array(list(bbox_dir) + [0])
    shift = xy_dist - np.linalg.norm(bbox.center[:2])
    new_points = (R @ points.T).T + shift * bbox_dir

    new_center = np.array(list(xy) + [bbox.center[2]])
    new_R = R @ bbox.R
    bbox = o3d.geometry.OrientedBoundingBox(new_center, new_R, bbox.extent)

    return (new_points, bbox)


def block(blockee, blockee_norms, blocker, blocker_norms, occlude_thresh):
    # Occlude blockee points with blocker points. See:
    # https://math.stackexchange.com/a/2599689/614328.

    Ds = blockee / blockee_norms[:, None]
    t_Ps = Ds @ blocker.T
    ray_dists = (blocker_norms**2 - t_Ps**2) ** 0.5
    less_thresh = ray_dists < occlude_thresh
    # Points can only be occluded by points that are closer to the sensor.
    closer_point = blockee_norms[:, None] > blocker_norms
    drop_mask = (less_thresh & closer_point).any(1)

    return drop_mask


def unocclude(new_points, env_vars: P2PEnv):
    # Empty array.
    if len(new_points) == 0:
        return new_points

    azims = env_vars.azims
    elevs = env_vars.elevs

    point_angles = np.arctan2(new_points[:, 1], new_points[:, 0])
    min_azim = point_angles.min() - env_vars.azim_pm
    max_azim = point_angles.max() + env_vars.azim_pm
    azims = azims[(min_azim < azims) & (azims < max_azim)]
    elev_azims = np.stack(np.meshgrid(-elevs, azims)).transpose(1, 2, 0).reshape(-1, 2)
    Rs = Rotation.from_euler("yz", elev_azims).as_matrix()

    Ps = new_points
    Ds = Rs @ env_vars.fwd
    t_Ps = Ds @ Ps.T
    ray_dists = (np.linalg.norm(Ps, axis=1) ** 2 - t_Ps**2) ** 0.5

    # Average closest points.
    closest_ray_dists = ray_dists.argsort(1)
    ray_dists[ray_dists >= env_vars.hit_thresh] = np.inf
    final_points = np.zeros((len(t_Ps), env_vars.neighbors, 3))
    final_dists = np.full((len(t_Ps), env_vars.neighbors), -1.0)
    total_points = np.zeros(len(t_Ps))
    idxs = np.arange(len(t_Ps))
    for neighbor in range(min(env_vars.neighbors, ray_dists.shape[1])):
        min_Ps = closest_ray_dists[:, neighbor]
        dists = ray_dists[idxs, min_Ps]
        keep = dists < env_vars.hit_thresh
        total_points[keep] += 1
        final_points[:, neighbor] = t_Ps[idxs, min_Ps][:, None] * Ds
        final_dists[:, neighbor] = dists

    keep = total_points > 0
    final_points = final_points[keep]
    final_dists = final_dists[keep]
    weights = np.zeros_like(final_dists)
    two_neighbors = (final_dists > 0).all(1)
    weights[two_neighbors] = 0.5
    weights[~two_neighbors, 0] = 1
    final_points = np.einsum("dnc,dn->dc", final_points, weights)
    two_neighbors = (weights > 0).all(1)
    very_close = (final_dists < env_vars.hit_thresh / 2).any(1)
    final_points = final_points[very_close | two_neighbors]
    final_points = final_points[~(final_points == 0).all(1)]

    return final_points


def simulate_scene(background_object_points, background_points, env_vars: P2PEnv):
    # Occlude object points with background points. See:
    # https://math.stackexchange.com/a/2599689/614328.

    # The points from the Ouster are in the "Sensor Coordinate Frame" by default, so we
    # have to shift the z coordinates.
    # See: https://static.ouster.dev/sdk-docs/python/api/client.html?highlight=xyzlut#ouster.client.XYZLut
    # and: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#sensor-coordinate-frame.
    background_object_points = background_object_points + env_vars.sensor2lidar
    background_points = background_points + env_vars.sensor2lidar

    # Only consider background points that are in the same cylindrical slice as the object
    # (+/- a few degrees).
    object_angles = np.arctan2(
        background_object_points[:, 1], background_object_points[:, 0]
    )
    (min_angle, max_angle) = (object_angles.min(), object_angles.max())
    min_angle -= env_vars.azim_pm
    max_angle += env_vars.azim_pm
    background_angles = np.arctan2(background_points[:, 1], background_points[:, 0])
    in_cyl = (min_angle < background_angles) & (background_angles < max_angle)
    in_cyl_background_points = background_points[in_cyl]

    obj_norms = np.linalg.norm(background_object_points, axis=1)
    in_cyl_norms = np.linalg.norm(in_cyl_background_points, axis=1)

    # Occlude background points with object points.
    further = in_cyl_norms >= obj_norms.min()
    further_background_points = in_cyl_background_points[further]
    further_norms = in_cyl_norms[further]
    drop_mask = block(
        further_background_points,
        further_norms,
        background_object_points,
        obj_norms,
        env_vars.occlude_background_thresh,
    )
    occluded_background_idxs = np.arange(len(background_points))[in_cyl][further][
        drop_mask
    ]

    # Get simulated point cloud based on position relative to sensor.
    unoccluded_points = unocclude(background_object_points, env_vars)

    # Occlude object points with background points.
    unoccluded_norms = np.linalg.norm(unoccluded_points, axis=1)
    closer = in_cyl_norms <= obj_norms.max()
    closer_background_points = in_cyl_background_points[closer]
    closer_norms = in_cyl_norms[closer]
    drop_mask = block(
        unoccluded_points,
        unoccluded_norms,
        closer_background_points,
        closer_norms,
        env_vars.occlude_object_thresh,
    )
    simulated_object_points = unoccluded_points[~drop_mask] - env_vars.sensor2lidar

    return (simulated_object_points.astype("float32"), occluded_background_idxs)


def put_object_in_background(object_f, background_points, new_xy, env_vars: P2PEnv):
    object_points = np.load(f"{env_vars.level_object_npys_path}/{object_f}")
    labels = np.load(f"{env_vars.level_object_labels_path}/{object_f}")

    x_start_end = env_vars.x_start_end_background
    x_length = x_start_end[1] - x_start_end[0]
    x = min(x_start_end[0], new_xy[0])
    x_start_end = (x - x_length / 2, x + x_length / 2)
    (background_R, background_z) = level_scene(
        background_points,
        x_start_end,
        env_vars.y_start_end_background,
        env_vars.n_grid_points,
    )
    if type(background_R) == bool:
        return ([], [], [])

    background_t_z = np.array([0, 0, background_z])

    # Shift object points to background level.
    object_points = object_points + background_t_z

    # Get bounding box info.
    center = labels[:3] + background_t_z
    extent = labels[3:6]
    bbox_R = labels[6:].reshape(3, 3)
    bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)

    if env_vars.move:
        # Move object points in leveled background.
        (new_object_points, new_bbox) = move_object_to_position(
            object_points, bbox, new_xy
        )
    else:
        (new_object_points, new_bbox) = (object_points, bbox)

    # Undo transformation to get object, bounding box center, and bounding box rotation
    # matrix into original background coordinate frame.
    background_object_points = (background_R.T @ new_object_points.T).T
    background_center = background_R.T @ new_bbox.center
    background_bbox_R = background_R.T @ new_bbox.R
    background_labels = np.concatenate(
        [background_center, extent, background_bbox_R.flatten()]
    )
    background_labels = background_labels.astype("float32")

    (simulated_object_points, occluded_background_idxs) = simulate_scene(
        background_object_points, background_points, env_vars
    )
    return (simulated_object_points, background_labels, occluded_background_idxs)
