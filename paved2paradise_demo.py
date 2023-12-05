import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sys

from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

# Visualization stuff.
VMIN = 0
VMAX = 10
CMAP_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
COLORMAP = "RdYlGn"
CMAP = plt.get_cmap(COLORMAP)
BBOX_COLOR = [1, 0, 0]
PERSON_COLOR = [1, 1, 1]
BG_COLOR = [0, 0, 0]
GRID_COLOR = [0.5, 0.5, 0.5]
CYL_RADIUS = 0.05
PARAM = o3d.io.read_pinhole_camera_parameters("fig_cam.json")

# Bounding box for object point cloud.
SEGMENTS_ANNOTATION = {
    "position": {
        "x": 3.9372718334198,
        "y": -0.6516044735908508,
        "z": 0.030873477458953857,
    },
    "dimensions": {
        "x": 0.889559268951416,
        "y": 0.7547129988670349,
        "z": 1.772523045539856,
    },
    "yaw": -1.1417025337925257,
}

try:
    WHICH = sys.argv[1]
    assert WHICH in {"sem_kitti", "kitti", "orchard"}
except IndexError:
    WHICH = "orchard"

# The location of where the object will be placed in the demo.
XYS = {"orchard": (9.98, 1.19), "kitti": (11.93, -4.61), "sem_kitti": (1.53, -6.84)}
# For when the object's location is taken from user input.
PROMPT_PARTS = [
    "Enter the ",
    "x",
    "-coordinate for where the object should be placed in the background: ",
]

# If an object point is within this distance of a lidar beam, the lidar beam potentially
# hits the object.
THRESH = 0.04
# The maximum number of close object points to a ray that are used to estimate the
# surface point of an object.
NEIGHBORS = 2
# Points that are within this distance of a ray cause the point associated with the ray
# to be occluded.
OCCLUDE_HUMAN_THRESHES = {"kitti": 0.08, "sem_kitti": 0.08, "orchard": 0.04}
OCCLUDE_ORCHARD_THRESHES = {"kitti": 0.03, "sem_kitti": 0.03, "orchard": 0.03}

# The number of points along one side of the grid used to estimate the ground plane.
N_GRID_POINTS = 100
# The range of the space used for building the grid.
GRID_LENGTH = 7.0
HALF_GRID_WIDTH = 3.0
GRID_LIMITS = {
    "kitti": {"x": (6, 6 + GRID_LENGTH), "y": (-1.7, 1.7)},
    "sem_kitti": {"x": (6, 6 + GRID_LENGTH), "y": (-1.7, 1.7)},
    "orchard": {"x": (0, GRID_LENGTH), "y": (-HALF_GRID_WIDTH, HALF_GRID_WIDTH)},
}

# LiDAR sensor configurations.
FWD = np.array([1, 0, 0])
SENSORS = {
    # Velodyne settings. See: https://hypertech.co.il/wp-content/uploads/2015/12/HDL-64E-Data-Sheet.pdf.
    # See also page 37 here: https://www.termocam.it/pdf/manuale-HDL-64E.pdf. The KITTI
    # dataset was collected at 10 Hz. See: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf.
    "kitti": {"elevs": (-24.8, 2), "vert_res": 64, "horiz_res": 2083, "z": 0.0},
    "sem_kitti": {"elevs": (-24.8, 2), "vert_res": 64, "horiz_res": 2083, "z": 0.0},
    # Ouster settings. See: https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p0-os1.pdf.
    # See also: https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2p0.pdf.
    "orchard": {
        "elevs": (-22.5, 22.5),
        "vert_res": 128,
        "horiz_res": 2048,
        "z": 36.180 / 1000,
    },
}

(MIN_ELEV, MAX_ELEV) = SENSORS[WHICH]["elevs"]
MIN_ELEV = np.deg2rad(MIN_ELEV)
MAX_ELEV = np.deg2rad(MAX_ELEV)
VERT_RES = SENSORS[WHICH]["vert_res"]
ELEVS = np.linspace(MIN_ELEV, MAX_ELEV, VERT_RES)
HORIZ_RES = SENSORS[WHICH]["horiz_res"]
AZIMS = np.linspace(-np.pi, np.pi, HORIZ_RES, False)
AZIM_PM = 2 * np.pi / HORIZ_RES
SENSOR2LIDAR = -np.array([0, 0, SENSORS[WHICH]["z"]])


def color_by_dist(pcd):
    pcd_points = np.array(pcd.points)
    col_vals = np.linalg.norm(pcd_points, axis=1)
    point_colors = CMAP(CMAP_NORM(col_vals))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)


def render_geoms(geoms):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geoms:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = BG_COLOR
    vis.get_view_control().convert_from_pinhole_camera_parameters(PARAM)
    vis.run()
    # See: https://github.com/isl-org/Open3D/issues/2006#issuecomment-779971650.
    vis.destroy_window()
    del opt
    del vis


def level_scene(pcd, x_start_end, y_start_end):
    points = np.array(pcd.points)
    (x_start, x_end) = x_start_end
    in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
    (y_start, y_end) = y_start_end
    in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
    region_points = points[in_x & in_y]

    # Create a grid of points.
    xs = np.linspace(x_start, x_end, N_GRID_POINTS)
    ys = np.linspace(y_start, y_end, N_GRID_POINTS)
    grid = np.stack(np.meshgrid(xs, ys)).transpose(1, 2, 0)
    grid_points = grid.reshape(-1, 2)
    grid_points = np.hstack([grid_points, np.zeros(len(grid_points))[:, None]])
    grid_points += np.array([0, 0, region_points[:, 2].min()])
    grid_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points))
    grid_pcd.paint_uniform_color(GRID_COLOR)
    render_geoms([pcd, grid_pcd])

    # For each grid point, find nearest neighbor in scene point cloud. These are our
    # "ground" points.
    nbrs = NearestNeighbors(n_neighbors=1).fit(region_points)
    (_, pcd_idxs) = nbrs.kneighbors(grid_points)
    pcd_idxs = np.unique(pcd_idxs)
    ground_points = region_points[pcd_idxs]
    ground_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
    color_by_dist(ground_pcd)
    render_geoms([ground_pcd])

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

    # Transform the scene.
    rot_points = (R @ points.T).T
    rot_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rot_points))
    rot_pcd.colors = pcd.colors
    width = x_end - x_start
    height = y_end - y_start
    ground_plane = o3d.geometry.TriangleMesh.create_box(
        width=width, height=height, depth=0.01
    )
    center = np.array([x_start + width / 2, 0, coefs[2]])
    ground_plane.translate(center, relative=False)
    # Not level.
    render_geoms([pcd, ground_plane])
    # Pretty level!
    render_geoms([rot_pcd, ground_plane])

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
    bbox.color = BBOX_COLOR

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


def unocclude(new_points):
    # The points from the Ouster are in the "Sensor Coordinate Frame" by default, so we
    # have to shift the z coordinates.
    # See: https://static.ouster.dev/sdk-docs/python/api/client.html?highlight=xyzlut#ouster.client.XYZLut
    # and: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#sensor-coordinate-frame.
    shift_pcd = np.array([0, 0.75, 0])
    new_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(new_points + shift_pcd)
    )
    new_points = new_points + SENSOR2LIDAR

    point_angles = np.arctan2(new_points[:, 1], new_points[:, 0])
    min_azim = point_angles.min() - AZIM_PM
    max_azim = point_angles.max() + AZIM_PM
    azims = AZIMS[(min_azim < AZIMS) & (AZIMS < max_azim)]
    elev_azims = np.stack(np.meshgrid(-ELEVS, azims)).transpose(1, 2, 0).reshape(-1, 2)
    Rs = Rotation.from_euler("yz", elev_azims).as_matrix()

    Ps = new_points
    Ds = Rs @ FWD
    t_Ps = Ds @ Ps.T
    ray_dists = (np.linalg.norm(Ps, axis=1) ** 2 - t_Ps**2) ** 0.5

    # Average closest points.
    closest_ray_dists = ray_dists.argsort(1)
    ray_dists[ray_dists >= THRESH] = np.inf
    final_points = np.zeros((len(t_Ps), NEIGHBORS, 3))
    final_dists = np.full((len(t_Ps), NEIGHBORS), -1.0)
    total_points = np.zeros(len(t_Ps))
    idxs = np.arange(len(t_Ps))
    for neighbor in range(NEIGHBORS):
        min_Ps = closest_ray_dists[:, neighbor]
        dists = ray_dists[idxs, min_Ps]
        keep = dists < THRESH
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
    very_close = (final_dists < THRESH / 2).any(1)
    final_points = final_points[very_close | two_neighbors]
    final_points = final_points[~(final_points == 0).all(1)]
    final_points = final_points - SENSOR2LIDAR

    final_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(new_points - shift_pcd)
    )
    o3d.visualization.draw_geometries([new_pcd, final_pcd], "Simulated LiDAR Points")

    return final_points


def simulate_scene(background_object_pcd, background_pcd):
    # The points from the Ouster are in the "Sensor Coordinate Frame" by default, so we
    # have to shift the z coordinates.
    # See: https://static.ouster.dev/sdk-docs/python/api/client.html?highlight=xyzlut#ouster.client.XYZLut
    # and: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#sensor-coordinate-frame.
    background_object_points = np.array(background_object_pcd.points) + SENSOR2LIDAR
    background_points = np.array(background_pcd.points) + SENSOR2LIDAR

    # Only consider background points that are in the same cylindrical slice as the
    # object (+/- a few degrees).
    object_angles = np.arctan2(
        background_object_points[:, 1], background_object_points[:, 0]
    )
    (min_angle, max_angle) = (object_angles.min(), object_angles.max())
    min_angle -= AZIM_PM
    max_angle += AZIM_PM
    background_angles = np.arctan2(background_points[:, 1], background_points[:, 0])
    in_cyl = (min_angle < background_angles) & (background_angles < max_angle)
    in_cyl_background_points = background_points[in_cyl]
    in_cyl_background_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(in_cyl_background_points)
    )
    color_by_dist(in_cyl_background_pcd)
    render_geoms([in_cyl_background_pcd, background_object_pcd])

    hum_norms = np.linalg.norm(background_object_points, axis=1)
    in_cyl_norms = np.linalg.norm(in_cyl_background_points, axis=1)

    # Occlude background points with object points.
    further = in_cyl_norms >= hum_norms.min()
    further_background_points = in_cyl_background_points[further]
    further_norms = in_cyl_norms[further]
    drop_mask = block(
        further_background_points,
        further_norms,
        background_object_points,
        hum_norms,
        OCCLUDE_ORCHARD_THRESHES[WHICH],
    )
    occluded_background_idxs = np.arange(len(background_points))[in_cyl][further][
        drop_mask
    ]
    keep_mask = np.ones(len(background_points), dtype="bool")
    keep_mask[occluded_background_idxs] = False
    occluded_background_points = background_points[keep_mask] - SENSOR2LIDAR

    # Get simulated point cloud based on position relative to sensor.
    unoccluded_points = unocclude(background_object_points)

    # Occlude object points with background points.
    unoccluded_norms = np.linalg.norm(unoccluded_points, axis=1)
    closer = in_cyl_norms <= hum_norms.max()
    closer_background_points = in_cyl_background_points[closer]
    closer_norms = in_cyl_norms[closer]
    drop_mask = block(
        unoccluded_points,
        unoccluded_norms,
        closer_background_points,
        closer_norms,
        OCCLUDE_HUMAN_THRESHES[WHICH],
    )
    simulated_object_points = unoccluded_points[~drop_mask] - SENSOR2LIDAR

    background_angles = np.arctan2(
        occluded_background_points[:, 1], occluded_background_points[:, 0]
    )
    in_cyl = (min_angle < background_angles) & (background_angles < max_angle)
    in_cyl_occluded_background_points = occluded_background_points[in_cyl]
    in_cyl_occluded_background_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(in_cyl_occluded_background_points)
    )
    color_by_dist(in_cyl_occluded_background_pcd)
    simulated_object_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(simulated_object_points)
    )
    color_by_dist(simulated_object_pcd)
    render_geoms([in_cyl_occluded_background_pcd, simulated_object_pcd])

    occluded_background_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(occluded_background_points)
    )
    color_by_dist(occluded_background_pcd)

    return (simulated_object_pcd, occluded_background_pcd)


def scene_demo():
    # Load the object scene point cloud ("68X8LC_0525.pcd").
    pcd = o3d.io.read_point_cloud("parking_lot.pcd")

    # Load the background point cloud.
    if WHICH == "kitti":
        background_pcd = o3d.io.read_point_cloud("kitti.pcd")

    elif WHICH == "sem_kitti":
        background_pcd = o3d.io.read_point_cloud("sem_kitti.pcd")

    else:
        # "run1_LIDAR_OS1-128_FRONT_1692808396.7093253_0800.pcd"
        background_pcd = o3d.io.read_point_cloud("orchard.pcd")

    color_by_dist(pcd)

    # Convert the Segmemts.ai annotation into an Open3D bounding box.
    center = []
    extent = []
    for xyz in ["x", "y", "z"]:
        center.append(SEGMENTS_ANNOTATION["position"][xyz])
        extent.append(SEGMENTS_ANNOTATION["dimensions"][xyz])

    center = np.array(center)
    extent = np.array(extent)
    bbox_R = Rotation.from_euler("Z", SEGMENTS_ANNOTATION["yaw"]).as_matrix()
    bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
    bbox.color = BBOX_COLOR

    # Render the object scene point cloud and bounding box.
    render_geoms([pcd, bbox])

    color_by_dist(background_pcd)
    render_geoms([background_pcd])

    # Naively combine the object point cloud and the background point cloud.
    object_pcd = pcd.crop(bbox)
    render_geoms([object_pcd])
    object_pcd.paint_uniform_color(PERSON_COLOR)
    render_geoms([object_pcd, background_pcd])

    object_points = np.array(object_pcd.points)
    if WHICH == "orchard":
        # Rotate object point cloud. Human's feet look like they're underground.
        angle = np.pi / 2.25
        R = Rotation.from_euler("Z", angle).as_matrix()
        new_xy = (R @ center)[:2]
        (rot_object_points, _) = move_object_to_position(object_points, bbox, new_xy)
        rot_object_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(rot_object_points)
        )
        rot_object_pcd.paint_uniform_color(PERSON_COLOR)
        render_geoms([rot_object_pcd, pcd])
        render_geoms([rot_object_pcd, background_pcd])

    use_prompt = False
    if use_prompt:
        # Select point in the background to place object.
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window()
        vis.add_geometry(background_pcd)
        vis.run()
        vis.destroy_window()

        prompt_parts = [
            "Enter the ",
            "x",
            "-coordinate for where the object should be placed in the background: ",
        ]
        x = float(input("".join(prompt_parts)))
        prompt_parts[1] = "y"
        y = float(input("".join(prompt_parts)))

    else:
        if WHICH == "orchard":
            (x, y) = (9.98, 1.19)
        elif WHICH == "kitti":
            (x, y) = (11.93, -4.61)
        else:
            (x, y) = (1.53, -6.84)

    if WHICH != "orchard":
        # Rotate the background scene so that the object will be placed in the middle.
        scene_R = Rotation.from_euler("z", -np.arctan2(y, x)).as_matrix()
        background_points = (scene_R @ np.array(background_pcd.points).T).T
        background_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(background_points)
        )
        color_by_dist(background_pcd)
        new_xy = scene_R[:2, :2] @ np.array([x, y])
        (x, y) = tuple(new_xy)

    # Estimate the plane for the ground points.
    (object_R, object_z) = level_scene(
        pcd, GRID_LIMITS["orchard"]["x"], GRID_LIMITS["orchard"]["y"]
    )
    if WHICH != "orchard":
        x_start_end = GRID_LIMITS[WHICH]["x"]
        x_length = x_start_end[1] - x_start_end[0]
        x_start_end = (x - x_length / 2, x + x_length / 2)
        (background_R, background_z) = level_scene(
            background_pcd, x_start_end, GRID_LIMITS[WHICH]["y"]
        )

    else:
        (background_R, background_z) = level_scene(
            background_pcd, GRID_LIMITS[WHICH]["x"], GRID_LIMITS[WHICH]["y"]
        )

    # Get object points in leveled object scene.
    leveled_object_points = (object_R @ object_points.T).T
    keep = leveled_object_points[:, 2] > object_z
    leveled_object_points = leveled_object_points[keep]
    leveled_object_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(leveled_object_points)
    )
    leveled_bbox = o3d.geometry.OrientedBoundingBox(
        object_R @ center, object_R @ bbox_R, extent
    )
    color_by_dist(leveled_object_pcd)
    render_geoms([leveled_object_pcd])

    if WHICH == "orchard":
        # Move object points in leveled object scene.
        (moved_object_points, moved_bbox) = move_object_to_position(
            leveled_object_points, leveled_bbox, new_xy
        )

        # Undo transformation to get object into original object scene coordinate frame.
        lot_object_points = (object_R.T @ moved_object_points.T).T
        # Rotated object is no longer underground.
        lot_object_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(lot_object_points)
        )
        lot_object_pcd.paint_uniform_color(PERSON_COLOR)
        render_geoms([lot_object_pcd, pcd])

    # Move object points to final position.
    t_z = np.array([0, 0, background_z - object_z])
    leveled_object_points = leveled_object_points + t_z
    new_xy = np.array([x, y])
    (new_object_points, new_bbox) = move_object_to_position(
        leveled_object_points, leveled_bbox, new_xy
    )
    new_object_points = new_object_points - t_z
    new_object_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(new_object_points)
    )
    color_by_dist(new_object_pcd)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(CYL_RADIUS)
    cyl.translate(new_bbox.center, relative=False)
    cyl.paint_uniform_color(BBOX_COLOR)
    render_geoms([new_object_pcd, cyl, leveled_object_pcd])

    # Undo transformation to get object into original background coordinate frame.
    background_object_points = new_object_points + t_z
    background_object_points = (background_R.T @ background_object_points.T).T
    background_object_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(background_object_points)
    )
    background_object_pcd.paint_uniform_color(PERSON_COLOR)
    render_geoms([background_object_pcd, background_pcd])

    # Undo transformation to get bounding box center and rotation matrix into original
    # background coordinate frame.
    background_center = new_bbox.center + t_z
    background_center = background_R.T @ background_center
    background_bbox_R = background_R.T @ new_bbox.R
    background_bbox = o3d.geometry.OrientedBoundingBox(
        background_center, background_bbox_R, extent
    )
    background_bbox.color = BBOX_COLOR

    # Unoccluded render.
    unoccluded_object_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(background_object_points)
    )
    color_by_dist(unoccluded_object_pcd)
    render_geoms([unoccluded_object_pcd, background_pcd])

    # Simulate scene.
    (simulated_object_pcd, occluded_background_pcd) = simulate_scene(
        background_object_pcd, background_pcd
    )
    simulated_object_points = np.array(simulated_object_pcd.points)

    if WHICH != "orchard":
        # Rotate scene back to original orientation.
        simulated_object_points = (scene_R.T @ simulated_object_points.T).T
        occluded_background_points = np.array(occluded_background_pcd.points)
        occluded_background_points = (scene_R.T @ occluded_background_points.T).T
        occluded_background_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(occluded_background_points)
        )
        color_by_dist(occluded_background_pcd)
        background_center = scene_R.T @ background_bbox.center
        background_bbox_R = scene_R.T @ background_bbox.R
        final_background_bbox = o3d.geometry.OrientedBoundingBox(
            background_center, background_bbox_R, extent
        )
        final_background_bbox.color = BBOX_COLOR

    else:
        final_background_bbox = background_bbox

    simulated_object_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(simulated_object_points)
    )
    color_by_dist(simulated_object_pcd)

    # Final render.
    geoms = [simulated_object_pcd, occluded_background_pcd, final_background_bbox]
    use_camera_viewpoint = False
    lookat = np.zeros(3)
    up = np.array([0, 0, 1])
    front = -np.array(list(background_center[:2]) + [0])
    front /= np.linalg.norm(front)
    zoom = 0.0001
    if use_camera_viewpoint:
        o3d.visualization.draw_geometries(
            geoms,
            lookat=lookat,
            up=up,
            front=front,
            zoom=zoom,
        )

    else:
        render_geoms(geoms)


def perspective_demo():
    cone = o3d.geometry.TriangleMesh.create_cone()
    cone.compute_vertex_normals()
    R = Rotation.from_euler("Y", np.pi / 2).as_matrix()
    cone.rotate(R)
    cone.translate(np.array([4, 0, 0]), relative=True)

    elev_azims = np.stack(np.meshgrid(-ELEVS, AZIMS)).transpose(1, 2, 0).reshape(-1, 2)
    Rs = Rotation.from_euler("yz", elev_azims).as_matrix()
    directions = Rs @ FWD
    origins = np.zeros_like(directions)
    rays = np.concatenate([origins, directions], 1)
    rays = o3d.core.Tensor(rays.astype("float32"), dtype=o3d.core.Dtype.Float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cone))
    ans = scene.cast_rays(rays)
    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=origins[0])
    pcd = pcd.to_legacy()
    o3d.visualization.draw_geometries([axes, pcd, cone])

    cone.translate(np.array([-4, 4, 0]), relative=True)
    points = np.array(pcd.points) + np.array([-4, 4, 0])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([axes, pcd, cone])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cone))
    ans = scene.cast_rays(rays)
    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    pcd = pcd.to_legacy()
    o3d.visualization.draw_geometries([axes, pcd, cone])

    cone = o3d.geometry.TriangleMesh.create_cone()
    cone.compute_vertex_normals()
    R = Rotation.from_euler("Y", np.pi / 2).as_matrix()
    cone.rotate(R)
    cone.translate(np.array([4, 0, 0]), relative=True)
    R = Rotation.from_euler("Z", np.pi / 2).as_matrix()
    cone.rotate(R, center=np.zeros(3))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cone))
    ans = scene.cast_rays(rays)
    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    pcd = pcd.to_legacy()
    o3d.visualization.draw_geometries([axes, pcd, cone])


if __name__ == "__main__":
    scene_demo()
