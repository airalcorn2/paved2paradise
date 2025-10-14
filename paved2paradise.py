import copy
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

from lidar2intrinsics import lidar2intrinsics
from numba import float64, int64, njit
from open3d.visualization import gui, rendering
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

# Visualization stuff.
VMIN = 0
VMAX = 12
CMAP_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
COLORMAP = "RdYlGn"
CMAP = plt.get_cmap(COLORMAP)
BBOX_COLOR = [1, 0, 0]
GRID_COLOR = [0.5, 0.5, 0.5]
CYL_RADIUS = 0.05
PCD_SUFFIXES = {"pcd", "npy"}

SENSOR_IDX2SENSOR = ["lidar_front"]
DEFAULT_OBJ_F = "./human_colored.pcd"
DEFAULT_BG_F = "./orchard_colored.pcd"


def color_by_dist(pcd):
    pcd_points = np.array(pcd.points)
    col_vals = np.linalg.norm(pcd_points, axis=1)
    point_colors = CMAP(CMAP_NORM(col_vals))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)


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


def unlevel(moved_object_points, bg_t, bg_R, moved_bbox):
    moved_object_points += bg_t
    moved_object_points = (bg_R.T @ moved_object_points.T).T
    moved_center = bg_R.T @ (moved_bbox.center + bg_t)
    moved_bbox_R = bg_R.T @ moved_bbox.R

    return (moved_object_points, moved_center, moved_bbox_R)


def unocclude(
    object_points,
    optical_center,
    azim_pm,
    elev_pm,
    D_elevs,
    max_sector_angle,
    D_azims,
    all_Ds,
    hit_thresh,
    max_neighbors,
):
    # The points from the Ouster are in the "Sensor Coordinate Frame" by default, so
    # we have to translate the points relative to the sensor's origin.
    # See: https://static.ouster.dev/sdk-docs/python/api/client.html#ouster.sdk.client.XYZLut
    # and: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#sensor-coordinate-frame.
    object_points[:, :3] -= optical_center

    point_azims = np.arctan2(object_points[:, 1], object_points[:, 0])
    min_azim = point_azims.min() - azim_pm
    max_azim = point_azims.max() + azim_pm

    point_norms = np.linalg.norm(object_points[:, :3], axis=1)
    point_elevs = np.arcsin(object_points[:, 2] / point_norms)
    min_elev = point_elevs.min() - elev_pm
    max_elev = point_elevs.max() + elev_pm
    elev_mask = (min_elev < D_elevs) & (D_elevs < max_elev)

    all_final_points = []
    max_sector_angle = np.radians(max_sector_angle)
    for start_angle in np.arange(min_azim, max_azim, max_sector_angle):
        # Only consider beams that are in the same cylindrical slice (+/- a few
        # degrees).
        end_angle = min(max_azim, start_angle + max_sector_angle)
        azim_mask = (start_angle < D_azims) & (D_azims < end_angle)
        Ds = all_Ds[azim_mask & elev_mask]
        if len(Ds) == 0:
            continue

        start_angle -= azim_pm
        end_angle += azim_pm

        in_slice = (start_angle < point_azims) & (point_azims < end_angle)
        Ps = object_points[in_slice, :3]
        t_Ps = Ds @ Ps.T
        ray_dists = (np.linalg.norm(Ps, axis=1) ** 2 - t_Ps**2) ** 0.5

        # Average closest points.
        closest_ray_dists = ray_dists.argsort(1)
        ray_dists[ray_dists >= hit_thresh] = np.inf
        final_points = np.zeros((len(t_Ps), max_neighbors, object_points.shape[1]))
        final_dists = np.full((len(t_Ps), max_neighbors), -1.0)
        total_points = np.zeros(len(t_Ps))
        idxs = np.arange(len(t_Ps))
        # Sometimes there's only a single object point in the sector.
        neighbors = min(max_neighbors, closest_ray_dists.shape[1])
        for neighbor in range(neighbors):
            min_Ps = closest_ray_dists[:, neighbor]
            dists = ray_dists[idxs, min_Ps]
            keep = dists < hit_thresh
            total_points[keep] += 1
            final_points[:, neighbor, :3] = t_Ps[idxs, min_Ps][:, None] * Ds
            if object_points.shape[1] > 3:
                final_points[:, neighbor, 3:] = object_points[in_slice][min_Ps, 3:]

            final_dists[:, neighbor] = dists

        keep = total_points > 0
        final_points = final_points[keep]
        if len(final_points) == 0:
            continue

        final_dists = final_dists[keep]
        weights = np.zeros_like(final_dists)
        n_neighbors = (final_dists > 0).sum(1)
        for neighbor in range(1, n_neighbors.max() + 1):
            weights[n_neighbors == neighbor] = 1 / neighbor

        final_points = np.einsum("dnc,dn->dc", final_points, weights)
        very_close = (final_dists < hit_thresh / 2).any(1)
        final_points = final_points[very_close | (n_neighbors > 1)]
        final_points = final_points[~(final_points == 0).all(1)]
        all_final_points.append(final_points)

    if len(all_final_points) == 0:
        return np.empty((0, object_points.shape[1]))
    else:
        all_final_points = np.concatenate(all_final_points)
        all_final_points[:, :3] += optical_center
        return all_final_points


@njit(
    # range_image
    float64[:, :, :](
        float64[:, :, :],  # range_image
        float64[:, :],  # points
        int64,  # height
        int64,  # width
        float64[:, :],  # rot_mat
        float64[:],  # optical_center
        float64[:],  # beams
        float64,  # bg_min_dist
        int64,  # obj_idx
    )
)
def project_points(
    range_image,
    points,
    height,
    width,
    rot_mat,
    optical_center,
    beams,
    bg_min_dist,
    obj_idx,
):
    has_color = range_image.shape[2] > 5
    (op_x, op_y, op_z) = optical_center
    for point in points:
        (x, y, z) = point[:3]
        # Transform the point to the LiDAR coordinate frame.
        og_x = rot_mat[0, 0] * x + rot_mat[0, 1] * y + rot_mat[0, 2] * z - op_x
        og_y = rot_mat[1, 0] * x + rot_mat[1, 1] * y + rot_mat[1, 2] * z - op_y
        og_z = rot_mat[2, 0] * x + rot_mat[2, 1] * y + rot_mat[2, 2] * z - op_z

        # See Equation (1) in: "Rethinking Range View Representation for LiDAR
        # Segmentation".
        # The coordinate frame has the x-axis pointing forward, the y-axis pointing
        # left, and the z-axis pointing up.

        # arctan2 gives the azimuth angle in the range [-pi, pi]. Dividing by pi scales
        # the azimuth angle to the range [-1, 1]. Subtracting from one makes it so -pi
        # maps to two and pi maps to zero, i.e., positive angles will be on the left
        # side of the image and negative angles will be on the right side of the image,
        # which matches what would be seen in the real world. Multiplying by 0.5 scales
        # the azimuth angle to the range [0, 1], and multiplying by width scales the
        # azimuth angle to the range [0, width].
        u = round(0.5 * (1 - np.arctan2(og_y, og_x) / np.pi) * width)
        if u < 0:
            u = 0
        elif u > width - 1:
            u = width - 1

        # arcsin gives the elevation angle in the range [-pi/2, pi/2]. Subtracting
        # min_elev shifts the range so min_elev maps to zero and max_elev maps to
        # elev_fov. Dividing by elev_fov scales the elevation angle to the range [0, 1].
        # Subtracting from one puts the higher elevations at the top of the image, which
        # matches what would be seen in the real world. Multiplying by height scales the
        # elevation angle to the range [0, height].
        depth = np.sqrt(og_x**2 + og_y**2 + og_z**2)
        angle = np.arcsin(og_z / depth)
        v = np.argmin(np.abs(angle - beams))
        if v < 0:
            v = 0
        elif v > height - 1:
            v = height - 1

        old_depth = range_image[v, u, 3]
        if (old_depth < bg_min_dist) or (depth < old_depth):
            range_image[v, u, 0] = obj_idx
            range_image[v, u, 1] = x
            range_image[v, u, 2] = y
            range_image[v, u, 3] = z
            range_image[v, u, 4] = depth
            if has_color:
                range_image[v, u, 5:] = point[3:]

    return range_image


class Paved2Paradise:
    def __init__(self):
        self.w = None
        self.theme = None
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.comb_mat = rendering.MaterialRecord()
        self.comb_mat.shader = "defaultUnlit"
        # See: https://github.com/isl-org/Open3D/issues/4921#issuecomment-1096845864.
        self.bbox_mat = rendering.MaterialRecord()
        self.bbox_mat.shader = "unlitLine"
        self.bbox_mat.line_width = 2
        self.obj_window = None
        self.bg_window = None
        self.comb_window = None

        self._settings_panel = None

        self.pcds = {}
        self.obj_pcd = None
        self.obj_bbox = None
        self.level_obj_bbox = None
        self.bg_pcd = None
        self.lidar_R = None
        self.Rs = None
        self.Ds = None
        self.D_azims = None
        self.D_elevs = None
        self.sensor2lidar = None
        self.optical_center = None
        self.beams = None
        self.azim_pm = None
        self.elev_pm = None
        self.height = None
        self.width = None
        self.pixel_shift_by_row = None
        self.obj_x_range = (0, 13)
        self.obj_y_range = (-5.625, 5.625)
        self.bg_x_range = (-1, 9)
        self.bg_y_range = (-9, 9)

        self.new_obj_xy = np.array([5.79, -1.00])
        self.obj_xy = None
        self.hit_thresh = 0.025
        self.bg_min_dist = 0.1
        self.max_sector_angle = 2.25
        self.neighbors = 2
        self.use_colors = True
        self.show_level_pcds = {"obj": True, "bg": True}
        self.level_pcds = {}
        self.level_at_location = True
        self.use_ground_z = True
        self.center_ground_ball = None
        self.local_t = None
        self.bg_t = None
        self.bg_R = None
        self.occlude = True
        self.sim_lidar = True
        self.show_new_obj_loc = True
        self.show_lidar = False
        self.show_beams = False
        self.prev_n_beams = 0
        self.start_beam_idx = 0
        self.end_beam_idx = 0
        self.point_size = 6.0

        self.grid_pcds = {}
        self.ground_planes = {}
        self.ground_pcds = {}
        self.level_ground_pcds = {}
        self.grid_points = 100
        self.grids_info = {
            "obj": {"length": 7, "width": 6},
            "bg": {"length": 7, "width": 6},
        }
        self.show_grid_pcds = {"obj": False, "bg": False}
        self.show_ground_planes = {"obj": False, "bg": False}
        self.show_ground_pcds = {"obj": False, "bg": False}

        self.final_pcd = o3d.geometry.PointCloud()
        self.simulated_object_points = None
        self.close_img = None
        self.bg_range_image = None

    def run(self):
        app = gui.Application.instance
        app.initialize()

        self.w = w = app.create_window("Paved2Paradise", 1024, 768)
        self.theme = w.theme

        self.obj_window = gui.SceneWidget()
        self.obj_window.scene = rendering.Open3DScene(w.renderer)
        self.obj_window.scene.set_background(np.zeros(4))
        self.obj_window.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0)
        )
        self.obj_window.scene.view.set_post_processing(False)
        w.add_child(self.obj_window)

        self.bg_window = gui.SceneWidget()
        self.bg_window.scene = rendering.Open3DScene(w.renderer)
        self.bg_window.scene.set_background(np.zeros(4))
        self.bg_window.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0)
        )
        self.bg_window.scene.view.set_post_processing(False)
        w.add_child(self.bg_window)

        self.comb_window = gui.SceneWidget()
        self.comb_window.scene = rendering.Open3DScene(w.renderer)
        self.comb_window.scene.set_background(np.zeros(4))
        self.comb_window.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0)
        )
        self.comb_window.scene.view.set_post_processing(False)
        w.add_child(self.comb_window)

        em = w.theme.font_size
        self._settings_panel = gui.ScrollableVert()

        pcd_loaders = gui.CollapsableVert(
            "Input Scenes", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )
        self._settings_panel.add_child(pcd_loaders)

        obj_scene_button = gui.Button("Object Scene")
        obj_scene_button.set_on_clicked(self._on_open_object_scene)

        bg_scene_button = gui.Button("Background Scene")
        bg_scene_button.set_on_clicked(self._on_open_background_scene)

        sensor_label = gui.Label("Sensor Selector")
        sensor_selector = gui.RadioButton(gui.RadioButton.Type.VERT)
        sensor_selector.set_items(SENSOR_IDX2SENSOR)
        sensor_selector.set_on_selection_changed(self._on_sensor_changed)

        obj_x_range_label = gui.Label("Object x Range (m)")
        obj_x_range = gui.TextEdit()
        obj_x_range.set_on_value_changed(self._on_obj_x_range_changed)
        obj_x_range.text_value = str(self.obj_x_range)

        obj_y_range_label = gui.Label("Object y Range (m)")
        obj_y_range = gui.TextEdit()
        obj_y_range.set_on_value_changed(self._on_obj_y_range_changed)
        obj_y_range.text_value = str(self.obj_y_range)

        bg_x_range_label = gui.Label("Background x Range (m)")
        bg_x_range = gui.TextEdit()
        bg_x_range.set_on_value_changed(self._on_bg_x_range_changed)
        bg_x_range.text_value = str(self.bg_x_range)

        bg_y_range_label = gui.Label("Background y Range (m)")
        bg_y_range = gui.TextEdit()
        bg_y_range.set_on_value_changed(self._on_bg_y_range_changed)
        bg_y_range.text_value = str(self.bg_y_range)

        v = gui.Vert(0.25 * em)
        v.add_child(obj_scene_button)
        v.add_child(bg_scene_button)
        v.add_child(sensor_label)
        v.add_child(sensor_selector)
        v.add_child(obj_x_range_label)
        v.add_child(obj_x_range)
        v.add_child(obj_y_range_label)
        v.add_child(obj_y_range)
        v.add_child(bg_x_range_label)
        v.add_child(bg_x_range)
        v.add_child(bg_y_range_label)
        v.add_child(bg_y_range)
        pcd_loaders.add_child(v)

        final_scene = gui.CollapsableVert(
            "Final Scene", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )
        self._settings_panel.add_child(final_scene)

        show_range_image_button = gui.Button("Show Range Images")
        show_range_image_button.set_on_clicked(self._on_show_range_image)

        obj_xy_label = gui.Label("Object (x, y)-coordinates (m)")
        self.obj_xy = gui.TextEdit()
        self.obj_xy.set_on_value_changed(self._on_obj_xy_changed)
        self.obj_xy.text_value = str(tuple(self.new_obj_xy))

        hit_label = gui.Label("Hit Threshold (m)")
        hit_thresh = gui.TextEdit()
        hit_thresh.set_on_value_changed(self._on_hit_thresh_changed)
        hit_thresh.text_value = str(self.hit_thresh)

        bg_min_dist_label = gui.Label("Background Min. Distance (m)")
        bg_min_dist = gui.TextEdit()
        bg_min_dist.set_on_value_changed(self._on_bg_min_dist_changed)
        bg_min_dist.text_value = str(self.bg_min_dist)

        max_sector_label = gui.Label("Maximum Sector (°)")
        max_sector = gui.TextEdit()
        max_sector.set_on_value_changed(self._on_max_sector_changed)
        max_sector.text_value = str(self.max_sector_angle)

        neighbors_label = gui.Label("Neighbors")
        neighbors = gui.TextEdit()
        neighbors.set_on_value_changed(self._on_neighbors_changed)
        neighbors.text_value = str(self.neighbors)

        point_size_label = gui.Label("Point Size")
        point_size = gui.TextEdit()
        point_size.set_on_value_changed(self._on_point_size_changed)
        point_size.text_value = str(self.point_size)
        self.mat.point_size = self.point_size
        self.comb_mat.point_size = self.point_size

        use_colors = gui.Checkbox("Use Colors")
        use_colors.set_on_checked(self._on_use_colors)
        use_colors.checked = self.use_colors

        level_obj = gui.Checkbox("Level Object")
        level_obj.set_on_checked(self._on_level_object)
        level_obj.checked = self.show_level_pcds["obj"]

        level_bg = gui.Checkbox("Level Background")
        level_bg.set_on_checked(self._on_level_background)
        level_bg.checked = self.show_level_pcds["bg"]

        level_loc = gui.Checkbox("Level at Object Location")
        level_loc.set_on_checked(self._on_level_location)
        level_loc.checked = self.level_at_location

        ground_z_check = gui.Checkbox("Use Local Ground z")
        ground_z_check.set_on_checked(self._on_use_ground_z)
        ground_z_check.checked = self.use_ground_z

        occlude = gui.Checkbox("Occlude")
        occlude.set_on_checked(self._on_occlude)
        occlude.checked = self.occlude

        sim_lidar = gui.Checkbox("Simulate LiDAR")
        sim_lidar.set_on_checked(self._on_sim_lidar)
        sim_lidar.checked = self.sim_lidar

        show_new_obj_loc = gui.Checkbox("New Object Location")
        show_new_obj_loc.set_on_checked(self._on_show_new_obj_loc)
        show_new_obj_loc.checked = self.show_new_obj_loc

        show_lidar = gui.Checkbox("Show LiDAR Cones")
        show_lidar.set_on_checked(self._on_show_lidar)

        show_beams = gui.Checkbox("Show Beams")
        show_beams.set_on_checked(self._on_show_beams)

        beam_range_label = gui.Label("Beam Index Range")
        beam_range = gui.TextEdit()
        beam_range.set_on_value_changed(self._on_beam_range_changed)
        beam_range.text_value = str(f"({self.start_beam_idx}, {self.end_beam_idx})")

        v = gui.Vert(0.25 * em)
        v.add_child(show_range_image_button)
        v.add_child(obj_xy_label)
        v.add_child(self.obj_xy)
        v.add_child(hit_label)
        v.add_child(hit_thresh)
        v.add_child(bg_min_dist_label)
        v.add_child(bg_min_dist)
        v.add_child(max_sector_label)
        v.add_child(max_sector)
        v.add_child(neighbors_label)
        v.add_child(neighbors)
        v.add_child(point_size_label)
        v.add_child(point_size)
        v.add_child(use_colors)
        v.add_child(level_obj)
        v.add_child(level_bg)
        v.add_child(level_loc)
        v.add_child(ground_z_check)
        v.add_child(occlude)
        v.add_child(sim_lidar)
        v.add_child(show_new_obj_loc)
        v.add_child(show_lidar)
        v.add_child(show_beams)
        v.add_child(beam_range_label)
        v.add_child(beam_range)
        final_scene.add_child(v)

        leveling = gui.CollapsableVert("Leveling", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._settings_panel.add_child(leveling)

        grid_points_label = gui.Label("Grid Points")
        grid_points = gui.TextEdit()
        grid_points.set_on_value_changed(self._on_grid_points_changed)
        grid_points.text_value = str(self.grid_points)

        obj_grid_length_label = gui.Label("Object Grid Length (m)")
        obj_grid_length = gui.TextEdit()
        obj_grid_length.set_on_value_changed(self._on_obj_grid_length_changed)
        obj_grid_length.text_value = str(self.grids_info["obj"]["length"])

        obj_grid_width_label = gui.Label("Object Grid Width (m)")
        obj_grid_width = gui.TextEdit()
        obj_grid_width.set_on_value_changed(self._on_obj_grid_width_changed)
        obj_grid_width.text_value = str(self.grids_info["obj"]["width"])

        bg_grid_length_label = gui.Label("Background Grid Length (m)")
        bg_grid_length = gui.TextEdit()
        bg_grid_length.set_on_value_changed(self._on_bg_grid_length_changed)
        bg_grid_length.text_value = str(self.grids_info["bg"]["length"])

        bg_grid_width_label = gui.Label("Background Grid Width (m)")
        bg_grid_width = gui.TextEdit()
        bg_grid_width.set_on_value_changed(self._on_bg_grid_width_changed)
        bg_grid_width.text_value = str(self.grids_info["bg"]["width"])

        obj_grid_points_check = gui.Checkbox("Object Grid")
        obj_grid_points_check.set_on_checked(self._on_show_obj_grid_points)

        bg_grid_points_check = gui.Checkbox("Background Grid")
        bg_grid_points_check.set_on_checked(self._on_show_bg_grid_points)

        obj_ground_plane_check = gui.Checkbox("Object Ground Plane")
        obj_ground_plane_check.set_on_checked(self._on_show_obj_ground_plane)

        bg_ground_plane_check = gui.Checkbox("Background Ground Plane")
        bg_ground_plane_check.set_on_checked(self._on_show_bg_ground_plane)

        obj_ground_points = gui.Checkbox("Object Ground Points")
        obj_ground_points.set_on_checked(self._on_show_obj_ground_points)

        bg_ground_points = gui.Checkbox("Background Ground Points")
        bg_ground_points.set_on_checked(self._on_show_bg_ground_points)

        v = gui.Vert(0.25 * em)
        v.add_child(grid_points_label)
        v.add_child(grid_points)
        v.add_child(obj_grid_length_label)
        v.add_child(obj_grid_length)
        v.add_child(obj_grid_width_label)
        v.add_child(obj_grid_width)
        v.add_child(bg_grid_length_label)
        v.add_child(bg_grid_length)
        v.add_child(bg_grid_width_label)
        v.add_child(bg_grid_width)
        v.add_child(obj_grid_points_check)
        v.add_child(obj_ground_plane_check)
        v.add_child(obj_ground_points)
        v.add_child(bg_grid_points_check)
        v.add_child(bg_ground_plane_check)
        v.add_child(bg_ground_points)
        leveling.add_child(v)

        w.add_child(self._settings_panel)
        w.set_on_layout(self._on_layout)

        self.update_sensor_params(SENSOR_IDX2SENSOR[0])
        self._on_load_object_dialog_done(DEFAULT_OBJ_F)
        self._on_load_background_dialog_done(DEFAULT_BG_F)

        self.comb_window.set_on_mouse(self._on_mouse_widget3d)

        app.run()

    def _on_mouse_widget3d(self, event):
        # See: https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/mouse_and_point_coord.py.
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN) and (
            event.is_modifier_down(gui.KeyModifier.ALT)
            or (event.buttons == int(gui.MouseButton.RIGHT))
        ):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the window, but
                # to dereference the image correctly we need them relative to the origin
                # of the widget. Note that even if the scene widget is the only thing in
                # the window, so if a menubar exists, it also takes up space in the
                # window (except on macOS).
                x = event.x - self.comb_window.frame.x
                y = event.y - self.comb_window.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                # Clicked on nothing (i.e., the far plane).
                if depth == 1.0:
                    pass
                else:
                    world = self.comb_window.scene.camera.unproject(
                        x,
                        y,
                        depth,
                        self.comb_window.frame.width,
                        self.comb_window.frame.height,
                    )
                    if event.buttons == int(gui.MouseButton.RIGHT):
                        print(world)
                    else:
                        obj_xy_text = f"({world[0]:.2f}, {world[1]:.2f})"
                        self.obj_xy.text_value = obj_xy_text
                        self._on_obj_xy_changed(self.obj_xy.text_value)

            self.comb_window.scene.scene.render_to_depth_image(depth_callback)

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_layout(self, layout_context):
        r = self.w.content_rect
        settings_width = 17 * layout_context.theme.font_size
        display_width = r.width - settings_width
        self.obj_window.frame = gui.Rect(r.x, r.y, display_width / 3, r.height)
        self.bg_window.frame = gui.Rect(
            r.x + display_width / 3 + 1, r.y, display_width / 3, r.height
        )
        self.comb_window.frame = gui.Rect(
            r.x + 2 * (display_width / 3 + 1), r.y, display_width / 3, r.height
        )
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel.frame = gui.Rect(
            r.get_right() - settings_width, r.y, settings_width, height
        )

    def _on_open_object_scene(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose object scene.", self.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_object_dialog_done)
        self.w.show_dialog(dlg)

    def _on_open_background_scene(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose background scene.", self.theme
        )
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_background_dialog_done)
        self.w.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.w.close_dialog()

    def render_obj_scene(self):
        self.obj_window.scene.clear_geometry()
        if self.show_ground_pcds["obj"]:
            if self.show_level_pcds["obj"]:
                pcd = self.level_ground_pcds["obj"]
            else:
                pcd = self.ground_pcds["obj"]
        else:
            if self.show_level_pcds["obj"]:
                pcd = self.level_pcds["obj"]
                bbox = self.level_obj_bbox
            else:
                pcd = self.pcds["obj"]
                bbox = self.obj_bbox

            self.obj_window.scene.add_geometry("Bounding Box", bbox, self.bbox_mat)

        self.obj_window.scene.add_geometry("Points", pcd, self.mat)

        if self.show_grid_pcds["obj"]:
            self.obj_window.scene.add_geometry(
                "Grid Points", self.grid_pcds["obj"], self.mat
            )

        if self.show_ground_planes["obj"]:
            self.obj_window.scene.add_geometry(
                "Ground Plane", self.ground_planes["obj"], self.mat
            )

    def _on_load_object_dialog_done(self, pcd_f):
        self.w.close_dialog()
        suffix = pcd_f.split(".")[-1]
        if suffix not in PCD_SUFFIXES:
            print("Object Scene must be a .pcd or .npy file.")
            return

        path = os.path.dirname(pcd_f)
        name = os.path.basename(pcd_f).split(".pcd")[0]
        try:
            with open(f"{path}/{name}.json") as f:
                bbox = json.load(f)

            # Convert the Segmemts.ai annotation into an Open3D bounding box.
            center = []
            extent = []
            for xyz in ["x", "y", "z"]:
                center.append(bbox["position"][xyz])
                extent.append(bbox["dimensions"][xyz])

            center = np.array(center)
            extent = np.array(extent)
            bbox_R = Rotation.from_euler("Z", bbox["yaw"]).as_matrix()

        except FileNotFoundError:
            bbox = np.load(f"{path}/{name}.npy")
            center = bbox[1:4]
            extent = bbox[4:7]
            bbox_R = Rotation.from_euler("Z", bbox[7]).as_matrix()

        self.obj_bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
        self.obj_bbox.color = BBOX_COLOR

        self.obj_pcd = o3d.io.read_point_cloud(pcd_f)
        self._on_obj_xy_range_changed()
        self.simulate_scene()

    def render_bg_scene(self):
        self.bg_window.scene.clear_geometry()
        if self.show_level_pcds["bg"]:
            if self.show_ground_pcds["bg"]:
                pcd = self.level_ground_pcds["bg"]
            else:
                pcd = self.level_pcds["bg"]
        else:
            if self.show_ground_pcds["bg"]:
                pcd = self.ground_pcds["bg"]
            else:
                pcd = self.pcds["bg"]

        self.bg_window.scene.add_geometry("Points", pcd, self.mat)

        if self.show_grid_pcds["bg"]:
            self.bg_window.scene.add_geometry(
                "Grid Points", self.grid_pcds["bg"], self.mat
            )

        if self.show_ground_planes["bg"]:
            self.bg_window.scene.add_geometry(
                "Ground Plane", self.ground_planes["bg"], self.mat
            )

        if self.show_new_obj_loc:
            cyl = o3d.geometry.TriangleMesh.create_cylinder(CYL_RADIUS)
            if self.show_level_pcds["bg"]:
                z = self.level_obj_bbox.center[2]
            else:
                z = self.obj_bbox.center[2]

            center = np.array([self.new_obj_xy[0], self.new_obj_xy[1], z])
            cyl.translate(center, relative=False)
            cyl.paint_uniform_color(BBOX_COLOR)
            self.bg_window.scene.add_geometry("New Object Location", cyl, self.mat)

    def _on_load_background_dialog_done(self, pcd_f):
        self.w.close_dialog()
        suffix = pcd_f.split(".")[-1]
        if suffix not in PCD_SUFFIXES:
            print("Background Scene must be a .pcd or .npy file.")
            return

        if suffix == "pcd":
            self.bg_pcd = o3d.io.read_point_cloud(pcd_f)
        else:
            self.bg_pcd = o3d.geometry.PointCloud()
            self.bg_pcd.points = o3d.utility.Vector3dVector(np.load(pcd_f))

        assert len(self.bg_pcd.points) == self.height * self.width
        points = np.array(self.bg_pcd.points)
        points = points.reshape(self.height, self.width, 3)

        depths = np.linalg.norm(points, axis=2)
        staggered = 255 * depths / depths.max()
        img = Image.fromarray(staggered.astype("uint8"))
        img.show()

        destaggered_points = np.zeros_like(points)
        for v in range(points.shape[0]):
            destaggered_points[v] = np.roll(
                points[v], self.pixel_shift_by_row[v], axis=0
            )

        destaggered_depths = np.linalg.norm(destaggered_points, axis=2)
        self.bg_range_image = np.concatenate(
            [destaggered_points, destaggered_depths[..., None]], axis=2
        )

        destaggered = 255 * destaggered_depths / destaggered_depths.max()
        img = Image.fromarray(destaggered.astype("uint8"))
        img.show()

        if self.bg_pcd.has_colors():
            colors = np.array(self.bg_pcd.colors)
            colors = colors.reshape(self.height, self.width, 3)
            destaggered_colors = np.zeros_like(colors)
            for v in range(colors.shape[0]):
                destaggered_colors[v] = np.roll(
                    colors[v], self.pixel_shift_by_row[v], axis=0
                )

            self.bg_range_image = np.concatenate(
                [self.bg_range_image, destaggered_colors], axis=2
            )
            img = Image.fromarray((255 * destaggered_colors).astype("uint8"))
            img.show()

        in_x = (self.bg_x_range[0] < destaggered_points[..., 0]) & (
            destaggered_points[..., 0] < self.bg_x_range[1]
        )
        in_y = (self.bg_y_range[0] < destaggered_points[..., 1]) & (
            destaggered_points[..., 1] < self.bg_y_range[1]
        )
        destaggered_depths[~(in_x & in_y)] = 0
        self.bg_range_image[~(in_x & in_y)] = 0
        destaggered = 255 * destaggered_depths / destaggered_depths.max()
        img = Image.fromarray(destaggered.astype("uint8"))
        img.show()

        self.close_img = img

        dims = 1 + self.bg_range_image.shape[2]
        bg_range_image = np.zeros((self.height, self.width, dims), dtype="float64")
        bg_range_image[..., 1:] = self.bg_range_image
        self.bg_range_image = bg_range_image

        self._on_bg_xy_range_changed()
        self.simulate_scene()

    def _on_dialog_ok(self):
        self.w.close_dialog()

    def _create_dialog(self, title, message):
        dlg = gui.Dialog(title)

        em = self.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        ok_button = gui.Button("OK")
        ok_button.set_on_clicked(self._on_dialog_ok)

        dlg_layout.add_child(ok_button)
        dlg.add_child(dlg_layout)
        self.w.show_dialog(dlg)

    def get_Ds(self):
        azims = np.linspace(-np.pi, np.pi, self.width, False)
        (min_elev, max_elev) = (self.beams.min(), self.beams.max())
        elev_azims = np.stack(np.meshgrid(-self.beams, azims))
        elev_azims = elev_azims.transpose(1, 2, 0).reshape(-1, 2)
        self.Rs = self.lidar_R @ Rotation.from_euler("yz", elev_azims).as_matrix()
        self.Ds = self.Rs @ np.array([1, 0, 0])
        self.D_azims = np.arctan2(self.Ds[:, 1], self.Ds[:, 0])
        self.azim_pm = 2 * np.pi / self.width
        self.D_elevs = np.arcsin(self.Ds[:, 2] / np.linalg.norm(self.Ds, axis=1))
        self.elev_pm = (max_elev - min_elev) / self.height

    def update_sensor_params(self, sensor):
        intrinsics = lidar2intrinsics[sensor]
        quat = intrinsics["sensor_model_rotation"]
        quat = np.array([quat["x"], quat["y"], quat["z"], quat["w"]])
        self.lidar_R = Rotation.from_quat(quat).as_matrix()
        self.beams = np.radians(intrinsics["beam_altitude_angles"])
        sensor2lidar = intrinsics["lidar_to_sensor_transform"][11] / 1000
        self.sensor2lidar = np.array([0, 0, sensor2lidar])
        self.optical_center = self.lidar_R @ self.sensor2lidar
        self.height = intrinsics["pixels_per_column"]
        self.width = intrinsics["columns_per_frame"]
        self.pixel_shift_by_row = intrinsics["pixel_shift_by_row"]
        self.get_Ds()

    def _on_sensor_changed(self, sensor_idx):
        self.update_sensor_params(SENSOR_IDX2SENSOR[sensor_idx])
        self.simulate_scene()

    def _on_show_range_image(self):
        self.create_range_images()

    def _on_obj_xy_range_changed(self):
        points = np.array(self.obj_pcd.points)
        in_x = (self.obj_x_range[0] < points[:, 0]) & (
            points[:, 0] < self.obj_x_range[1]
        )
        in_y = (self.obj_y_range[0] < points[:, 1]) & (
            points[:, 1] < self.obj_y_range[1]
        )
        obj_pcd = self.obj_pcd.select_by_index(np.where(in_x & in_y)[0])
        self.pcds["obj"] = obj_pcd
        if not obj_pcd.has_colors():
            color_by_dist(obj_pcd)

        self.obj_window.scene.clear_geometry()
        self.obj_window.scene.add_geometry("Points", obj_pcd, self.mat)
        if self.obj_window.scene.camera.get_field_of_view() == 90.0:
            bbox = self.obj_window.scene.bounding_box
            self.obj_window.setup_camera(60.0, bbox, bbox.get_center())

        self.create_grid_points("obj")
        self.create_ground_plane("obj")
        self.level_scene("obj")
        self.render_obj_scene()
        self.simulate_scene()

    def _on_obj_x_range_changed(self, obj_x_range):
        try:
            self.obj_x_range = eval(obj_x_range)
            self._on_obj_xy_range_changed()
        except:
            self._create_dialog(
                "Input Error", "Object x-range must be a tuple of floats."
            )
            return

    def _on_obj_y_range_changed(self, obj_y_range):
        try:
            self.obj_y_range = eval(obj_y_range)
            self._on_obj_xy_range_changed()
        except:
            self._create_dialog(
                "Input Error", "Object y-range must be a tuple of floats."
            )
            return

    def _on_bg_xy_range_changed(self):
        points = np.array(self.bg_pcd.points)
        in_x = (self.bg_x_range[0] < points[:, 0]) & (points[:, 0] < self.bg_x_range[1])
        in_y = (self.bg_y_range[0] < points[:, 1]) & (points[:, 1] < self.bg_y_range[1])
        bg_pcd = self.bg_pcd.select_by_index(np.where(in_x & in_y)[0])
        self.pcds["bg"] = bg_pcd
        if not bg_pcd.has_colors():
            color_by_dist(bg_pcd)

        self.bg_window.scene.clear_geometry()
        self.bg_window.scene.add_geometry("Points", bg_pcd, self.mat)
        if self.bg_window.scene.camera.get_field_of_view() == 90.0:
            bbox = self.bg_window.scene.bounding_box
            self.bg_window.setup_camera(60.0, bbox, bbox.get_center())

        self.create_grid_points("bg")
        self.create_ground_plane("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_bg_x_range_changed(self, bg_x_range):
        try:
            self.bg_x_range = eval(bg_x_range)
            self._on_bg_xy_range_changed()
        except:
            self._create_dialog(
                "Input Error", "Background x-range must be a tuple of floats."
            )
            return

    def _on_bg_y_range_changed(self, bg_y_range):
        try:
            self.bg_y_range = eval(bg_y_range)
            self._on_bg_xy_range_changed()
        except:
            self._create_dialog(
                "Input Error", "Background y-range must be a tuple of floats."
            )
            return

    def _on_obj_xy_changed(self, obj_xy):
        try:
            self.new_obj_xy = np.array(eval(obj_xy))
        except:
            self._create_dialog(
                "Input Error", "Object (x, y)-coordinates must be a tuple of floats."
            )
            return

        self.create_ground_plane("bg")
        self.create_grid_points("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_hit_thresh_changed(self, hit_thresh):
        try:
            self.hit_thresh = float(hit_thresh)
        except ValueError:
            self._create_dialog("Input Error", "Hit Threshold must be a float.")
            return

        self.simulate_scene()

    def _on_bg_min_dist_changed(self, bg_min_dist):
        try:
            self.bg_min_dist = float(bg_min_dist)
            self._on_bg_xy_range_changed()
        except:
            self._create_dialog(
                "Input Error", "Background min. distance must be a float."
            )
            return

    def _on_max_sector_changed(self, max_sector_angle):
        try:
            self.max_sector_angle = float(max_sector_angle)
        except ValueError:
            self._create_dialog("Input Error", "Maximum Sector must be a float.")
            return

        self.simulate_scene()

    def _on_neighbors_changed(self, neighbors):
        try:
            self.neighbors = int(neighbors)
        except ValueError:
            self._create_dialog("Input Error", "Neighbors must be an integer.")
            return

        self.simulate_scene()

    def _on_point_size_changed(self, point_size):
        try:
            self.point_size = float(point_size)
        except ValueError:
            self._create_dialog("Input Error", "Point Size must be a float.")
            return

        self.mat.point_size = self.point_size
        self.comb_mat.point_size = self.point_size
        self.render_obj_scene()
        self.render_bg_scene()
        self.simulate_scene()

    def level_scene(self, which):
        which_pcd = self.pcds[which]
        points = np.array(which_pcd.points)

        grid_x, grid_y = self.get_grid_xy(which)
        x_start = grid_x - self.grids_info[which]["length"] / 2
        x_end = x_start + self.grids_info[which]["length"]
        y_start = grid_y - self.grids_info[which]["width"] / 2
        y_end = y_start + self.grids_info[which]["width"]
        in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
        in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
        region_pcd = which_pcd.select_by_index(np.where(in_x & in_y)[0])

        # For each grid point, find nearest neighbor in scene point cloud. These are our
        # "ground" points.
        nbrs = NearestNeighbors(n_neighbors=1).fit(np.array(region_pcd.points))
        grid_points = np.array(self.grid_pcds[which].points)
        (_, pcd_idxs) = nbrs.kneighbors(grid_points)
        pcd_idxs = np.unique(pcd_idxs)
        ground_pcd = region_pcd.select_by_index(pcd_idxs)
        self.ground_pcds[which] = ground_pcd
        if not ground_pcd.has_colors():
            color_by_dist(ground_pcd)

        # Estimate the plane for the ground points.
        ground_points = np.array(ground_pcd.points)
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
        t = np.array([0, 0, -coefs[2]])
        level_points = (R @ points.T).T + t
        level_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(level_points))
        level_pcd.colors = self.pcds[which].colors
        self.level_pcds[which] = level_pcd

        # Transform the ground points.
        level_ground_points = (R @ ground_points.T).T + t
        level_ground_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(level_ground_points)
        )
        level_ground_pcd.colors = ground_pcd.colors
        self.level_ground_pcds[which] = level_ground_pcd

        if which == "obj":
            center = R @ self.obj_bbox.center + t
            bbox_R = R @ self.obj_bbox.R
            self.level_obj_bbox = o3d.geometry.OrientedBoundingBox(
                center, bbox_R, self.obj_bbox.extent
            )
            self.level_obj_bbox.color = BBOX_COLOR

        else:
            if self.use_ground_z:
                # Find the height of the ground point that is directly above the center
                # grid point.
                center = np.array([grid_x, grid_y])
                center_idx = np.argmin(
                    np.linalg.norm(ground_points[:, :2] - center, axis=1)
                )
                self.center_ground_ball = o3d.geometry.TriangleMesh.create_sphere(0.05)
                self.center_ground_ball.translate(
                    ground_points[center_idx], relative=False
                )
                self.center_ground_ball.paint_uniform_color([0, 0, 1])
                ground_z = level_ground_points[center_idx][2]
                self.local_t = np.array([0, 0, ground_z])

            self.bg_t = -t
            self.bg_R = R

    def _on_use_colors(self, use_colors):
        self.use_colors = use_colors
        self.simulate_scene()

    def _on_level_object(self, level_obj):
        self.show_level_pcds["obj"] = level_obj
        self.render_obj_scene()
        self.simulate_scene()

    def _on_level_background(self, level_bg):
        self.show_level_pcds["bg"] = level_bg
        self.render_bg_scene()
        self.simulate_scene()

    def _on_level_location(self, level_loc):
        self.level_at_location = level_loc
        self.create_ground_plane("bg")
        self.create_grid_points("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_use_ground_z(self, use_ground_z):
        self.use_ground_z = use_ground_z
        if not use_ground_z:
            self.comb_window.scene.remove_geometry("Center Ground Point")

        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_occlude(self, occlude):
        self.occlude = occlude
        self.simulate_scene()

    def _on_sim_lidar(self, sim_lidar):
        self.sim_lidar = sim_lidar
        self.simulate_scene()

    def _on_show_new_obj_loc(self, show_new_obj_loc):
        self.show_new_obj_loc = show_new_obj_loc
        self.render_bg_scene()
        self.simulate_scene()

    def _on_show_lidar(self, show_lidar):
        self.show_lidar = show_lidar
        if not show_lidar:
            self.comb_window.scene.remove_geometry("Top LiDAR Cone")
            self.comb_window.scene.remove_geometry("Bottom LiDAR Cone")

        self.simulate_scene()

    def _on_show_beams(self, show_beams):
        self.show_beams = show_beams
        if not show_beams:
            self.comb_window.scene.remove_geometry("Unoccluded Point Cloud")
            for idx in range(self.prev_n_beams):
                self.comb_window.scene.remove_geometry(f"Arrow {idx}")

        self.simulate_scene()

    def _on_beam_range_changed(self, beam_range):
        try:
            self.start_beam_idx, self.end_beam_idx = eval(beam_range)
            if (self.start_beam_idx < 0) or (self.end_beam_idx > len(self.Rs)):
                raise ValueError

            if self.start_beam_idx > self.end_beam_idx:
                raise ValueError

            is_zero = self.start_beam_idx != 0
            if (self.start_beam_idx == self.end_beam_idx) and is_zero:
                raise ValueError

        except ValueError:
            self._create_dialog(
                "Input Error", "Beam Range must be a tuple of integers."
            )
            return

        self.simulate_scene()

    def get_grid_xy(self, which):
        if which == "obj":
            grid_x = self.obj_bbox.center[0]
            grid_y = self.obj_bbox.center[1]
        else:
            if self.level_at_location:
                grid_x = self.new_obj_xy[0]
                grid_y = self.new_obj_xy[1]
            else:
                grid_x = self.grids_info[which]["length"] / 2
                grid_y = 0

        return (grid_x, grid_y)

    def create_grid_points(self, which):
        points = np.array(self.pcds[which].points)

        (grid_x, grid_y) = self.get_grid_xy(which)
        x_start = grid_x - self.grids_info[which]["length"] / 2
        x_end = x_start + self.grids_info[which]["length"]
        in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
        y_start = grid_y - self.grids_info[which]["width"] / 2
        y_end = y_start + self.grids_info[which]["width"]
        in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
        region_points = points[in_x & in_y]

        # Create a grid of points.
        xs = np.linspace(x_start, x_end, self.grid_points)
        ys = np.linspace(y_start, y_end, self.grid_points)
        grid = np.stack(np.meshgrid(xs, ys)).transpose(1, 2, 0)
        grid_points = grid.reshape(-1, 2)
        grid_points = np.hstack([grid_points, np.zeros(len(grid_points))[:, None]])
        grid_points += np.array([0, 0, region_points[:, 2].min()])
        grid_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points))
        grid_pcd.paint_uniform_color(GRID_COLOR)

        self.grid_pcds[which] = grid_pcd

    def create_ground_plane(self, which):
        grid_width = self.grids_info[which]["width"]
        grid_length = self.grids_info[which]["length"]
        ground_plane = o3d.geometry.TriangleMesh.create_box(
            width=grid_length, height=grid_width, depth=0.01
        )
        (grid_x, grid_y) = self.get_grid_xy(which)
        center = np.array([grid_x, grid_y, 0])
        ground_plane.translate(center, relative=False)

        self.ground_planes[which] = ground_plane

    def _on_grid_points_changed(self, grid_points):
        try:
            self.grid_points = int(grid_points)
        except ValueError:
            self._create_dialog("Input Error", "Grid Points must be an integer.")
            return

        for which in ["obj", "bg"]:
            self.create_grid_points(which)
            self.level_scene(which)

        self.render_obj_scene()
        self.render_bg_scene()
        self.simulate_scene()

    def _on_obj_grid_length_changed(self, grid_length):
        try:
            self.grids_info["obj"]["length"] = float(grid_length)
        except ValueError:
            self._create_dialog("Input Error", "Object Grid Length must be a float.")
            return

        self.create_grid_points("obj")
        self.create_ground_plane("obj")
        self.level_scene("obj")
        self.render_obj_scene()
        self.simulate_scene()

    def _on_obj_grid_width_changed(self, grid_width):
        try:
            self.grids_info["obj"]["width"] = float(grid_width)
        except ValueError:
            self._create_dialog("Input Error", "Object Grid Width must be a float.")
            return

        self.create_grid_points("obj")
        self.create_ground_plane("obj")
        self.level_scene("obj")
        self.render_obj_scene()
        self.simulate_scene()

    def _on_bg_grid_length_changed(self, grid_length):
        try:
            self.grids_info["bg"]["length"] = float(grid_length)
        except ValueError:
            self._create_dialog(
                "Input Error", "Background Grid Length must be a float."
            )
            return

        self.create_grid_points("bg")
        self.create_ground_plane("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_bg_grid_width_changed(self, grid_length):
        try:
            self.grids_info["bg"]["width"] = float(grid_length)
        except ValueError:
            self._create_dialog("Input Error", "Background Grid Width must be a float.")
            return

        self.create_grid_points("bg")
        self.create_ground_plane("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_show_obj_grid_points(self, show_obj_ground_points):
        self.show_grid_pcds["obj"] = show_obj_ground_points
        self.render_obj_scene()

    def _on_show_obj_ground_plane(self, show_obj_ground_plane):
        self.show_ground_planes["obj"] = show_obj_ground_plane
        self.render_obj_scene()

    def _on_show_obj_ground_points(self, show_obj_ground_points):
        self.show_ground_pcds["obj"] = show_obj_ground_points
        self.render_obj_scene()

    def _on_show_bg_grid_points(self, show_bg_ground_points):
        self.show_grid_pcds["bg"] = show_bg_ground_points
        self.render_bg_scene()

    def _on_show_bg_ground_plane(self, show_bg_ground_plane):
        self.show_ground_planes["bg"] = show_bg_ground_plane
        self.render_bg_scene()

    def _on_show_bg_ground_points(self, show_bg_ground_points):
        self.show_ground_pcds["bg"] = show_bg_ground_points
        self.render_bg_scene()

    def render_beams(self, simulated_object_points, moved_object_points):
        self.comb_window.scene.remove_geometry("Unoccluded Point Cloud")
        unfiltered_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(simulated_object_points[:, :3])
        )
        unfiltered_pcd.paint_uniform_color([1, 0, 0])
        self.comb_window.scene.add_geometry(
            "Unoccluded Point Cloud", unfiltered_pcd, self.comb_mat
        )
        range_image = np.zeros((self.height, self.width, 5), dtype="float64")
        range_image = project_points(
            range_image,
            simulated_object_points,
            self.height,
            self.width,
            self.lidar_R.T,
            self.sensor2lidar,
            self.beams,
            self.bg_min_dist,
            1,
        )[..., 4]
        image = 255 * range_image / range_image.max()
        img = Image.fromarray(image.astype("uint8"))
        img.show()

        for idx in range(self.prev_n_beams):
            self.comb_window.scene.remove_geometry(f"Arrow {idx}")

        if self.start_beam_idx == self.end_beam_idx == 0:
            point_dists = np.linalg.norm(simulated_object_points, axis=1)
            beam_length = point_dists.max()
        else:
            beam_length = max(np.abs(self.bg_x_range))
            beam_length = max(beam_length, np.abs(self.bg_y_range))

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.001,  # Radius of the shaft.
            cone_radius=0.005,  # Radius of the head.
            cylinder_height=beam_length,  # Length of the shaft.
            cone_height=0.05,  # Length of the head.
            resolution=20,  # Number of segments (smoothness).
            cylinder_split=4,
            cone_split=1,
        )
        pitch_R = Rotation.from_euler("Y", np.pi / 2).as_matrix()
        arrow.rotate(pitch_R, center=(0, 0, 0))
        if self.start_beam_idx == self.end_beam_idx == 0:
            point_azims = np.arctan2(
                moved_object_points[:, 1], moved_object_points[:, 0]
            )
            min_azim = point_azims.min() - self.azim_pm
            max_azim = point_azims.max() + self.azim_pm
            azim_mask = (min_azim < self.D_azims) & (self.D_azims < max_azim)

            point_norms = np.linalg.norm(moved_object_points[:, :3], axis=1)
            point_elevs = np.arcsin(moved_object_points[:, 2] / point_norms)
            min_elev = point_elevs.min() - self.elev_pm
            max_elev = point_elevs.max() + self.elev_pm
            elev_mask = (min_elev < self.D_elevs) & (self.D_elevs < max_elev)

            Rs = self.Rs[azim_mask & elev_mask]

        else:
            Rs = self.Rs[self.start_beam_idx : self.end_beam_idx]

        self.prev_n_beams = len(Rs)
        for idx, R in enumerate(Rs):
            arrow_copy = copy.deepcopy(arrow)
            arrow_copy.rotate(R, center=(0, 0, 0))
            arrow_copy.translate(self.optical_center, relative=True)
            self.comb_window.scene.add_geometry(
                f"Arrow {idx}", arrow_copy, self.comb_mat
            )

    def simulate_scene(self):
        if not (("obj" in self.pcds) and ("bg" in self.pcds)):
            return

        if self.show_level_pcds["obj"]:
            object_pcd = self.level_pcds["obj"]
            bbox = self.level_obj_bbox
        else:
            object_pcd = self.pcds["obj"]
            bbox = self.obj_bbox

        if self.show_level_pcds["bg"]:
            bg_t = np.copy(self.bg_t)
            bg_R = self.bg_R
        else:
            bg_t = 0
            bg_R = np.eye(3)

        if self.use_ground_z:
            bg_t += self.local_t

        object_pcd = object_pcd.crop(bbox)
        if self.show_level_pcds["obj"]:
            # Only keep points above the ground.
            object_pcd = object_pcd.select_by_index(
                np.where(np.array(object_pcd.points)[:, 2] > 0)[0]
            )

        if self.show_new_obj_loc:
            # Move object points in leveled background.
            (moved_object_points, moved_bbox) = move_object_to_position(
                np.array(object_pcd.points), bbox, self.new_obj_xy
            )

        else:
            moved_object_points = np.array(object_pcd.points)
            moved_bbox = bbox

        # Undo transformation to get object, bounding box center, and bounding box
        # rotation matrix into original background coordinate frame.
        (moved_object_points, moved_center, moved_bbox_R) = unlevel(
            moved_object_points, bg_t, bg_R, moved_bbox
        )
        moved_bbox = o3d.geometry.OrientedBoundingBox(
            moved_center, moved_bbox_R, moved_bbox.extent
        )
        moved_bbox.color = BBOX_COLOR

        both_colored = self.bg_pcd.has_colors() and self.obj_pcd.has_colors()
        if both_colored:
            obj_colors = np.array(object_pcd.colors)
            moved_object_points = np.hstack([moved_object_points, obj_colors])

        if self.sim_lidar:
            # Get simulated point cloud based on position relative to sensor.
            simulated_object_points = unocclude(
                moved_object_points,
                self.optical_center,
                self.azim_pm,
                self.elev_pm,
                self.D_elevs,
                self.max_sector_angle,
                self.D_azims,
                self.Ds,
                self.hit_thresh,
                self.neighbors,
            )
            if self.show_beams:
                self.render_beams(simulated_object_points, moved_object_points)

        else:
            simulated_object_points = moved_object_points

        if self.occlude:
            range_image = project_points(
                self.bg_range_image.copy(),
                simulated_object_points,
                self.height,
                self.width,
                self.lidar_R.T,
                self.sensor2lidar,
                self.beams,
                self.bg_min_dist,
                1,
            )
            final_points = range_image[..., 1:4].reshape(-1, 3)
            if both_colored:
                final_colors = range_image[..., 5:8].reshape(-1, 3)
                final_points = np.hstack([final_points, final_colors])

        else:
            bg_points = np.array(self.pcds["bg"].points)
            if both_colored:
                bg_colors = np.array(self.pcds["bg"].colors)
                bg_points = np.hstack([bg_points, bg_colors])

            all_points = np.concatenate([simulated_object_points, bg_points])
            final_points = all_points

        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(final_points[:, :3])
        self.final_pcd = final_pcd
        if self.use_colors and both_colored:
            final_pcd.colors = o3d.utility.Vector3dVector(final_points[:, 3:6])
        else:
            color_by_dist(final_pcd)

        self.simulated_object_points = simulated_object_points

        self.comb_window.scene.remove_geometry("Final Point Cloud")
        self.comb_window.scene.remove_geometry("Final Bounding Box")

        self.comb_window.scene.add_geometry(
            "Final Point Cloud", final_pcd, self.comb_mat
        )
        self.comb_window.scene.add_geometry(
            "Final Bounding Box", moved_bbox, self.bbox_mat
        )

        if self.use_ground_z:
            self.comb_window.scene.remove_geometry("Center Ground Point")
            self.comb_window.scene.add_geometry(
                "Center Ground Point", self.center_ground_ball, self.comb_mat
            )

        if self.show_lidar:
            self.comb_window.scene.remove_geometry("Top LiDAR Cone")
            self.comb_window.scene.remove_geometry("Bottom LiDAR Cone")

            height = self.bg_x_range[1] - self.bg_x_range[0]
            max_elev = np.radians(90) - np.max(np.abs(self.beams))
            radius = np.tan(max_elev) * height
            cone = o3d.geometry.TriangleMesh.create_cone(radius, height)

            top_lidar = o3d.geometry.LineSet.create_from_triangle_mesh(cone)
            flip_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            top_lidar.rotate(flip_R, center=(0, 0, 0))
            top_lidar.translate(np.array([0, 0, height]), relative=True)
            top_lidar.rotate(self.lidar_R, center=(0, 0, 0))
            top_lidar.translate(self.optical_center, relative=True)

            bottom_lidar = o3d.geometry.LineSet.create_from_triangle_mesh(cone)
            bottom_lidar.translate(np.array([0, 0, -height]), relative=True)
            bottom_lidar.rotate(self.lidar_R, center=(0, 0, 0))
            bottom_lidar.translate(self.optical_center, relative=True)

            self.comb_window.scene.add_geometry(
                "Top LiDAR Cone", top_lidar, self.comb_mat
            )
            self.comb_window.scene.add_geometry(
                "Bottom LiDAR Cone", bottom_lidar, self.comb_mat
            )

        if self.comb_window.scene.camera.get_field_of_view() == 90.0:
            bbox = self.comb_window.scene.bounding_box
            self.comb_window.setup_camera(60.0, bbox, bbox.get_center())

    def create_range_images(self):
        range_image = project_points(
            self.bg_range_image.copy(),
            self.simulated_object_points,
            self.height,
            self.width,
            self.lidar_R.T,
            self.sensor2lidar,
            self.beams,
            self.bg_min_dist,
            1,
        )[..., 4]
        image = 255 * range_image / range_image.max()
        img = Image.fromarray(image.astype("uint8"))
        img.show()

        diff_image = image / 255 - np.array(self.close_img) / 255
        diff_image = np.abs(diff_image)
        diff_image = 255 * (diff_image / diff_image.max())
        diff_img = Image.fromarray(diff_image.astype("uint8"))
        diff_img.show()
        self.close_img.show()


def main():
    Paved2Paradise().run()


if __name__ == "__main__":
    main()
