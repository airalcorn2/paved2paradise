import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

from open3d.visualization import gui, rendering
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

# Visualization stuff.
VMIN = 0
VMAX = 10
CMAP_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
COLORMAP = "RdYlGn"
CMAP = plt.get_cmap(COLORMAP)
BBOX_COLOR = [1, 0, 0]
GRID_COLOR = [0.5, 0.5, 0.5]
CYL_RADIUS = 0.05

# The maximum number of close object points to a ray that are used to estimate the
# surface point of an object.
NEIGHBORS = 2

# The forward vector of the Ouster sensor.
FWD = np.array([1, 0, 0])


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


class Paved2Paradise:
    def __init__(self):
        self.w = None
        self.theme = None
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        # See: https://github.com/isl-org/Open3D/issues/4921#issuecomment-1096845864.
        self.bbox_mat = rendering.MaterialRecord()
        self.bbox_mat.shader = "unlitLine"
        self.bbox_mat.line_width = 2
        self.obj_window = None
        self.bg_window = None
        self.scene_R = np.eye(2)
        self.pcds = {}
        self.grid_pcds = {}
        self.ground_planes = {}
        self.ground_pcds = {}
        self.level_pcds = {}
        self.level_ground_pcds = {}
        self.show_grid_pcds = {"obj": False, "bg": False}
        self.show_ground_planes = {"obj": False, "bg": False}
        self.show_ground_pcds = {"obj": False, "bg": False}
        self.show_level_pcds = {"obj": True, "bg": True}
        self.obj_bbox = None
        self._settings_panel = None
        self.grid_points = 100
        self.new_obj_xy = np.array([9.98, 1.19])
        self.in_obj_xy = np.array([9.98, 1.19])
        self.grids_info = {
            "obj": {"length": 7, "width": 6},
            "bg": {"length": 7, "width": 6},
        }
        self.show_new_obj_loc = True
        self.occlude = True
        self.sim_lidar = True
        self.show_sector = False
        self.rot_bg_scene = False
        self.hit_thresh = 0.04
        self.occlude_obj_thresh = 0.04
        self.occlude_bg_thresh = 0.03
        self.min_elev = -22.5
        self.max_elev = 22.5
        self.elev_res = 128
        self.azim_res = 2048
        self.sensor2lidar = 0.03618
        self.new_scene = True

    def run(self):
        app = gui.Application.instance
        app.initialize()

        self.w = w = app.create_window("Paved2Paradise", 1024, 768)
        self.theme = w.theme

        self.obj_window = gui.SceneWidget()
        self.obj_window.scene = rendering.Open3DScene(w.renderer)
        self.obj_window.scene.set_background(np.zeros(4))
        self.obj_window.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 1)
        )
        w.add_child(self.obj_window)

        self.bg_window = gui.SceneWidget()
        self.bg_window.scene = rendering.Open3DScene(w.renderer)
        self.bg_window.scene.set_background(np.zeros(4))
        self.bg_window.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 1)
        )
        w.add_child(self.bg_window)

        self.comb_window = gui.SceneWidget()
        self.comb_window.scene = rendering.Open3DScene(w.renderer)
        self.comb_window.scene.set_background(np.zeros(4))
        self.comb_window.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 1)
        )
        w.add_child(self.comb_window)

        em = w.theme.font_size
        self._settings_panel = gui.ScrollableVert()

        pcd_loaders = gui.CollapsableVert(
            "PCD Loaders", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )
        self._settings_panel.add_child(pcd_loaders)

        obj_scene_button = gui.Button("Object Scene")
        obj_scene_button.set_on_clicked(self._on_open_object_scene)

        bg_scene_button = gui.Button("Background Scene")
        bg_scene_button.set_on_clicked(self._on_open_background_scene)

        rot_bg_scene = gui.Checkbox("360° Background Scene")
        rot_bg_scene.set_on_checked(self._on_rot_bg_scene)

        v = gui.Vert(0.25 * em)
        v.add_child(obj_scene_button)
        v.add_child(bg_scene_button)
        v.add_child(rot_bg_scene)
        pcd_loaders.add_child(v)

        final_scene = gui.CollapsableVert(
            "Final Scene", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )
        self._settings_panel.add_child(final_scene)

        obj_x_label = gui.Label("Object x-coordinate (m)")
        obj_x = gui.TextEdit()
        obj_x.set_on_value_changed(self._on_obj_x_changed)
        obj_x.text_value = str(self.in_obj_xy[0])

        obj_y_label = gui.Label("Object y-coordinate (m)")
        obj_y = gui.TextEdit()
        obj_y.set_on_value_changed(self._on_obj_y_changed)
        obj_y.text_value = str(self.in_obj_xy[1])

        level_obj = gui.Checkbox("Level Object")
        level_obj.set_on_checked(self._on_level_object)
        level_obj.checked = True

        level_bg = gui.Checkbox("Level Background")
        level_bg.set_on_checked(self._on_level_background)
        level_bg.checked = True

        occlude = gui.Checkbox("Occlude")
        occlude.checked = True
        occlude.set_on_checked(self._on_occlude)

        sim_lidar = gui.Checkbox("Simulate LiDAR")
        sim_lidar.checked = True
        sim_lidar.set_on_checked(self._on_sim_lidar)

        show_new_obj_loc = gui.Checkbox("New Object Location")
        show_new_obj_loc.checked = True
        show_new_obj_loc.set_on_checked(self._on_show_new_obj_loc)

        show_sector = gui.Checkbox("Sector")
        show_sector.set_on_checked(self._on_show_sector)

        v = gui.Vert(0.25 * em)
        v.add_child(obj_x_label)
        v.add_child(obj_x)
        v.add_child(obj_y_label)
        v.add_child(obj_y)
        v.add_child(level_obj)
        v.add_child(level_bg)
        v.add_child(occlude)
        v.add_child(sim_lidar)
        v.add_child(show_new_obj_loc)
        v.add_child(show_sector)
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

        sensor_settings = gui.CollapsableVert(
            "Sensor Settings", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )
        self._settings_panel.add_child(sensor_settings)

        hit_label = gui.Label("Hit Threshold (m)")
        hit_thresh = gui.TextEdit()
        hit_thresh.set_on_value_changed(self._on_hit_thresh_changed)
        hit_thresh.text_value = str(self.hit_thresh)

        occlude_obj_label = gui.Label("Occlude Object Threshold (m)")
        occlude_obj_thresh = gui.TextEdit()
        occlude_obj_thresh.set_on_value_changed(self._on_occlude_obj_thresh_changed)
        occlude_obj_thresh.text_value = str(self.occlude_obj_thresh)

        occlude_bg_label = gui.Label("Occlude Background Threshold (m)")
        occlude_bg_thresh = gui.TextEdit()
        occlude_bg_thresh.set_on_value_changed(self._on_occlude_bg_thresh_changed)
        occlude_bg_thresh.text_value = str(self.occlude_bg_thresh)

        min_elev_label = gui.Label("Minimum Elevation (°)")
        min_elev = gui.TextEdit()
        min_elev.set_on_value_changed(self._on_min_elev_changed)
        min_elev.text_value = str(self.min_elev)

        max_elev_label = gui.Label("Maximum Elevation (°)")
        max_elev = gui.TextEdit()
        max_elev.set_on_value_changed(self._on_max_elev_changed)
        max_elev.text_value = str(self.max_elev)

        elev_res_label = gui.Label("Elevation Resolution")
        elev_res = gui.TextEdit()
        elev_res.set_on_value_changed(self._on_elev_res_changed)
        elev_res.text_value = str(self.elev_res)

        azim_res_label = gui.Label("Azimuth Resolution")
        azim_res = gui.TextEdit()
        azim_res.set_on_value_changed(self._on_azim_res_changed)
        azim_res.text_value = str(self.azim_res)

        sensor2lidar_label = gui.Label("Sensor to LiDAR (m)")
        sensor2lidar = gui.TextEdit()
        sensor2lidar.set_on_value_changed(self._on_sensor2lidar_changed)
        sensor2lidar.text_value = str(self.sensor2lidar)

        v = gui.Vert(0.25 * em)
        v.add_child(hit_label)
        v.add_child(hit_thresh)
        v.add_child(occlude_obj_label)
        v.add_child(occlude_obj_thresh)
        v.add_child(occlude_bg_label)
        v.add_child(occlude_bg_thresh)
        v.add_child(min_elev_label)
        v.add_child(min_elev)
        v.add_child(max_elev_label)
        v.add_child(max_elev)
        v.add_child(elev_res_label)
        v.add_child(elev_res)
        v.add_child(azim_res_label)
        v.add_child(azim_res)
        v.add_child(sensor2lidar_label)
        v.add_child(sensor2lidar)
        sensor_settings.add_child(v)

        w.add_child(self._settings_panel)
        w.set_on_layout(self._on_layout)

        self._on_load_object_dialog_done("./parking_lot.pcd")
        self._on_load_background_dialog_done("./orchard.pcd")

        app.run()

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
        self.obj_path = pcd_f
        self.w.close_dialog()
        obj_pcd = o3d.io.read_point_cloud(pcd_f)
        self.pcds["obj"] = obj_pcd
        color_by_dist(obj_pcd)
        self.obj_window.scene.clear_geometry()
        self.obj_window.scene.add_geometry("Points", obj_pcd, self.mat)
        bbox = self.obj_window.scene.bounding_box
        self.obj_window.setup_camera(60.0, bbox, bbox.get_center())

        path = os.path.dirname(pcd_f)
        name = os.path.basename(pcd_f).split(".pcd")[0]
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
        self.obj_bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
        self.obj_bbox.color = BBOX_COLOR

        self.create_grid_points("obj")
        self.create_ground_plane("obj")
        self.level_scene("obj")
        self.render_obj_scene()
        self.new_scene = True
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
        self.bg_path = pcd_f
        self.w.close_dialog()
        bg_pcd = o3d.io.read_point_cloud(pcd_f)
        if self.rot_bg_scene:
            x = self.in_obj_xy[0]
            y = self.in_obj_xy[1]
            # Rotate the background scene so that the object will be placed in the middle.
            self.scene_R = Rotation.from_euler("z", -np.arctan2(y, x)).as_matrix()
            background_points = (self.scene_R @ np.array(bg_pcd.points).T).T
            bg_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(background_points)
            )
            self.new_obj_xy = self.scene_R[:2, :2] @ np.array([x, y])

        self.pcds["bg"] = bg_pcd
        color_by_dist(bg_pcd)
        self.bg_window.scene.clear_geometry()
        self.bg_window.scene.add_geometry("Points", bg_pcd, self.mat)
        bbox = self.bg_window.scene.bounding_box
        self.bg_window.setup_camera(60.0, bbox, bbox.get_center())
        self.create_grid_points("bg")
        self.create_ground_plane("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.new_scene = True
        self.simulate_scene()

    def _on_rot_bg_scene(self, rot_bg_scene):
        self.rot_bg_scene = rot_bg_scene

    def _on_obj_x_changed(self, obj_x):
        self.in_obj_xy[0] = float(obj_x)
        self.new_obj_xy = self.scene_R[:2, :2] @ self.in_obj_xy
        self.create_ground_plane("bg")
        self.create_grid_points("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_obj_y_changed(self, obj_y):
        self.in_obj_xy[1] = float(obj_y)
        self.new_obj_xy = self.scene_R[:2, :2] @ self.in_obj_xy
        self.create_ground_plane("bg")
        self.create_grid_points("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def level_scene(self, which):
        points = np.array(self.pcds[which].points)
        if which == "obj":
            grid_x = self.obj_bbox.center[0]
            grid_y = self.obj_bbox.center[1]
        else:
            grid_x = self.new_obj_xy[0]
            grid_y = self.new_obj_xy[1]

        x_start = grid_x - self.grids_info[which]["length"] / 2
        x_end = x_start + self.grids_info[which]["length"]
        y_start = grid_y - self.grids_info[which]["width"] / 2
        y_end = y_start + self.grids_info[which]["width"]
        in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
        in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
        region_points = points[in_x & in_y]

        # For each grid point, find nearest neighbor in scene point cloud. These are our
        # "ground" points.
        nbrs = NearestNeighbors(n_neighbors=1).fit(region_points)
        (_, pcd_idxs) = nbrs.kneighbors(np.array(self.grid_pcds[which].points))
        pcd_idxs = np.unique(pcd_idxs)
        ground_points = region_points[pcd_idxs]
        ground_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
        self.ground_pcds[which] = ground_pcd
        color_by_dist(ground_pcd)

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
            self.bg_t = -t
            self.bg_R = R

    def _on_level_object(self, level_obj):
        self.show_level_pcds["obj"] = level_obj
        self.render_obj_scene()
        self.simulate_scene()

    def _on_level_background(self, level_bg):
        self.show_level_pcds["bg"] = level_bg
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

    def _on_show_sector(self, show_sector):
        self.show_sector = show_sector
        self.simulate_scene()

    def create_grid_points(self, which):
        points = np.array(self.pcds[which].points)
        if which == "obj":
            x = self.obj_bbox.center[0]
            y = self.obj_bbox.center[1]
        else:
            x = self.new_obj_xy[0]
            y = self.new_obj_xy[1]

        x_start = x - self.grids_info[which]["length"] / 2
        x_end = x_start + self.grids_info[which]["length"]
        in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
        y_start = y - self.grids_info[which]["width"] / 2
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
        if which == "obj":
            grid_x = self.obj_bbox.center[0]
            grid_y = self.obj_bbox.center[1]
        else:
            grid_x = self.new_obj_xy[0]
            grid_y = self.new_obj_xy[1]

        center = np.array([grid_x, grid_y, 0])
        ground_plane.translate(center, relative=False)

        self.ground_planes[which] = ground_plane

    def _on_grid_points_changed(self, grid_points):
        self.grid_points = int(grid_points)
        for which in ["obj", "bg"]:
            self.create_grid_points(which)
            self.level_scene(which)

        self.render_obj_scene()
        self.render_bg_scene()
        self.simulate_scene()

    def _on_obj_grid_length_changed(self, grid_length):
        self.grids_info["obj"]["length"] = float(grid_length)
        self.create_grid_points("obj")
        self.create_ground_plane("obj")
        self.level_scene("obj")
        self.render_obj_scene()
        self.simulate_scene()

    def _on_obj_grid_width_changed(self, grid_width):
        self.grids_info["obj"]["width"] = float(grid_width)
        self.create_grid_points("obj")
        self.create_ground_plane("obj")
        self.level_scene("obj")
        self.render_obj_scene()
        self.simulate_scene()

    def _on_bg_grid_length_changed(self, grid_length):
        self.grids_info["bg"]["length"] = float(grid_length)
        self.create_grid_points("bg")
        self.create_ground_plane("bg")
        self.level_scene("bg")
        self.render_bg_scene()
        self.simulate_scene()

    def _on_bg_grid_width_changed(self, grid_length):
        self.grids_info["bg"]["width"] = float(grid_length)
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

    def _on_hit_thresh_changed(self, hit_thresh):
        self.hit_thresh = float(hit_thresh)
        self.simulate_scene()

    def _on_occlude_obj_thresh_changed(self, occlude_obj_thresh):
        self.occlude_obj_thresh = float(occlude_obj_thresh)
        self.simulate_scene()

    def _on_occlude_bg_thresh_changed(self, occlude_bg_thresh):
        self.occlude_bg_thresh = float(occlude_bg_thresh)
        self.simulate_scene()

    def _on_min_elev_changed(self, min_elev):
        self.min_elev = float(min_elev)
        self.simulate_scene()

    def _on_max_elev_changed(self, max_elev):
        self.max_elev = float(max_elev)
        self.simulate_scene()

    def _on_elev_res_changed(self, elev_res):
        self.elev_res = int(elev_res)
        self.simulate_scene()

    def _on_azim_res_changed(self, azim_res):
        self.azim_res = int(azim_res)
        self.simulate_scene()

    def _on_sensor2lidar_changed(self, sensor2lidar):
        self.sensor2lidar = float(sensor2lidar)
        self.simulate_scene()

    def block(self, blockee, blockee_norms, blocker, blocker_norms, occlude_thresh):
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

    def unocclude(self, new_points):
        point_angles = np.arctan2(new_points[:, 1], new_points[:, 0])
        azim_pm = 2 * np.pi / self.azim_res
        min_azim = point_angles.min() - azim_pm
        max_azim = point_angles.max() + azim_pm
        azims = np.linspace(-np.pi, np.pi, self.azim_res, False)
        azims = azims[(min_azim < azims) & (azims < max_azim)]
        (min_elev, max_elev) = (np.deg2rad(self.min_elev), np.deg2rad(self.max_elev))
        elevs = np.linspace(min_elev, max_elev, self.elev_res)
        elev_azims = (
            np.stack(np.meshgrid(-elevs, azims)).transpose(1, 2, 0).reshape(-1, 2)
        )
        Rs = Rotation.from_euler("yz", elev_azims).as_matrix()

        Ps = new_points
        Ds = Rs @ FWD
        t_Ps = Ds @ Ps.T
        ray_dists = (np.linalg.norm(Ps, axis=1) ** 2 - t_Ps**2) ** 0.5

        # Average closest points.
        closest_ray_dists = ray_dists.argsort(1)
        ray_dists[ray_dists >= self.hit_thresh] = np.inf
        final_points = np.zeros((len(t_Ps), NEIGHBORS, 3))
        final_dists = np.full((len(t_Ps), NEIGHBORS), -1.0)
        total_points = np.zeros(len(t_Ps))
        idxs = np.arange(len(t_Ps))
        for neighbor in range(NEIGHBORS):
            min_Ps = closest_ray_dists[:, neighbor]
            dists = ray_dists[idxs, min_Ps]
            keep = dists < self.hit_thresh
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
        very_close = (final_dists < self.hit_thresh / 2).any(1)
        final_points = final_points[very_close | two_neighbors]
        final_points = final_points[~(final_points == 0).all(1)]

        return final_points

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
            bg_t = self.bg_t
            bg_R = self.bg_R
        else:
            bg_t = 0
            bg_R = np.eye(3)

        object_pcd = object_pcd.crop(bbox)
        object_points = np.array(object_pcd.points)
        if self.show_level_pcds["obj"]:
            # Only keep points above the ground.
            object_points = object_points[object_points[:, 2] > 0]

        if self.show_new_obj_loc:
            (background_object_points, bbox) = move_object_to_position(
                object_points, bbox, self.new_obj_xy
            )
        else:
            background_object_points = object_points

        background_object_points += bg_t
        background_object_points = (bg_R.T @ background_object_points.T).T
        center = bg_R.T @ (bbox.center + bg_t)
        bbox_R = bg_R.T @ bbox.R
        bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, bbox.extent)
        bbox.color = BBOX_COLOR

        # The points from the Ouster are in the "Sensor Coordinate Frame" by default, so
        # we have to shift the z coordinates.
        # See: https://static.ouster.dev/sdk-docs/python/api/client.html?highlight=xyzlut#ouster.client.XYZLut
        # and: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#sensor-coordinate-frame.
        background_object_points = background_object_points + self.sensor2lidar
        background_points = np.array(self.pcds["bg"].points) + self.sensor2lidar

        if self.sim_lidar:
            # Get simulated point cloud based on position relative to sensor.
            unoccluded_points = self.unocclude(background_object_points)

        else:
            unoccluded_points = background_object_points

        # Only consider background points that are in the same cylindrical slice as the
        # object (+/- a few degrees).
        object_angles = np.arctan2(
            background_object_points[:, 1], background_object_points[:, 0]
        )
        (min_angle, max_angle) = (object_angles.min(), object_angles.max())
        azim_pm = 2 * np.pi / self.azim_res
        min_angle -= azim_pm
        max_angle += azim_pm
        if self.occlude:
            background_angles = np.arctan2(
                background_points[:, 1], background_points[:, 0]
            )
            in_cyl = (min_angle < background_angles) & (background_angles < max_angle)
            in_cyl_background_points = background_points[in_cyl]

            obj_norms = np.linalg.norm(background_object_points, axis=1)
            in_cyl_norms = np.linalg.norm(in_cyl_background_points, axis=1)

            # Occlude background points with object points.
            further = in_cyl_norms >= obj_norms.min()
            further_background_points = in_cyl_background_points[further]
            further_norms = in_cyl_norms[further]
            drop_mask = self.block(
                further_background_points,
                further_norms,
                background_object_points,
                obj_norms,
                self.occlude_bg_thresh,
            )
            occluded_background_idxs = np.arange(len(background_points))[in_cyl][
                further
            ][drop_mask]
            keep_mask = np.ones(len(background_points), dtype="bool")
            keep_mask[occluded_background_idxs] = False
            occluded_background_points = background_points[keep_mask]

            # Occlude object points with background points.
            unoccluded_norms = np.linalg.norm(unoccluded_points, axis=1)
            closer = in_cyl_norms <= obj_norms.max()
            closer_background_points = in_cyl_background_points[closer]
            closer_norms = in_cyl_norms[closer]
            drop_mask = self.block(
                unoccluded_points,
                unoccluded_norms,
                closer_background_points,
                closer_norms,
                self.occlude_obj_thresh,
            )
            simulated_object_points = unoccluded_points[~drop_mask]

        else:
            occluded_background_points = background_points
            simulated_object_points = unoccluded_points

        occluded_background_points -= self.sensor2lidar
        simulated_object_points -= self.sensor2lidar

        simulated_object_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(simulated_object_points)
        )
        color_by_dist(simulated_object_pcd)

        self.comb_window.scene.remove_geometry("Final Object")
        self.comb_window.scene.remove_geometry("Final Bounding Box")
        self.comb_window.scene.remove_geometry("Final Background")
        self.comb_window.scene.add_geometry(
            "Final Object", simulated_object_pcd, self.mat
        )
        self.comb_window.scene.add_geometry("Final Bounding Box", bbox, self.bbox_mat)
        if self.show_sector:
            background_angles = np.arctan2(
                occluded_background_points[:, 1], occluded_background_points[:, 0]
            )
            in_cyl = (min_angle < background_angles) & (background_angles < max_angle)
            in_cyl_occluded_background_points = occluded_background_points[in_cyl]
            in_cyl_occluded_background_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(in_cyl_occluded_background_points)
            )
            color_by_dist(in_cyl_occluded_background_pcd)
            self.comb_window.scene.add_geometry(
                "Final Background", in_cyl_occluded_background_pcd, self.mat
            )

        else:
            occluded_background_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(occluded_background_points)
            )
            color_by_dist(occluded_background_pcd)
            self.comb_window.scene.add_geometry(
                "Final Background", occluded_background_pcd, self.mat
            )

        if self.new_scene:
            self.new_scene = False
            bbox = self.comb_window.scene.bounding_box
            self.comb_window.setup_camera(60.0, bbox, bbox.get_center())


def main():
    Paved2Paradise().run()


if __name__ == "__main__":
    main()
