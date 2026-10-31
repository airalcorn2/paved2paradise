"""Interactive Open3D GUI for exploring the Paved2Paradise LiDAR augmentation pipeline.

Loads an "object" point cloud (e.g., a person) and a "background" point cloud (e.g.,
an orchard), lets the user interactively place the object in the background scene,
and simulates how the combined scene would look from the LiDAR sensor's perspective
-- including which points on the object a beam would actually hit and which object
points are occluded by the background. See core.py for the underlying augmentation
algorithm.
"""

import ast
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

from core import move_object_to_position, project_points, unlevel, unocclude
from lidar2intrinsics import lidar2intrinsics
from matplotlib.colors import Normalize
from open3d.visualization import gui
# open3d's stubs don't declare the "rendering" submodule in some versions.
from open3d.visualization import rendering  # pyright: ignore[reportAttributeAccessIssue]
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from typing import Literal, cast

Which = Literal["obj", "bg"]

# Visualization stuff.
VMIN = 0
VMAX = 12
CMAP_NORM = Normalize(vmin=VMIN, vmax=VMAX)
COLORMAP = "RdYlGn"
CMAP = plt.get_cmap(COLORMAP)
BBOX_COLOR = [1, 0, 0]
GRID_COLOR = [0.5, 0.5, 0.5]
CYL_RADIUS = 0.05
PCD_SUFFIXES = {"pcd", "npy"}

SENSOR_IDX2SENSOR = ["lidar_front"]
DEFAULT_OBJ_F = "./human_colored.pcd"
DEFAULT_BG_F = "./orchard_colored.pcd"


def create_depth_image(depths: np.ndarray, max_depth: float) -> Image.Image:
    """Render a depth map (in meters) as a grayscale image, near is bright."""
    depths[depths == 0] = max_depth
    depth_image = 255 * (1 - depths / max_depth)
    return Image.fromarray(depth_image.astype("uint8"))


def color_by_dist(pcd: o3d.geometry.PointCloud) -> None:
    """Color a point cloud's points by their distance from the sensor origin."""
    pcd_points = np.array(pcd.points)
    col_vals = np.linalg.norm(pcd_points, axis=1)
    point_colors = CMAP(CMAP_NORM(col_vals))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)


class Paved2Paradise:
    """Open3D GUI application for the Paved2Paradise demo.

    Three scene views are shown side by side: the object scene, the background
    scene, and the combined/simulated scene. A settings panel on the right exposes
    the augmentation pipeline's parameters.

    Most of the object- and background-scene logic is shared, keyed by `which`
    ("obj" or "bg") into the various `self.*` dicts (e.g., `self.pcds`,
    `self.grid_pcds`), rather than duplicated.
    """

    def __init__(self) -> None:
        # These are placeholders that get assigned real values before any of them are
        # read, in run() and update_sensor_params() (both called before the event loop
        # starts) -- the annotations document their real types for the type checker.
        self.w: gui.Window = cast(gui.Window, None)
        self.theme: gui.Theme = cast(gui.Theme, None)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.comb_mat = rendering.MaterialRecord()
        self.comb_mat.shader = "defaultUnlit"
        # See: https://github.com/isl-org/Open3D/issues/4921#issuecomment-1096845864.
        self.bbox_mat = rendering.MaterialRecord()
        self.bbox_mat.shader = "unlitLine"
        self.bbox_mat.line_width = 2
        self.obj_window: gui.SceneWidget = cast(gui.SceneWidget, None)
        self.bg_window: gui.SceneWidget = cast(gui.SceneWidget, None)
        self.comb_window: gui.SceneWidget = cast(gui.SceneWidget, None)

        self._settings_panel: gui.ScrollableVert = cast(gui.ScrollableVert, None)

        self.pcds: dict[Which, o3d.geometry.PointCloud] = {}
        self.obj_pcd: o3d.geometry.PointCloud = cast(o3d.geometry.PointCloud, None)
        self.obj_bbox: o3d.geometry.OrientedBoundingBox = cast(
            o3d.geometry.OrientedBoundingBox, None
        )
        self.level_obj_bbox: o3d.geometry.OrientedBoundingBox = cast(
            o3d.geometry.OrientedBoundingBox, None
        )
        self.bg_pcd: o3d.geometry.PointCloud = cast(o3d.geometry.PointCloud, None)
        self.lidar_R: np.ndarray = cast(np.ndarray, None)
        self.Rs: np.ndarray = cast(np.ndarray, None)
        self.Ds: np.ndarray = cast(np.ndarray, None)
        self.D_azims: np.ndarray = cast(np.ndarray, None)
        self.D_elevs: np.ndarray = cast(np.ndarray, None)
        self.sensor2lidar: np.ndarray = cast(np.ndarray, None)
        self.optical_center: np.ndarray = cast(np.ndarray, None)
        self.beams: np.ndarray = cast(np.ndarray, None)
        self.azim_pm: float = cast(float, None)
        self.elev_pm: float = cast(float, None)
        self.height: int = cast(int, None)
        self.width: int = cast(int, None)
        self.pixel_shift_by_row: list = cast(list, None)
        self.obj_x_range = (0, 13)
        self.obj_y_range = (-5.625, 5.625)
        self.bg_x_range = (-1, 9)
        self.bg_y_range = (-9, 9)

        self.new_obj_xy = np.array([5.79, -1.00])
        self.obj_xy: gui.TextEdit = cast(gui.TextEdit, None)
        self.hit_thresh = 0.025
        self.bg_min_dist = 0.1
        self.max_sector_angle = 2.25
        self.neighbors = 2
        self.use_colors = True
        self.show_level_pcds: dict[Which, bool] = {"obj": True, "bg": True}
        self.level_pcds: dict[Which, o3d.geometry.PointCloud] = {}
        self.level_at_location = True
        self.use_ground_z = True
        self.center_ground_ball: o3d.geometry.TriangleMesh = cast(
            o3d.geometry.TriangleMesh, None
        )
        self.local_t: np.ndarray = cast(np.ndarray, None)
        self.bg_t: np.ndarray = cast(np.ndarray, None)
        self.bg_R: np.ndarray = cast(np.ndarray, None)
        self.occlude = True
        self.sim_lidar = True
        self.placement_mode = False
        self.placement_mode_check: gui.Checkbox = cast(gui.Checkbox, None)
        self.show_new_obj_loc = True
        self.show_lidar = False
        self.show_beams = False
        self.prev_n_beams = 0
        self.start_beam_idx = 0
        self.end_beam_idx = 0
        self.point_size = 6.0

        self.grid_pcds: dict[Which, o3d.geometry.PointCloud] = {}
        self.ground_planes: dict[Which, o3d.geometry.TriangleMesh] = {}
        self.ground_pcds: dict[Which, o3d.geometry.PointCloud] = {}
        self.level_ground_pcds: dict[Which, o3d.geometry.PointCloud] = {}
        self.grid_points = 100
        self.grids_info: dict[Which, dict[str, float]] = {
            "obj": {"length": 7.0, "width": 6.0},
            "bg": {"length": 7.0, "width": 6.0},
        }
        self.show_grid_pcds: dict[Which, bool] = {"obj": False, "bg": False}
        self.show_ground_planes: dict[Which, bool] = {"obj": False, "bg": False}
        self.show_ground_pcds: dict[Which, bool] = {"obj": False, "bg": False}

        self.final_pcd = o3d.geometry.PointCloud()
        self.simulated_object_points: np.ndarray = cast(np.ndarray, None)
        self.close_img: Image.Image = cast(Image.Image, None)
        self.og_bg_range_image: np.ndarray = cast(np.ndarray, None)
        self.bg_range_image: np.ndarray = cast(np.ndarray, None)

    def run(self) -> None:
        """Build the GUI window and widgets, load the default scenes, and block
        on Open3D's event loop until the window is closed."""
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
        # open3d's stub declares the callback parameter as the broader SupportsInt
        # rather than int, which is actually passed at runtime.
        sensor_selector.set_on_selection_changed(
            self._on_sensor_changed  # pyright: ignore[reportArgumentType]
        )

        obj_x_range_label = gui.Label("Object x Range (m)")
        obj_x_range = gui.TextEdit()
        obj_x_range.set_on_value_changed(
            lambda v: self._on_range_changed("obj", "x", "Object x-range", v)
        )
        obj_x_range.text_value = str(self.obj_x_range)

        obj_y_range_label = gui.Label("Object y Range (m)")
        obj_y_range = gui.TextEdit()
        obj_y_range.set_on_value_changed(
            lambda v: self._on_range_changed("obj", "y", "Object y-range", v)
        )
        obj_y_range.text_value = str(self.obj_y_range)

        bg_x_range_label = gui.Label("Background x Range (m)")
        bg_x_range = gui.TextEdit()
        bg_x_range.set_on_value_changed(
            lambda v: self._on_range_changed("bg", "x", "Background x-range", v)
        )
        bg_x_range.text_value = str(self.bg_x_range)

        bg_y_range_label = gui.Label("Background y Range (m)")
        bg_y_range = gui.TextEdit()
        bg_y_range.set_on_value_changed(
            lambda v: self._on_range_changed("bg", "y", "Background y-range", v)
        )
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
        level_obj.set_on_checked(lambda c: self._on_level_pcd("obj", c))
        level_obj.checked = self.show_level_pcds["obj"]

        level_bg = gui.Checkbox("Level Background")
        level_bg.set_on_checked(lambda c: self._on_level_pcd("bg", c))
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

        self.placement_mode_check = gui.Checkbox("Placement Mode")
        self.placement_mode_check.set_on_checked(self._on_placement_mode_changed)
        self.placement_mode_check.checked = self.placement_mode

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
        v.add_child(self.placement_mode_check)
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
        obj_grid_length.set_on_value_changed(
            lambda v: self._on_grid_dim_changed("obj", "length", "Object Grid Length", v)
        )
        obj_grid_length.text_value = str(self.grids_info["obj"]["length"])

        obj_grid_width_label = gui.Label("Object Grid Width (m)")
        obj_grid_width = gui.TextEdit()
        obj_grid_width.set_on_value_changed(
            lambda v: self._on_grid_dim_changed("obj", "width", "Object Grid Width", v)
        )
        obj_grid_width.text_value = str(self.grids_info["obj"]["width"])

        bg_grid_length_label = gui.Label("Background Grid Length (m)")
        bg_grid_length = gui.TextEdit()
        bg_grid_length.set_on_value_changed(
            lambda v: self._on_grid_dim_changed(
                "bg", "length", "Background Grid Length", v
            )
        )
        bg_grid_length.text_value = str(self.grids_info["bg"]["length"])

        bg_grid_width_label = gui.Label("Background Grid Width (m)")
        bg_grid_width = gui.TextEdit()
        bg_grid_width.set_on_value_changed(
            lambda v: self._on_grid_dim_changed(
                "bg", "width", "Background Grid Width", v
            )
        )
        bg_grid_width.text_value = str(self.grids_info["bg"]["width"])

        obj_grid_points_check = gui.Checkbox("Object Grid")
        obj_grid_points_check.set_on_checked(
            lambda c: self._on_show_grid_points("obj", c)
        )

        bg_grid_points_check = gui.Checkbox("Background Grid")
        bg_grid_points_check.set_on_checked(
            lambda c: self._on_show_grid_points("bg", c)
        )

        obj_ground_plane_check = gui.Checkbox("Object Ground Plane")
        obj_ground_plane_check.set_on_checked(
            lambda c: self._on_show_ground_plane("obj", c)
        )

        bg_ground_plane_check = gui.Checkbox("Background Ground Plane")
        bg_ground_plane_check.set_on_checked(
            lambda c: self._on_show_ground_plane("bg", c)
        )

        obj_ground_points = gui.Checkbox("Object Ground Points")
        obj_ground_points.set_on_checked(
            lambda c: self._on_show_ground_points("obj", c)
        )

        bg_ground_points = gui.Checkbox("Background Ground Points")
        bg_ground_points.set_on_checked(
            lambda c: self._on_show_ground_points("bg", c)
        )

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
        self.w.set_on_key(self._on_key)

        app.run()

    def _on_mouse_widget3d(
        self, event: gui.MouseEvent
    ) -> gui.Widget.EventCallbackResult:
        """Right-click in the combined scene to print the clicked (x, y, z) world
        coordinates, and -- while placement mode (toggled by pressing "P") is
        active -- also move the object there."""
        # See: https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/mouse_and_point_coord.py.
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN) and (
            event.buttons == int(gui.MouseButton.RIGHT)
        ):

            def depth_callback(depth_image) -> None:
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
                    print(world)
                    if self.placement_mode:
                        obj_xy_text = f"({world[0]:.2f}, {world[1]:.2f})"
                        self.obj_xy.text_value = obj_xy_text
                        self._on_obj_xy_changed(self.obj_xy.text_value)

            self.comb_window.scene.scene.render_to_depth_image(depth_callback)

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(self, event: gui.KeyEvent) -> bool:
        """Pressing "P" toggles placement mode (see _on_mouse_widget3d)."""
        if (
            (event.type == gui.KeyEvent.Type.DOWN)
            and (event.key == gui.KeyName.P)
            and not event.is_repeat
        ):
            self.placement_mode = not self.placement_mode
            self.placement_mode_check.checked = self.placement_mode

        return False

    def _on_placement_mode_changed(self, checked: bool) -> None:
        self.placement_mode = checked

    def _on_layout(self, layout_context: gui.LayoutContext) -> None:
        """Lay the three scene views out side by side, with the settings panel
        docked to the right."""
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

    def _on_open_object_scene(self) -> None:
        """Open a file dialog to choose a new object scene point cloud."""
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose object scene.", self.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_object_dialog_done)
        self.w.show_dialog(dlg)

    def _on_open_background_scene(self) -> None:
        """Open a file dialog to choose a new background scene point cloud."""
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose background scene.", self.theme
        )
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_background_dialog_done)
        self.w.show_dialog(dlg)

    def _on_file_dialog_cancel(self) -> None:
        self.w.close_dialog()

    def render_scene(self, which: Which) -> None:
        """Redraw the object or background scene view from current toggle state
        (leveled/raw, grid points, ground plane, ground points)."""
        window = self.obj_window if which == "obj" else self.bg_window
        window.scene.clear_geometry()

        if self.show_ground_pcds[which]:
            pcd = (
                self.level_ground_pcds[which]
                if self.show_level_pcds[which]
                else self.ground_pcds[which]
            )
        else:
            pcd = (
                self.level_pcds[which]
                if self.show_level_pcds[which]
                else self.pcds[which]
            )
            if which == "obj":
                bbox = (
                    self.level_obj_bbox
                    if self.show_level_pcds[which]
                    else self.obj_bbox
                )
                window.scene.add_geometry("Bounding Box", bbox, self.bbox_mat)

        window.scene.add_geometry("Points", pcd, self.mat)

        if self.show_grid_pcds[which]:
            window.scene.add_geometry("Grid Points", self.grid_pcds[which], self.mat)

        if self.show_ground_planes[which]:
            window.scene.add_geometry(
                "Ground Plane", self.ground_planes[which], self.mat
            )

        if (which == "bg") and self.show_new_obj_loc:
            cyl = o3d.geometry.TriangleMesh.create_cylinder(CYL_RADIUS)
            z_bbox = self.level_obj_bbox if self.show_level_pcds["bg"] else self.obj_bbox
            center = np.array([self.new_obj_xy[0], self.new_obj_xy[1], z_bbox.center[2]])
            cyl.translate(center, relative=False)
            cyl.paint_uniform_color(BBOX_COLOR)
            window.scene.add_geometry("New Object Location", cyl, self.mat)

    def _on_load_object_dialog_done(self, pcd_f: str) -> None:
        """Load a new object scene point cloud and its bounding box annotation
        (a sibling .json from Segments.ai, or a sibling .npy), then refresh."""
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

        # open3d's stub omits plain str from the accepted path types, though it
        # works fine at runtime.
        self.obj_pcd = o3d.io.read_point_cloud(
            pcd_f  # pyright: ignore[reportArgumentType]
        )
        self._on_obj_xy_range_changed()
        self.simulate_scene()

    def _on_load_background_dialog_done(self, pcd_f: str) -> None:
        """Load a new background scene point cloud, destagger it into a range
        image (undoing the sensor's per-row pixel shift), and refresh."""
        self.w.close_dialog()
        suffix = pcd_f.split(".")[-1]
        if suffix not in PCD_SUFFIXES:
            print("Background Scene must be a .pcd or .npy file.")
            return

        if suffix == "pcd":
            # open3d's stub omits plain str from the accepted path types, though it
            # works fine at runtime.
            self.bg_pcd = o3d.io.read_point_cloud(
                pcd_f  # pyright: ignore[reportArgumentType]
            )
        else:
            self.bg_pcd = o3d.geometry.PointCloud()
            self.bg_pcd.points = o3d.utility.Vector3dVector(np.load(pcd_f))

        assert len(self.bg_pcd.points) == self.height * self.width
        points = np.array(self.bg_pcd.points)
        points = points.reshape(self.height, self.width, 3)

        depths = np.linalg.norm(points, axis=2)
        img = create_depth_image(depths, depths.max())
        img.show()

        destaggered_points = np.zeros_like(points)
        for v in range(points.shape[0]):
            destaggered_points[v] = np.roll(
                points[v], self.pixel_shift_by_row[v], axis=0
            )

        destaggered_depths = np.linalg.norm(destaggered_points, axis=2)
        self.og_bg_range_image = np.concatenate(
            [destaggered_points, destaggered_depths[..., None]], axis=2
        )

        img = create_depth_image(destaggered_depths, destaggered_depths.max())
        img.show()

        if self.bg_pcd.has_colors():
            colors = np.array(self.bg_pcd.colors)
            colors = colors.reshape(self.height, self.width, 3)
            destaggered_colors = np.zeros_like(colors)
            for v in range(colors.shape[0]):
                destaggered_colors[v] = np.roll(
                    colors[v], self.pixel_shift_by_row[v], axis=0
                )

            self.og_bg_range_image = np.concatenate(
                [self.og_bg_range_image, destaggered_colors], axis=2
            )
            img = Image.fromarray((255 * destaggered_colors).astype("uint8"))
            img.show()

        self._on_bg_xy_range_changed()
        self.simulate_scene()

    def _on_dialog_ok(self) -> None:
        self.w.close_dialog()

    def _create_dialog(self, title: str, message: str) -> None:
        """Show a modal dialog with a single OK button."""
        dlg = gui.Dialog(title)

        em = self.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        ok_button = gui.Button("OK")
        ok_button.set_on_clicked(self._on_dialog_ok)

        dlg_layout.add_child(ok_button)
        dlg.add_child(dlg_layout)
        self.w.show_dialog(dlg)

    def get_Ds(self) -> None:
        """Compute each LiDAR beam's unit ray direction (self.Ds), plus its
        azimuth/elevation (self.D_azims/self.D_elevs) and per-pixel angular size
        (self.azim_pm/self.elev_pm), from the current sensor's beam altitude angles
        and orientation. Used by unocclude()/render_beams() to test which beams
        pass near the object."""
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

    def update_sensor_params(self, sensor: str) -> None:
        """Load the named sensor's intrinsics (from lidar2intrinsics) and recompute
        its beam ray directions."""
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

    def _on_sensor_changed(self, sensor_idx: int) -> None:
        self.update_sensor_params(SENSOR_IDX2SENSOR[sensor_idx])
        self.simulate_scene()

    def _on_show_range_image(self) -> None:
        self.create_range_images()

    def _on_obj_xy_range_changed(self) -> None:
        """Re-crop the object scene to self.obj_x_range/self.obj_y_range and
        refresh everything derived from it (grid, ground plane, leveling, view)."""
        points = np.array(self.obj_pcd.points)
        in_x = (self.obj_x_range[0] < points[:, 0]) & (
            points[:, 0] < self.obj_x_range[1]
        )
        in_y = (self.obj_y_range[0] < points[:, 1]) & (
            points[:, 1] < self.obj_y_range[1]
        )
        obj_pcd = self.obj_pcd.select_by_index(np.where(in_x & in_y)[0].tolist())
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
        self.render_scene("obj")
        self.simulate_scene()

    def _on_range_changed(
        self, which: Which, axis: Literal["x", "y"], label: str, value_str: str
    ) -> None:
        """Parse a "(min, max)" range text field into self.<which>_<axis>_range and
        refresh the matching scene's point selection."""
        try:
            setattr(self, f"{which}_{axis}_range", ast.literal_eval(value_str))
        except (ValueError, SyntaxError, TypeError):
            self._create_dialog("Input Error", f"{label} must be a tuple of floats.")
            return

        if which == "obj":
            self._on_obj_xy_range_changed()
        else:
            self._on_bg_xy_range_changed()

    def _on_bg_xy_range_changed(self) -> None:
        """Re-crop the background scene to self.bg_x_range/self.bg_y_range, rebuild
        its range image, and refresh everything derived from it."""
        points = np.array(self.bg_pcd.points)
        in_x = (self.bg_x_range[0] < points[:, 0]) & (points[:, 0] < self.bg_x_range[1])
        in_y = (self.bg_y_range[0] < points[:, 1]) & (points[:, 1] < self.bg_y_range[1])
        bg_pcd = self.bg_pcd.select_by_index(np.where(in_x & in_y)[0].tolist())
        self.pcds["bg"] = bg_pcd
        if not bg_pcd.has_colors():
            color_by_dist(bg_pcd)

        self.bg_window.scene.clear_geometry()
        self.bg_window.scene.add_geometry("Points", bg_pcd, self.mat)
        if self.bg_window.scene.camera.get_field_of_view() == 90.0:
            bbox = self.bg_window.scene.bounding_box
            self.bg_window.setup_camera(60.0, bbox, bbox.get_center())

        dims = 1 + self.og_bg_range_image.shape[2]
        bg_range_image = np.zeros((self.height, self.width, dims), dtype="float64")
        bg_range_image[..., 1:] = self.og_bg_range_image

        in_x = (self.bg_x_range[0] < bg_range_image[..., 1]) & (
            bg_range_image[..., 1] < self.bg_x_range[1]
        )
        in_y = (self.bg_y_range[0] < bg_range_image[..., 2]) & (
            bg_range_image[..., 2] < self.bg_y_range[1]
        )
        bg_range_image[~(in_x & in_y)] = 0
        self.bg_range_image = bg_range_image
        img = create_depth_image(bg_range_image[..., 4], bg_range_image[..., 4].max())
        img.show()
        self.close_img = img

        self.create_grid_points("bg")
        self.create_ground_plane("bg")
        self.level_scene("bg")
        self.render_scene("bg")
        self.simulate_scene()

    def _on_obj_xy_changed(self, obj_xy: str) -> None:
        """Parse the "(x, y)" new-object-location text field and refresh the
        background scene's placement marker."""
        try:
            new_obj_xy = np.array(ast.literal_eval(obj_xy))
        except (ValueError, SyntaxError, TypeError):
            self._create_dialog(
                "Input Error", "Object (x, y)-coordinates must be a tuple of floats."
            )
            return

        previous_obj_xy = self.new_obj_xy
        self.new_obj_xy = new_obj_xy
        if not self._has_bg_points_near_grid():
            self.new_obj_xy = previous_obj_xy
            self.obj_xy.text_value = str(tuple(previous_obj_xy))
            self._create_dialog(
                "Input Error",
                "That location has no nearby background points to level against. "
                "Try a location within the background scene.",
            )
            return

        self.create_ground_plane("bg")
        self.create_grid_points("bg")
        self.level_scene("bg")
        self.render_scene("bg")
        self.simulate_scene()

    def _on_hit_thresh_changed(self, hit_thresh: str) -> None:
        try:
            self.hit_thresh = float(hit_thresh)
        except ValueError:
            self._create_dialog("Input Error", "Hit Threshold must be a float.")
            return

        self.simulate_scene()

    def _on_bg_min_dist_changed(self, bg_min_dist: str) -> None:
        try:
            self.bg_min_dist = float(bg_min_dist)
        except ValueError:
            self._create_dialog(
                "Input Error", "Background min. distance must be a float."
            )
            return

        self._on_bg_xy_range_changed()

    def _on_max_sector_changed(self, max_sector_angle: str) -> None:
        try:
            self.max_sector_angle = float(max_sector_angle)
        except ValueError:
            self._create_dialog("Input Error", "Maximum Sector must be a float.")
            return

        self.simulate_scene()

    def _on_neighbors_changed(self, neighbors: str) -> None:
        try:
            self.neighbors = int(neighbors)
        except ValueError:
            self._create_dialog("Input Error", "Neighbors must be an integer.")
            return

        self.simulate_scene()

    def _on_point_size_changed(self, point_size: str) -> None:
        try:
            self.point_size = float(point_size)
        except ValueError:
            self._create_dialog("Input Error", "Point Size must be a float.")
            return

        self.mat.point_size = self.point_size
        self.comb_mat.point_size = self.point_size
        self.render_scene("obj")
        self.render_scene("bg")
        self.simulate_scene()

    def level_scene(self, which: Which) -> None:
        """Estimate the ground plane under the object or background scene's
        leveling grid, and store the rotation/translation that levels it (i.e.,
        makes the ground plane horizontal at z=0).

        For "obj", also levels self.obj_bbox into self.level_obj_bbox. For "bg",
        instead stores the leveling transform itself (self.bg_t/self.bg_R, used by
        unlevel() to place an object into this background) and, if
        self.use_ground_z, the local ground height under the placement location
        (self.local_t).
        """
        which_pcd = self.pcds[which]
        points = np.array(which_pcd.points)

        grid_x, grid_y = self.get_grid_xy(which)
        x_start = grid_x - self.grids_info[which]["length"] / 2
        x_end = x_start + self.grids_info[which]["length"]
        y_start = grid_y - self.grids_info[which]["width"] / 2
        y_end = y_start + self.grids_info[which]["width"]
        in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
        in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
        region_pcd = which_pcd.select_by_index(np.where(in_x & in_y)[0].tolist())

        # For each grid point, find nearest neighbor in scene point cloud. These are our
        # "ground" points.
        nbrs = NearestNeighbors(n_neighbors=1).fit(np.array(region_pcd.points))
        grid_points = np.array(self.grid_pcds[which].points)
        (_, pcd_idxs) = nbrs.kneighbors(grid_points)
        pcd_idxs = np.unique(pcd_idxs)
        ground_pcd = region_pcd.select_by_index(pcd_idxs.tolist())
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

    def _on_use_colors(self, use_colors: bool) -> None:
        self.use_colors = use_colors
        self.simulate_scene()

    def _on_level_pcd(self, which: Which, checked: bool) -> None:
        """Toggle whether the object or background scene is shown in its leveled
        frame."""
        self.show_level_pcds[which] = checked
        self.render_scene(which)
        self.simulate_scene()

    def _on_level_location(self, level_loc: bool) -> None:
        self.level_at_location = level_loc
        self.create_ground_plane("bg")
        self.create_grid_points("bg")
        self.level_scene("bg")
        self.render_scene("bg")
        self.simulate_scene()

    def _on_use_ground_z(self, use_ground_z: bool) -> None:
        self.use_ground_z = use_ground_z
        if not use_ground_z:
            self.comb_window.scene.remove_geometry("Center Ground Point")

        self.level_scene("bg")
        self.render_scene("bg")
        self.simulate_scene()

    def _on_occlude(self, occlude: bool) -> None:
        self.occlude = occlude
        self.simulate_scene()

    def _on_sim_lidar(self, sim_lidar: bool) -> None:
        self.sim_lidar = sim_lidar
        self.simulate_scene()

    def _on_show_new_obj_loc(self, show_new_obj_loc: bool) -> None:
        self.show_new_obj_loc = show_new_obj_loc
        self.render_scene("bg")
        self.simulate_scene()

    def _on_show_lidar(self, show_lidar: bool) -> None:
        self.show_lidar = show_lidar
        if not show_lidar:
            self.comb_window.scene.remove_geometry("Top LiDAR Cone")
            self.comb_window.scene.remove_geometry("Bottom LiDAR Cone")

        self.simulate_scene()

    def _on_show_beams(self, show_beams: bool) -> None:
        self.show_beams = show_beams
        if not show_beams:
            self.comb_window.scene.remove_geometry("Unoccluded Point Cloud")
            for idx in range(self.prev_n_beams):
                self.comb_window.scene.remove_geometry(f"Arrow {idx}")

        self.simulate_scene()

    def _on_beam_range_changed(self, beam_range: str) -> None:
        """Parse the "(start, end)" beam index range text field; (0, 0) means
        auto-detect the beams near the object instead of a fixed range."""
        try:
            self.start_beam_idx, self.end_beam_idx = ast.literal_eval(beam_range)
            if (self.start_beam_idx < 0) or (self.end_beam_idx > len(self.Rs)):
                raise ValueError

            if self.start_beam_idx > self.end_beam_idx:
                raise ValueError

            is_zero = self.start_beam_idx != 0
            if (self.start_beam_idx == self.end_beam_idx) and is_zero:
                raise ValueError

        except (ValueError, SyntaxError, TypeError):
            self._create_dialog(
                "Input Error", "Beam Range must be a tuple of integers."
            )
            return

        self.simulate_scene()

    def get_grid_xy(self, which: Which) -> tuple[float, float]:
        """Get the (x, y) center of the object or background scene's leveling
        grid: the object's bounding box center for "obj", or either the new
        object location or the grid's default position for "bg", depending on
        self.level_at_location."""
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

    def _has_bg_points_near_grid(self) -> bool:
        """Whether the background scene has any points inside the leveling grid
        centered at its current grid location (see get_grid_xy). Used to reject
        a new object location before create_grid_points()/level_scene() -- which
        assume there's at least one point to find a ground plane from -- run
        against it."""
        (grid_x, grid_y) = self.get_grid_xy("bg")
        x_start = grid_x - self.grids_info["bg"]["length"] / 2
        x_end = x_start + self.grids_info["bg"]["length"]
        y_start = grid_y - self.grids_info["bg"]["width"] / 2
        y_end = y_start + self.grids_info["bg"]["width"]
        points = np.array(self.pcds["bg"].points)
        in_x = (x_start < points[:, 0]) & (points[:, 0] < x_end)
        in_y = (y_start < points[:, 1]) & (points[:, 1] < y_end)
        return bool(np.any(in_x & in_y))

    def create_grid_points(self, which: Which) -> None:
        """Build a flat grid of points, at the lowest z in the region, used by
        level_scene() to sample the ground surface under the object or
        background scene."""
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

    def create_ground_plane(self, which: Which) -> None:
        """Build a flat box mesh spanning the object or background scene's
        leveling grid, for visualizing where level_scene() samples the ground."""
        grid_width = self.grids_info[which]["width"]
        grid_length = self.grids_info[which]["length"]
        ground_plane = o3d.geometry.TriangleMesh.create_box(
            width=grid_length, height=grid_width, depth=0.01
        )
        (grid_x, grid_y) = self.get_grid_xy(which)
        center = np.array([grid_x, grid_y, 0])
        ground_plane.translate(center, relative=False)

        self.ground_planes[which] = ground_plane

    def _on_grid_points_changed(self, grid_points: str) -> None:
        try:
            self.grid_points = int(grid_points)
        except ValueError:
            self._create_dialog("Input Error", "Grid Points must be an integer.")
            return

        for which in ("obj", "bg"):
            self.create_grid_points(which)
            self.level_scene(which)

        self.render_scene("obj")
        self.render_scene("bg")
        self.simulate_scene()

    def _on_grid_dim_changed(
        self, which: Which, dim: Literal["length", "width"], label: str, value_str: str
    ) -> None:
        """Parse a grid length/width text field into self.grids_info[which][dim]
        and rebuild everything derived from the leveling grid."""
        try:
            self.grids_info[which][dim] = float(value_str)
        except ValueError:
            self._create_dialog("Input Error", f"{label} must be a float.")
            return

        self.create_grid_points(which)
        self.create_ground_plane(which)
        self.level_scene(which)
        self.render_scene(which)
        self.simulate_scene()

    def _on_show_grid_points(self, which: Which, checked: bool) -> None:
        """Toggle the leveling grid point overlay for the object or background
        scene."""
        self.show_grid_pcds[which] = checked
        self.render_scene(which)

    def _on_show_ground_plane(self, which: Which, checked: bool) -> None:
        """Toggle the leveling grid's ground-plane overlay for the object or
        background scene."""
        self.show_ground_planes[which] = checked
        self.render_scene(which)

    def _on_show_ground_points(self, which: Which, checked: bool) -> None:
        """Toggle the detected ground point overlay for the object or background
        scene."""
        self.show_ground_pcds[which] = checked
        self.render_scene(which)

    def render_beams(
        self, simulated_object_points: np.ndarray, moved_object_points: np.ndarray
    ) -> None:
        """Visualize the LiDAR beams used to determine which points on the object
        a beam would actually hit: draws the pre-unocclude ("Unoccluded") object
        point cloud, shows the range image it would project to, and draws an
        arrow for each beam that's either in self.start_beam_idx/self.end_beam_idx
        or (if that range is (0, 0)) auto-detected as passing near the object."""
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
        img = create_depth_image(range_image, self.bg_range_image[..., 4].max())
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
            # D_azims/D_elevs are beam directions relative to the optical center (see
            # get_Ds() and the matching shift in unocclude()), so the points need the
            # same shift before their angles are comparable.
            centered_points = moved_object_points[:, :3] - self.optical_center
            point_azims = np.arctan2(centered_points[:, 1], centered_points[:, 0])
            min_azim = point_azims.min() - self.azim_pm
            max_azim = point_azims.max() + self.azim_pm
            azim_mask = (min_azim < self.D_azims) & (self.D_azims < max_azim)

            point_norms = np.linalg.norm(centered_points, axis=1)
            point_elevs = np.arcsin(centered_points[:, 2] / point_norms)
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

    def simulate_scene(self) -> None:
        """Run the full augmentation pipeline and render the result in the
        combined scene view: place the object at self.new_obj_xy in the
        background's frame, keep only the object points a LiDAR beam would
        actually hit (unocclude()) if self.sim_lidar, then simulate
        occlusion by the background (project_points()) if self.occlude."""
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
            bg_t = np.zeros(3)
            bg_R = np.eye(3)

        if self.use_ground_z:
            bg_t += self.local_t

        object_pcd = object_pcd.crop(bbox)
        if self.show_level_pcds["obj"]:
            # Only keep points above the ground.
            object_pcd = object_pcd.select_by_index(
                np.where(np.array(object_pcd.points)[:, 2] > 0)[0].tolist()
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

    def create_range_images(self) -> None:
        """Show the simulated scene's range image, and a second image isolating
        just the pixels that came from the placed object (its "difference" from
        the background), alongside the background's own range image."""
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
        )
        img = create_depth_image(range_image[..., 4], range_image[..., 4].max())
        img.show()

        diff_image = np.array(img) * (range_image[..., 0] == 1)
        diff_img = Image.fromarray(diff_image.astype("uint8"))
        diff_img.show()
        self.close_img.show()


def main() -> None:
    Paved2Paradise().run()


if __name__ == "__main__":
    main()
