"""Pure point-cloud functions implementing the Paved2Paradise augmentation pipeline.

These have no GUI or visualization dependency, so they can be imported and tested (see
tests/test_core.py) without launching Open3D's GUI module, and reused directly for
batch dataset generation.
"""

import numpy as np
import open3d as o3d

from numba import float64, int64, njit
from scipy.spatial.transform import Rotation


def move_object_to_position(
    points: np.ndarray, bbox: o3d.geometry.OrientedBoundingBox, xy: np.ndarray
) -> tuple[np.ndarray, o3d.geometry.OrientedBoundingBox]:
    """Rotate and translate an object's points/bounding box to a new (x, y) location.

    The object is rotated about the z-axis so its bearing from the origin matches
    `xy`'s bearing, then shifted radially so its distance from the origin matches
    `xy`'s distance. This keeps the object's orientation relative to the sensor
    perspective-consistent with having originally appeared at `xy`.

    Returns the transformed points and a new bounding box at the same location
    (`bbox` itself is left unmodified).
    """
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
    new_bbox = o3d.geometry.OrientedBoundingBox(new_center, new_R, bbox.extent)

    return (new_points, new_bbox)


def unlevel(
    moved_object_points: np.ndarray,
    bg_t: np.ndarray,
    bg_R: np.ndarray,
    moved_bbox: o3d.geometry.OrientedBoundingBox,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undo a background scene's leveling transform for an object placed in it.

    `bg_t`/`bg_R` are the translation/rotation that were applied to level the
    background scene (see `Paved2Paradise.level_scene` in paved2paradise.py). This
    applies their inverse to bring the object back into the background scene's
    original, unleveled coordinate frame.

    Returns the object's points, its bounding box center, and its bounding box
    rotation matrix, all in that original frame.
    """
    moved_object_points = moved_object_points + bg_t
    moved_object_points = (bg_R.T @ moved_object_points.T).T
    moved_center = bg_R.T @ (moved_bbox.center + bg_t)
    moved_bbox_R = bg_R.T @ moved_bbox.R

    return (moved_object_points, moved_center, moved_bbox_R)


def unocclude(
    object_points: np.ndarray,
    optical_center: np.ndarray,
    azim_pm: float,
    elev_pm: float,
    D_elevs: np.ndarray,
    max_sector_angle: float,
    D_azims: np.ndarray,
    all_Ds: np.ndarray,
    hit_thresh: float,
    max_neighbors: int,
) -> np.ndarray:
    """Keep only the object points a LiDAR beam would actually hit from this position.

    For each beam ray direction (given by `all_Ds`, with azimuth/elevation `D_azims`/
    `D_elevs`) that passes near the object, finds the closest object point(s) within
    `hit_thresh` of that ray and projects them onto it, simulating how the object
    would self-occlude as seen from the sensor at `optical_center`. `azim_pm`/
    `elev_pm` are the angular size of one beam column/row, used to decide which
    points/rays are "near" each other; `max_sector_angle` (degrees) batches the
    search into azimuthal slices for efficiency; `max_neighbors` caps how many nearby
    points are averaged per ray.

    Does not mutate `object_points`. Returns the surviving (hit) points, in the same
    frame as `object_points` and with any extra (e.g., color) columns preserved.
    """
    # The points from the Ouster are in the "Sensor Coordinate Frame" by default, so
    # we have to translate the points relative to the sensor's origin.
    # See: https://static.ouster.dev/sdk-docs/python/api/client.html#ouster.sdk.client.XYZLut
    # and: https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html#sensor-coordinate-frame.
    object_points = object_points.copy()
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
    range_image: np.ndarray,
    points: np.ndarray,
    height: int,
    width: int,
    rot_mat: np.ndarray,
    optical_center: np.ndarray,
    beams: np.ndarray,
    bg_min_dist: float,
    obj_idx: int,
) -> np.ndarray:
    """Rasterize 3D points into a LiDAR range image, keeping the closer point per pixel.

    For each point, projects it to a (row, column) pixel using `beams` (the sensor's
    per-row elevation angles) and `rot_mat`/`optical_center` (to transform into the
    LiDAR frame), and writes it into `range_image` -- [source id, x, y, z, depth, and
    optionally color] -- if that pixel is empty (its stored depth is below
    `bg_min_dist`) or the new point is closer than what's already there. `obj_idx` is
    the source id written for points from this call, letting callers later tell which
    pixels came from this point set.

    Mutates and returns `range_image`.
    """
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

        old_depth = range_image[v, u, 4]
        if (old_depth < bg_min_dist) or (depth < old_depth):
            range_image[v, u, 0] = obj_idx
            range_image[v, u, 1] = x
            range_image[v, u, 2] = y
            range_image[v, u, 3] = z
            range_image[v, u, 4] = depth
            if has_color:
                range_image[v, u, 5:] = point[3:]

    return range_image
