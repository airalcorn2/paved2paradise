import numpy as np
import open3d as o3d

from core import move_object_to_position, project_points, unlevel, unocclude
from scipy.spatial.transform import Rotation


def test_move_object_to_position_rotates_and_translates():
    center = np.array([5.0, 0.0, 1.0])
    bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), [1.0, 1.0, 1.0])
    points = np.array([center])
    xy = np.array([0.0, 5.0])  # Same distance from the origin, rotated 90 degrees.

    (new_points, new_bbox) = move_object_to_position(points, bbox, xy)

    assert np.allclose(new_points[0], [0.0, 5.0, 1.0])
    assert np.allclose(new_bbox.center, [0.0, 5.0, 1.0])


def test_unlevel_inverts_the_leveling_transform():
    R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t_level = np.array([1.0, -2.0, 0.5])

    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, 3))
    leveled_points = (R @ points.T).T + t_level

    bg_t = -t_level
    bg_R = R
    bbox = o3d.geometry.OrientedBoundingBox(leveled_points[0], np.eye(3), [1, 1, 1])

    (recovered_points, recovered_center, _) = unlevel(
        leveled_points.copy(), bg_t, bg_R, bbox
    )

    assert np.allclose(recovered_points, points)
    assert np.allclose(recovered_center, points[0])


def test_unocclude_keeps_points_near_a_ray_and_drops_distant_ones():
    optical_center = np.zeros(3)
    D_elevs = np.array([0.0])
    D_azims = np.array([0.0])
    all_Ds = np.array([[1.0, 0.0, 0.0]])

    near_ray_point = np.array([[5.0, 0.002, 0.0]])
    kept = unocclude(
        near_ray_point, optical_center, 0.1, 0.1, D_elevs, 360, D_azims, all_Ds, 0.01, 1
    )
    assert np.allclose(kept, [[5.0, 0.0, 0.0]])

    far_from_ray_point = np.array([[5.0, 1.0, 0.0]])
    dropped = unocclude(
        far_from_ray_point,
        optical_center,
        0.5,
        0.5,
        D_elevs,
        360,
        D_azims,
        all_Ds,
        0.01,
        1,
    )
    assert dropped.shape == (0, 3)


def test_unocclude_does_not_mutate_the_caller_array():
    optical_center = np.array([1.0, 0.0, 0.0])
    D_elevs = np.array([0.0])
    D_azims = np.array([0.0])
    all_Ds = np.array([[1.0, 0.0, 0.0]])

    points = np.array([[5.0, 0.0, 0.0]])
    points_before = points.copy()
    unocclude(points, optical_center, 0.1, 0.1, D_elevs, 360, D_azims, all_Ds, 0.01, 1)

    assert np.allclose(points, points_before)


def test_project_points_places_a_point_in_the_expected_pixel():
    range_image = np.zeros((1, 4, 5), dtype="float64")
    points = np.array([[1.0, 0.0, 0.0]])
    rot_mat = np.eye(3)
    optical_center = np.zeros(3)
    beams = np.array([0.0])

    out = project_points(range_image, points, 1, 4, rot_mat, optical_center, beams, 0.1, 1)

    assert np.allclose(out[0, 2], [1.0, 1.0, 0.0, 0.0, 1.0])
    assert np.allclose(out[0, [0, 1, 3]], 0.0)


def test_project_points_keeps_closer_point_on_collision():
    range_image = np.zeros((1, 4, 5), dtype="float64")
    rot_mat = np.eye(3)
    optical_center = np.zeros(3)
    beams = np.array([0.0])

    near_then_far = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    out = project_points(
        range_image.copy(), near_then_far, 1, 4, rot_mat, optical_center, beams, 0.1, 1
    )
    assert out[0, 2, 4] == 1.0

    far_then_near = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = project_points(
        range_image.copy(), far_then_near, 1, 4, rot_mat, optical_center, beams, 0.1, 1
    )
    assert out[0, 2, 4] == 1.0
