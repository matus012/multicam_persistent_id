"""Tests for mcreid.calib.schema — the calib.json pydantic contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mcreid.calib.schema import SCHEMA_VERSION, CameraCalib, GroundPlane, Intrinsics, RigCalib
from mcreid.sim.virtual_camera import VirtualCamera


def _intrinsics(**overrides: object) -> Intrinsics:
    kwargs: dict[str, object] = dict(
        fx=900.0,
        fy=900.0,
        cx=640.0,
        cy=360.0,
        dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_width=1280,
        image_height=720,
        rms_reproj_px=0.2,
        n_views=10,
    )
    kwargs.update(overrides)
    return Intrinsics(**kwargs)  # type: ignore[arg-type]


def _ground_plane(**overrides: object) -> GroundPlane:
    H = np.array([[0.01, 0.0, -1.0], [0.0, 0.01, -1.0], [0.0, 0.0, 1.0]])
    kwargs: dict[str, object] = dict(
        H_img2world=[[float(x) for x in row] for row in H],
        method="synthetic",
        rms_error_m=0.01,
        n_correspondences=4,
    )
    kwargs.update(overrides)
    return GroundPlane(**kwargs)  # type: ignore[arg-type]


def _camera_calib(camera_id: str = "camA") -> CameraCalib:
    return CameraCalib(camera_id=camera_id, intrinsics=_intrinsics(), ground=_ground_plane())


def _bedroom_rig_calib() -> RigCalib:
    cams = [
        VirtualCamera("cam0", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0),
        VirtualCamera("cam1", (5.7, 0.3, 1.6), yaw_deg=135.0, pitch_deg=20.0),
    ]
    return RigCalib(cameras=[c.to_calib(floor_extent_m=(0.0, 0.0, 6.0, 5.0)) for c in cams])


# --- Intrinsics -------------------------------------------------------------


def test_intrinsics_properties() -> None:
    intr = _intrinsics(fx=901.0, fy=902.0, cx=641.0, cy=361.0)
    assert intr.K.shape == (3, 3), "K must be 3x3"
    assert intr.K[0, 0] == pytest.approx(901.0)
    assert intr.K[1, 1] == pytest.approx(902.0)
    assert intr.K[0, 2] == pytest.approx(641.0)
    assert intr.K[1, 2] == pytest.approx(361.0)
    assert intr.image_size == (1280, 720), "image_size must be (width, height)"
    assert intr.dist.shape == (5,)


def test_intrinsics_rejects_principal_point_outside_image() -> None:
    with pytest.raises(ValidationError, match="principal point"):
        _intrinsics(cx=5000.0)


def test_intrinsics_rejects_non_finite_dist_coeffs() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _intrinsics(dist_coeffs=[float("nan"), 0.0, 0.0, 0.0, 0.0])


def test_intrinsics_from_matrices_round_trip() -> None:
    K = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.01, -0.02, 0.0, 0.0, 0.0])
    intr = Intrinsics.from_matrices(K, dist, (1280, 720), rms_reproj_px=0.3, n_views=12)
    assert np.allclose(intr.K, K), "K must round-trip through from_matrices"
    assert np.allclose(intr.dist, dist)
    assert intr.n_views == 12


def test_intrinsics_from_matrices_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="3x3"):
        Intrinsics.from_matrices(np.eye(2), np.zeros(5), (1280, 720), rms_reproj_px=0.0, n_views=1)


# --- GroundPlane -------------------------------------------------------------


def test_ground_plane_from_matrix_normalises_h22() -> None:
    H = np.array([[2.0, 0.0, -200.0], [0.0, 2.0, -200.0], [0.0, 0.0, 2.0]])
    gp = GroundPlane.from_matrix(H, method="synthetic", rms_error_m=0.0, n_correspondences=4)
    assert gp.H[2, 2] == pytest.approx(1.0), "H must be normalised so H[2,2] == 1"
    assert gp.H[0, 0] == pytest.approx(1.0)


def test_ground_plane_rejects_singular_homography() -> None:
    H = np.zeros((3, 3))
    H[2, 2] = 1.0
    with pytest.raises(ValidationError, match="singular"):
        GroundPlane(
            H_img2world=H.tolist(), method="synthetic", rms_error_m=0.0, n_correspondences=4
        )


def test_ground_plane_rejects_reversed_floor_extent() -> None:
    with pytest.raises(ValidationError, match="floor_extent_m"):
        _ground_plane(floor_extent_m=(5.0, 5.0, 0.0, 0.0))


def test_ground_plane_h_inv_is_inverse_of_h() -> None:
    gp = _ground_plane()
    identity = gp.H @ gp.H_inv
    assert np.allclose(identity, np.eye(3), atol=1e-9), "H_inv must invert H"


# --- CameraCalib --------------------------------------------------------------


def test_camera_calib_undistort_maps_needed_false_for_zero_distortion() -> None:
    cam = _camera_calib()
    assert cam.undistort_maps_needed() is False


def test_camera_calib_undistort_maps_needed_true_for_nonzero_distortion() -> None:
    intr = _intrinsics(dist_coeffs=[0.05, 0.0, 0.0, 0.0, 0.0])
    cam = CameraCalib(camera_id="camA", intrinsics=intr, ground=_ground_plane())
    assert cam.undistort_maps_needed() is True


# --- RigCalib ------------------------------------------------------------------


def test_rig_calib_rejects_duplicate_camera_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        RigCalib(cameras=[_camera_calib("camA"), _camera_calib("camA")])


def test_rig_calib_get_and_camera_ids() -> None:
    rig = RigCalib(cameras=[_camera_calib("camA"), _camera_calib("camB")])
    assert rig.camera_ids == ["camA", "camB"]
    assert rig.get("camB").camera_id == "camB"
    with pytest.raises(KeyError):
        rig.get("camZ")


def test_rig_calib_floor_extent_union() -> None:
    rig = _bedroom_rig_calib()
    x0, y0, x1, y1 = rig.floor_extent()
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 6.0, 5.0))


def test_rig_calib_floor_extent_raises_without_any_declared() -> None:
    rig = RigCalib(cameras=[_camera_calib("camA")])
    with pytest.raises(ValueError, match="floor_extent_m"):
        rig.floor_extent()


def test_rig_calib_save_load_round_trip(tmp_path: Path) -> None:
    rig = _bedroom_rig_calib()
    path = rig.save(tmp_path / "sub" / "calib.json")
    assert path.is_file(), "save() must create the file (and parent dirs)"

    loaded = RigCalib.load(path)
    assert loaded.camera_ids == rig.camera_ids
    assert loaded.schema_version == SCHEMA_VERSION
    for cam_id in rig.camera_ids:
        assert np.allclose(loaded.get(cam_id).ground.H, rig.get(cam_id).ground.H)


def test_rig_calib_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RigCalib.load(tmp_path / "does_not_exist.json")


def test_rig_calib_load_schema_version_mismatch_raises(tmp_path: Path) -> None:
    rig = _bedroom_rig_calib()
    path = rig.save(tmp_path / "calib.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version mismatch"):
        RigCalib.load(path)
