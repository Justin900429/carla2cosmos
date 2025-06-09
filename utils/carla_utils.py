import numpy as np
import carla
import pathlib
import imageio
import imageio.v3 as iio

################################################################################
# Coordinate helpers – convert CARLA left‑hand ➜ right‑hand
################################################################################
_MIRROR_Y = np.diag([1, -1, 1, 1])  # homogeneous matrix that flips Y‑axis


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tf_to_rds(carla_tf: carla.Transform) -> np.ndarray:
    """Return 4×4 SE(3) matrix (world←ego) in right‑hand ENU.

    This should be the same as the following code:
    ```python
        # translation
        m = np.eye(4)
        loc = carla_tf.location
        m[:3, 3] = np.array([loc.x, -loc.y, loc.z])  # mirror Y
        # rotation – CARLA gives degrees, left‑handed ENU
        r = carla_tf.rotation
        R_lh = _rot_z(np.radians(r.yaw)) @ _rot_y(np.radians(r.pitch)) @ _rot_x(np.radians(r.roll))
        # convert to right‑hand by mirroring Y on both sides (change-of-basis)
        R_rh = _MIRROR_Y[:3, :3] @ R_lh @ _MIRROR_Y[:3, :3]
        m[:3, :3] = R_rh
    ```
    """

    se3_matrix = np.array(carla_tf.get_matrix())
    se3_matrix[1, 3] *= -1
    se3_matrix[:3, :3] = _MIRROR_Y[:3, :3] @ se3_matrix[:3, :3] @ _MIRROR_Y[:3, :3]

    return se3_matrix


def points_to_rds(points: np.ndarray) -> np.ndarray:
    """Mirror Y coordinate in‑place (x y z i)."""
    points[:, 1] *= -1.0
    return points


def yaw_to_rds(yaw_deg: float) -> float:
    return -np.radians(yaw_deg)  # negate + deg→rad


def build_mp4(
    frame_paths: list[pathlib.Path],
    out_file: pathlib.Path,
    fps: int = 30,
    quality: float = 5.0,
    bitrate: int | None = None,
    macro_block_size: int | None = 16,
):
    list_of_frames = [iio.imread(str(frame)) for frame in frame_paths]
    with imageio.get_writer(
        str(out_file),
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    ) as writer:
        for frame in list_of_frames:
            writer.append_data(frame)
    logger.info("Video written to %s", out_file)
