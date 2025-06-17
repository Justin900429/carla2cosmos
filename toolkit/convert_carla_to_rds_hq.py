import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import imageio
from tqdm import tqdm
import tyro
from termcolor import cprint
import open3d as o3d
from utils.opendriver_parser import load_xodr_and_parse
from utils.wds_utils import write_to_tar, encode_dict_to_npz_bytes


def convert_carla_intrinsics(
    root_dir: Path,
    out_dir: Path,
    clip_id: str,
):
    """
    Read the *camera_front.json* file and write the intrinsic matrix to a tar file.

    Parameters
    ----------
    root_dir : Path
        Folder that already contains `calibration/`.
    out_dir : Path
        Root where the `pinhole_intrinsic/<clip_id>.tar` should be created.
    clip_id : str
    """
    sample = {"__key__": clip_id}

    with (root_dir / "calibration" / "camera_front.json").open() as fp:
        data = json.load(fp)

    intrinsic = data["intrinsic"]
    sample["pinhole_intrinsic.front.npy"] = np.asarray(
        [
            intrinsic["fx"],
            intrinsic["fy"],
            intrinsic["cx"],
            intrinsic["cy"],
            intrinsic["width"],
            intrinsic["height"],
        ],
        dtype=np.float32,
    )

    write_to_tar(sample, out_dir / "pinhole_intrinsic" / f"{clip_id}.tar")


def convert_carla_hdmap(
    record_root: Path,
    out_dir: Path,
    clip_id: str,
):
    """
    Parse the OpenDRIVE (*.xodr) map saved by *record_clip()* and emit four
    tars (two are currently ignored: speed_bump and driveway in Waymo):

        3d_lanes/<clip_id>.tar
        3d_lanelines/<clip_id>.tar
        3d_road_boundaries/<clip_id>.tar
        3d_crosswalks/<clip_id>.tar

    Each tar contains one JSON whose schema matches the Waymo converter:
        { "__key__": clip_id,
          "<layer>.json": { "labels": [ { "labelData":{ "shape3d":{…}}}, … ]}}

    Parameters
    ----------
    record_root : Path
        Folder that already contains `hdmap/` and `calibration/`.
    out_dir : Path
        Root where the `3d_lanes/<clip_id>.tar`, `3d_lanelines/<clip_id>.tar`, `3d_road_boundaries/<clip_id>.tar`, and `3d_crosswalks/<clip_id>.tar` should be created.
    clip_id : str
        Identifier for this clip (becomes the tar file name and the __key__).
    """

    crosswalks_sample = {"__key__": clip_id, "crosswalks.json": {"labels": []}}
    xodr_file = record_root / "hdmap" / "static_map.xodr"
    road_network = load_xodr_and_parse(xodr_file)
    for road in road_network.roads:
        for crosswalk in road.crosswalks:
            s = crosswalk.sPos
            t = crosswalk.tPos
            hdg_obj = crosswalk.hdg

            # 1) locate object origin in world
            res = road.planView.planview_at(s)
            if res is None:
                continue
            x_ref, y_ref, hdg_ref = res
            x_obj = x_ref - t * math.sin(hdg_ref)
            y_obj = y_ref + t * math.cos(hdg_ref)
            hdg = hdg_ref + hdg_obj

            # 2) transform every cornerLocal (u,v) into world
            polys = []
            for u, v in crosswalk.corners:
                x = x_obj + u * math.cos(hdg) - v * math.sin(hdg)
                y = y_obj + u * math.sin(hdg) + v * math.cos(hdg)
                polys.append([x, y, 0.0])

            crosswalks_sample["crosswalks.json"]["labels"].append(
                {
                    "labelData": {
                        "shape3d": {"surface": {"vertices": polys}},
                    }
                }
            )
    write_to_tar(crosswalks_sample, out_dir / "3d_crosswalks" / f"{clip_id}.tar")

    boundaries_sample = {"__key__": clip_id}
    with open(record_root / "hdmap" / "road_boundaries.json", "r") as fp:
        boundaries_sample["road_boundaries.json"] = json.load(fp)
    write_to_tar(boundaries_sample, out_dir / "3d_road_boundaries" / f"{clip_id}.tar")

    lanelines_sample = {"__key__": clip_id}
    with open(record_root / "hdmap" / "lanelines.json", "r") as fp:
        lanelines_sample["lanelines.json"] = json.load(fp)
    write_to_tar(lanelines_sample, out_dir / "3d_lanelines" / f"{clip_id}.tar")

    traffic_lights_sample = {"__key__": clip_id}
    with open(record_root / "hdmap" / "traffic_lights.json", "r") as fp:
        traffic_lights_sample["traffic_lights.json"] = json.load(fp)
    write_to_tar(traffic_lights_sample, out_dir / "3d_traffic_lights" / f"{clip_id}.tar")

    traffic_signs_sample = {"__key__": clip_id}
    with open(record_root / "hdmap" / "traffic_signs.json", "r") as fp:
        traffic_signs_sample["traffic_signs.json"] = json.load(fp)
    write_to_tar(traffic_signs_sample, out_dir / "3d_traffic_signs" / f"{clip_id}.tar")
    poles_sample = {"__key__": clip_id}

    with open(record_root / "hdmap" / "poles.json", "r") as fp:
        poles_sample["poles.json"] = json.load(fp)
    write_to_tar(poles_sample, out_dir / "3d_poles" / f"{clip_id}.tar")


def convert_carla_pose(
    record_root: Path,
    out_dir: Path,
    clip_id: str,
    index_scale_ratio: int = 1,  # CARLA is already 30 Hz → keep 1
):
    """
    Read *ego_pose/*.npy (world ← ego) and each camera’s extrinsic
    (ego ← camera) from *calibration/camera_*.json*, then write two
    tars:

    • pose/<clip_id>.tar              – camera-to-world in **OpenCV** convention
    • vehicle_pose/<clip_id>.tar      – vehicle-to-world (FLU convention)

    File names inside each tar follow the Waymo converter pattern.

    Parameters
    ----------
    record_root : Path
        Folder that already contains `ego_pose/` and `calibration/`.
    out_dir : Path
        Root where the `pose/<clip_id>.tar` and `vehicle_pose/<clip_id>.tar` should be created.
    clip_id : str
        Identifier for this clip (becomes the tar file name and the __key__).
    index_scale_ratio : int, default 1
        Multiply the frame index by this factor when naming files.
        • Waymo uses 3 because they later up-sample 10 Hz → 30 Hz.
        • Our CARLA recorder is already 30 Hz, so **leave this as 1**.
    """
    # ──────────────────────────────────────────────────────────────────────
    # load camera   extrinsic matrices  (ego ← camera)  in FLU / RDS frame
    # ──────────────────────────────────────────────────────────────────────
    cam_to_vehicle = {}
    calib_dir = record_root / "calibration"
    for js in calib_dir.glob("camera_*.json"):
        cam_name = js.stem.split("camera_")[1]  # e.g. 'front'
        with js.open() as fp:
            cam_to_vehicle[cam_name] = np.asarray(json.load(fp)["extrinsic"], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # prepare samples
    # ──────────────────────────────────────────────────────────────────────
    cam_sample = {"__key__": clip_id}
    veh_sample = {"__key__": clip_id}

    pose_dir = record_root / "ego_pose"
    pose_files = sorted(pose_dir.glob("*.npy"))

    for idx, fp in enumerate(pose_files):
        frame_idx = idx * index_scale_ratio
        veh_to_world = np.load(fp)  # (4,4) world ← ego, FLU

        # 1. vehicle pose ----------------------------------------------------
        veh_sample[f"{frame_idx:06d}.vehicle_pose.npy"] = veh_to_world

        # 2. camera poses ----------------------------------------------------
        for cam_name, cam2veh in cam_to_vehicle.items():
            cam_to_world_flu = veh_to_world @ cam2veh  # world ← cam  (FLU)
            # convert FLU (x-fwd, y-left, z-up) ➜ OpenCV (x-right, y-down, z-fwd)
            cam_to_world_cv = np.concatenate(
                (
                    -cam_to_world_flu[:, 1:2],  #  -Y  → X'
                    -cam_to_world_flu[:, 2:3],  #  -Z  → Y'
                    cam_to_world_flu[:, 0:1],  #   X  → Z'
                    cam_to_world_flu[:, 3:4],  #   t  (unchanged)
                ),
                axis=1,
            )
            cam_sample[f"{frame_idx:06d}.pose.{cam_name}.npy"] = cam_to_world_cv

    write_to_tar(cam_sample, out_dir / "pose" / f"{clip_id}.tar")
    write_to_tar(veh_sample, out_dir / "vehicle_pose" / f"{clip_id}.tar")


def convert_carla_timestamp(
    root_dir: Path,
    out_dir: Path,
    clip_id: str,
    index_scale_ratio: int = 1,
) -> None:
    """
    Read the *timestamp.json* dumped by `record_clip()` and store every
    per-frame epoch time (in seconds) as text files inside a tar.

    Parameters
    ----------
    record_root : Path
        Folder that already contains `timestamp.json`.
    out_dir : Path
        Root where the `timestamp/<clip_id>.tar` should be created.
    clip_id : str
        Identifier for this clip (becomes the tar file name and the __key__).
    index_scale_ratio : int, default 1
        Multiply the frame index by this factor when naming files.
        • Waymo uses 3 because they later up-sample 10 Hz → 30 Hz.
        • Our CARLA recorder is already 30 Hz, so **leave this as 1**.
    """
    ts_path = root_dir / "timestamp.json"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)

    seconds = json.loads(ts_path.read_text())

    sample = {"__key__": clip_id}
    for idx, t_sec in enumerate(seconds):
        fname = f"{idx * index_scale_ratio:06d}.timestamp_micros.txt"
        sample[fname] = str(t_sec)

    write_to_tar(sample, out_dir / "timestamp" / f"{clip_id}.tar")


def convert_carla_bbox(
    root_dir: Path,
    out_dir: Path,
    clip_id: str,
    fps: int = 30,
    min_moving_speed: float = 0.2,
    index_scale_ratio: int = 1,
):
    """
    Read the per-frame JSON files written by *record_clip()* (folder
    **labels_3d/**) and pack them into a tar. Since the center of the
    bounding box is represent in the world coordinate, no conversion
    is needed.

    Each JSON is a list of dicts written by the recorder:
        {
          "center": [x, y, z],
          "size":   [l, w, h],
          "yaw":    float (rad, RDS convention, right-hand ENU),
          "class":  "Vehicle"|"Pedestrian"|…,
          "track_id": int
        }

    Parameters
    ----------
    root_dir : Path
        Folder that already contains `labels_3d/` and `timestamp.json`.
    out_dir : Path
        Root where the `all_object_info/<clip_id>.tar` should be created.
    clip_id : str
        Identifier for this clip (becomes the tar file name and the __key__).
    index_scale_ratio : int, default 1
        Multiply the frame index by this factor when naming files.
        • Waymo uses 3 because they later up-sample 10 Hz → 30 Hz.
        • Our CARLA recorder is already 30 Hz, so **leave this as 1**.
    """

    label_dir = root_dir / "labels_3d"
    frame_files = sorted(label_dir.glob("*.json"))

    VALID_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]

    # keep last position per track id to decide if the object is moving
    prev_center: dict[int, np.ndarray] = {}

    sample = {"__key__": clip_id}

    for f_idx, fp in enumerate(frame_files):
        frame_key = f"{f_idx * index_scale_ratio:06d}.all_object_info.json"
        sample[frame_key] = {}

        with fp.open() as jf:
            objects = json.load(jf)

        for obj in objects:
            if obj["class"] not in VALID_CLASSES:
                continue

            obj_id = str(obj["track_id"])  # keys must be str
            x, y, z = obj["center"]
            l, w, h = obj["size"]
            yaw = obj["yaw"]  # already rad

            # --- SE(3) object → world ------------------------------------------------
            c, s = np.cos(yaw), np.sin(yaw)
            obj_to_world = np.eye(4, dtype=float)
            obj_to_world[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            obj_to_world[:3, 3] = [x, y, z]

            # --- speed --------------------------------------------------------------
            center_now = np.array([x, y, z])
            prev = prev_center.get(obj["track_id"])
            if prev is None:
                speed = 0.0
            else:
                dist = np.linalg.norm(center_now - prev)
                speed = dist * fps  # m s-¹ (dist per frame × fps)
            prev_center[obj["track_id"]] = center_now
            is_moving = bool(speed > min_moving_speed)

            # --- commit -------------------------------------------------------------
            sample[frame_key][obj_id] = {
                "object_to_world": obj_to_world.tolist(),
                "object_lwh": [l, w, h],
                "object_is_moving": is_moving,
                "object_type": obj["class"],
            }

    write_to_tar(sample, out_dir / "all_object_info" / f"{clip_id}.tar")


def convert_carla_lidar(
    root_dir: Path,
    out_dir: Path,
    clip_id: str,
    index_scale_ratio: int = 1,  # keep 1 – we already record at 30 Hz
):
    """
    Pack CARLA LiDAR frames into a tar.

    Source layout (created by *record_clip*):
        root_dir/
            ego_pose/0000.npy …
            lidar/0000.npz   …     # {'points': (N,4) xyz-i}
            calibration/lidar.json # {'extrinsic': 4×4 ego←lidar}

    Output inside the tar:
        {frame_idx:06d}.lidar.npz
            ├─ 'xyz'            (N,3)  float32
            └─ 'lidar_to_world' (4,4)  float32

    Parameters
    ----------
    root_dir : Path
        Folder that already contains `lidar/` and `ego_pose/`.
    out_dir : Path
        Root where the `lidar_raw/<clip_id>.tar` should be created.
    clip_id : str
        Identifier for this clip (becomes the tar file name and the __key__).
    index_scale_ratio : int, default 1
        Multiply the frame index by this factor when naming files.
        • Waymo uses 3 because they later up-sample 10 Hz → 30 Hz.
        • Our CARLA recorder is already 30 Hz, so **leave this as 1**.
    """
    # ---------------------------------------------------- load static extrinsic
    lidar_to_ego = np.asarray(
        json.loads((root_dir / "calibration" / "lidar.json").read_text())["extrinsic"],
        dtype=np.float32,
    )  # ego ← lidar  (4×4)

    # ---------------------------------------------------- helper for .npz bytes
    sample = {"__key__": clip_id}

    lidar_files = sorted((root_dir / "lidar").glob("*.ply"))
    for idx, lidar_fp in enumerate(lidar_files):
        # # world ← ego  (4×4) for this frame
        ego_to_world = np.load(root_dir / "ego_pose" / f"{idx:04d}.npy")
        # world ← lidar  = (world ← ego) · (ego ← lidar)
        lidar_to_world = (ego_to_world @ lidar_to_ego).astype(np.float32)

        # points: take xyz only
        pts = np.asarray(o3d.io.read_point_cloud(lidar_fp).points)
        pts[:, 1] *= -1

        frame_key = f"{idx * index_scale_ratio:06d}.lidar_raw.npz"
        sample[frame_key] = encode_dict_to_npz_bytes({"xyz": pts, "lidar_to_world": lidar_to_world})

    write_to_tar(sample, out_dir / "lidar_raw" / f"{clip_id}.tar")


def convert_carla_image(
    root_dir: Path,
    out_dir: Path,
    clip_id: str,
    fps: int = 30,
    single_camera: bool = False,
):
    """
    Turn each saved CARLA camera-frame sequence (**camera_* sub-folders**) into an
    MP4, matching the folder/filename convention used by the Waymo converter.

    Source layout written by *record_clip()*:
        root_dir/
            camera_front/0000.png …
            camera_left/ 0000.png …   # optional
            …

    Output:
        <out_dir>/pinhole_<name>/<clip_id>.mp4

    Parameters
    ----------
    root_dir : Path
        Folder that already contains `camera_front/` and `camera_left/`.
    out_dir : Path
        Root where the `pinhole_<name>/<clip_id>.mp4` should be created.
    clip_id : str
        Identifier for this clip (becomes the tar file name and the __key__).
    fps : int, default 30
        Frames per second.
    single_camera : bool, default False
        If True, only process the `camera_front/` folder.
    """
    # -------------------------------------------------------------------------
    # discover cameras – any sub-folder whose name starts with 'camera_'
    # -------------------------------------------------------------------------
    camera_dirs = [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("camera_")]
    if single_camera:
        camera_dirs = [p for p in camera_dirs if p.name == "camera_front"]

    for cam_dir in camera_dirs:
        cam_name = cam_dir.name.split("camera_")[1]  # 'front', 'left', …

        # collect frames in index order
        frame_paths = sorted(cam_dir.glob("*.png"))
        if not frame_paths:
            continue

        out_mp4 = out_dir / f"pinhole_{cam_name}" / f"{clip_id}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # write video – use imageio like the Waymo script
        # ---------------------------------------------------------------------
        writer = imageio.get_writer(
            out_mp4.as_posix(),
            fps=fps,  # CARLA recorder is already 30 Hz
            macro_block_size=None,  # exact frame count, no 16× rounding
        )
        for img_path in frame_paths:
            writer.append_data(iio.imread(img_path))
        writer.close()
        cprint(f"Saved {out_mp4}", "green")


def convert_carla_to_wds(
    root_dir: Path,
    out_dir: Path,
    clip_id: str,
    single_camera: bool = False,
):
    clip_id = clip_id.zfill(4)
    root_dir = root_dir / f"clip_{clip_id}"
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")

    if (root_dir / "lidar_raw" / f"{clip_id}.tar").exists():
        print(f"Skipping {clip_id} because it already exists")
        return

    convert_carla_intrinsics(root_dir, out_dir, clip_id)
    convert_carla_hdmap(root_dir, out_dir, clip_id)
    convert_carla_pose(root_dir, out_dir, clip_id)
    convert_carla_timestamp(root_dir, out_dir, clip_id)
    convert_carla_bbox(root_dir, out_dir, clip_id)
    convert_carla_lidar(root_dir, out_dir, clip_id)
    convert_carla_image(root_dir, out_dir, clip_id, single_camera=single_camera)


def main(
    root_dir: Path,
    out_dir: Path,
    num_workers: int = 1,
    single_camera: bool = False,
):
    all_clips = sorted(root_dir.glob("clip_*"))
    print(f"Found {len(all_clips)} clips")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                convert_carla_to_wds,
                root_dir=root_dir,
                out_dir=out_dir,
                clip_id=str(int(os.path.basename(clip_id).split("_")[1])),
                single_camera=single_camera,
            )
            for clip_id in all_clips
        ]

        for future in tqdm(as_completed(futures), total=len(all_clips), desc="Converting"):
            future.result()


if __name__ == "__main__":
    tyro.cli(main)
