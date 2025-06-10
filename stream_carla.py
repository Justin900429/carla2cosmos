import json
import math
import pathlib
import queue

import logging
import carla
import cv2
import numpy as np
from tqdm import tqdm
import tyro
from manager import CarlaManager
import attrs
from misc.custom_logger import setup_logging
from config.stream_config import StreamConfig, serialize
from utils.carla_utils import tf_to_rds, points_to_rds, yaw_to_rds, build_mp4

CLASS_MAP = {
    "vehicle": "Vehicle",
    "walker": "Pedestrian",
    "traffic": "TrafficSign",
}


logger: logging.Logger


def make_queue(sensor: carla.Sensor):
    q: "queue.Queue[carla.SensorData]" = queue.Queue()
    sensor.listen(q.put)
    return q


def get_data_from_queue(q: queue.Queue[carla.SensorData], frame: int, timeout: float = 1.0):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        logger.warning(f"missing {q.name} frame {frame}")
        return None


def record_clip(
    carla_manager: CarlaManager,
    target_out_dir: pathlib.Path,
    config: StreamConfig,
):
    carla_manager.load_town(config.town)
    world_manager = carla_manager.world_manager
    existing_agents: list[carla.Location] = []

    # —— spawn ego vehicle
    ego = world_manager.random_spawn_car(
        config.ego.vehicle_model,
        autopilot=config.ego.autopilot,
        existing_agents=existing_agents,
    )

    # —— attach LiDAR
    lidar_bp = world_manager.find_blueprint("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", str(config.lidar.range))
    lidar_bp.set_attribute("rotation_frequency", str(config.lidar.rotation_frequency))
    lidar_bp.set_attribute("points_per_second", str(config.lidar.points_per_second))
    lidar_bp.set_attribute("channels", str(config.lidar.channels))
    lidar_bp.set_attribute("upper_fov", str(config.lidar.upper_fov))
    lidar_bp.set_attribute("lower_fov", str(config.lidar.lower_fov))
    lidar_tf = carla.Transform(carla.Location(z=2.2))  # simple roof mount
    lidar = world_manager.spawn_actor(lidar_bp, lidar_tf, attach_to=ego)

    # ── Front RGB camera ───────────────────────────────────────────────────────
    cam_bp = world_manager.find_blueprint("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(config.camera.image_size_x))
    cam_bp.set_attribute("image_size_y", str(config.camera.image_size_y))
    cam_bp.set_attribute("fov", str(config.camera.fov))
    cam_bp.set_attribute("sensor_tick", str(config.camera.sensor_tick))  # 30 Hz
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.5))  # bonnet-edge mount
    camera = world_manager.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # ── sensor queues
    lidar_q = make_queue(lidar)
    cam_q = make_queue(camera)

    # —— spawn background traffic
    world_manager.random_spawn_cars_with_nums(
        autopilot=config.autopilot,
        spawn_nums=config.num_vehicles,
        existing_agents=existing_agents,
    )
    walkers = world_manager.random_spawn_walkers_with_nums(
        spawn_nums=config.num_walkers,
        existing_agents=existing_agents,
    )
    world_manager.world.set_pedestrians_cross_factor(config.percentage_pedestrians_crossing)
    if config.autopilot:
        world_manager.tick()
        world_manager.set_ai_walkers(walkers)

    for d in ["ego_pose", "lidar", "camera_front", "labels_3d", "calibration", "hdmap"]:
        (target_out_dir / d).mkdir(parents=True, exist_ok=True)

    # —— write static map once
    xodr_path = target_out_dir / "hdmap" / "static_map.xodr"
    with open(xodr_path, "w", encoding="utf-8") as fp:
        fp.write(world_manager.map.to_opendrive())

    # ── calibration: LiDAR extrinsic ──────────────────────────────────────────
    with open(target_out_dir / "calibration" / "lidar.json", "w") as fp:
        json.dump({"extrinsic": tf_to_rds(lidar_tf).tolist()}, fp, indent=2)

    # ── calibration: camera extrinsic + intrinsic ─────────────────────────────
    width, height, fov = config.camera.image_size_x, config.camera.image_size_y, config.camera.fov
    fx = fy = (width / 2) / math.tan(math.radians(fov / 2))
    cam_calib = {
        "extrinsic": tf_to_rds(cam_tf).tolist(),
        "intrinsic": {
            "fx": fx,
            "fy": fy,
            "cx": width / 2,
            "cy": height / 2,
            "width": width,
            "height": height,
            "fov": fov,
        },
    }
    with open(target_out_dir / "calibration" / "camera_front.json", "w") as fp:
        json.dump(cam_calib, fp, indent=2)

    timestamps: list[float] = []

    # —— main synchronous loop
    for frame in tqdm(range(config.buffer_frames + config.num_frames), desc="Simulating", ncols=0):
        world_manager.tick()
        if frame < config.buffer_frames:
            get_data_from_queue(lidar_q, frame)
            get_data_from_queue(cam_q, frame)
            continue

        frame -= config.buffer_frames  # calibration frame numbers
        snap = world_manager.snapshot
        timestamps.append(snap.timestamp.elapsed_seconds)

        # 1. ego pose ----------------------------------------------------------------
        np.save(target_out_dir / "ego_pose" / f"{frame:04d}.npy", tf_to_rds(ego.get_transform()))

        # 2. LiDAR -------------------------------------------------------------------
        lidar_meas: carla.LidarMeasurement = get_data_from_queue(lidar_q, frame)
        lidar_meas.save_to_disk(str(target_out_dir / "lidar" / f"{frame:04d}.ply"))

        # 3. RGB image ---------------------------------------------------------------
        cam_meas: carla.Image = get_data_from_queue(cam_q, frame)
        np_img = np.reshape(np.copy(cam_meas.raw_data), (cam_meas.height, cam_meas.width, -1))[:, :, :3]
        # using cv2 here to prevent color conversion from RGB to BGR and also the saving speed is much faster
        cv2.imwrite(str(target_out_dir / "camera_front" / f"{frame:04d}.png"), np_img)

        # 4. dynamic actors ➜ 3‑D boxes --------------------------------------------
        labels = []
        for actor in world_manager.agents["vehicle"] + world_manager.agents["walker"]:
            if not actor.is_alive:
                continue
            if actor.id == ego.id or not actor.bounding_box:
                continue
            tf = actor.get_transform()
            bb = actor.bounding_box  # the location of the actor might not be the center of the bounding box
            class_type = actor.type_id.split(".")[0]
            if class_type not in CLASS_MAP:
                continue
            labels.append(
                {
                    "center": [
                        tf.location.x + bb.location.x,
                        -(tf.location.y + bb.location.y),
                        (tf.location.z + bb.location.z),
                    ],
                    "size": [bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2],
                    "yaw": yaw_to_rds(tf.rotation.yaw),
                    "class": CLASS_MAP[class_type],
                    "track_id": actor.id,
                }
            )
        with open(target_out_dir / "labels_3d" / f"{frame:04d}.json", "w") as fp:
            json.dump(labels, fp)

    # ── store timestamps -----------------------------------------------------------
    with open(target_out_dir / "timestamp.json", "w") as fp:
        json.dump(timestamps, fp)

    logger.info("Saved clip to %s", target_out_dir)


def main(config: StreamConfig):
    # automatically assign clip_id if not provided
    if config.clip_id is None:
        list_of_clips = list(config.out_dir.glob("clip_*"))
        if len(list_of_clips) == 0:
            config.clip_id = 0
        else:
            config.clip_id = max([int(clip.name.split("_", 1)[1]) for clip in list_of_clips]) + 1
    global logger
    target_out_dir = config.out_dir / f"clip_{config.clip_id:04d}"
    target_out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(target_out_dir, config.log_level)

    try:
        with CarlaManager(
            host=config.host,
            port=config.port,
            tm_port=config.tm_port,
            logger=logger,
            fps=config.fps,
        ) as carla_manager:
            record_clip(carla_manager, target_out_dir, config)

        # dump the config file
        with open(target_out_dir / "config.json", "w") as fp:
            json.dump(attrs.asdict(config, value_serializer=serialize), fp, indent=2)

        if config.make_video:
            build_mp4(
                sorted(list((target_out_dir / "camera_front").glob("*.png"))),
                target_out_dir / "camera_front.mp4",
                fps=config.fps,
            )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Closing down...")


if __name__ == "__main__":
    tyro.cli(main)
