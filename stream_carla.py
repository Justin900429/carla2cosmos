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
from utils.carla_utils import tf_to_rds, yaw_to_rds, build_mp4

CLASS_MAP = {
    "vehicle": "Vehicle",
    "walker": "Pedestrian",
    # "traffic": "TrafficSign",
}

NOT_ALLOWED_PIERCING = set(["buildings", "walls"])


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


def is_actor_visible(
    world: carla.World,
    actor: carla.Actor,
    camera_location: carla.Location,
    visible_distance_threshold: float = 10.0,
):
    # Add vertical offset to aim at center of object, not ground
    bbox = actor.bounding_box
    actor_transform = actor.get_transform()
    offset = carla.Location(
        x=actor_transform.location.x + bbox.location.x,
        y=actor_transform.location.y + bbox.location.y,
        z=actor_transform.location.z + bbox.location.z,
    )
    target_location = offset

    # Perform ray casting
    raycast_list = world.cast_ray(target_location, camera_location)

    if not raycast_list:
        return False

    if len(raycast_list) > 2:
        for ray_cast in raycast_list[1:-1]:  # skip the first and the last ray cast
            label = str(ray_cast.label).lower()
            if label in NOT_ALLOWED_PIERCING:
                return False

    actor_distance = offset.distance(camera_location)

    return actor_distance < visible_distance_threshold


def lateral_shift(transform: carla.Transform, shift: float) -> carla.Location:
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def check_is_boundary(waypoint: carla.Waypoint, sign: int) -> bool:
    if sign > 0:  # right side
        right_point: carla.Waypoint | None = waypoint.get_right_lane()
        if (right_point is None and waypoint.is_junction) or (
            right_point is not None and right_point.lane_type != carla.LaneType.Driving
        ):
            return True
    elif sign < 0:  # left side
        left_point: carla.Waypoint | None = waypoint.get_left_lane()
        if left_point is not None and left_point.lane_type != carla.LaneType.Driving:
            return True
    return False


def location_to_list(location: carla.Location) -> list[float]:
    """Flip the y-axis to match the coordinate system of the dataset."""
    return [location.x, -location.y, location.z]


def get_single_side_points(
    waypoints: list[carla.Waypoint], sign: int = 1
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    all_lanes_waypoints = []
    all_boundaries_waypoints = []
    temp_waypoints = []
    current_lane_marking = carla.LaneMarkingType.NONE
    for sample in waypoints:
        lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking
        if lane_marking is None:
            continue
        marking_type = lane_marking.type

        if len(temp_waypoints) > 0 and current_lane_marking != marking_type:
            current_lane_marking = marking_type

            is_boundary = check_is_boundary(temp_waypoints[(len(temp_waypoints) - 1) // 2], sign)
            processed_waypoints = [
                location_to_list(lateral_shift(sample.transform, sign * 0.5 * sample.lane_width))
                for sample in temp_waypoints[:-1]
            ]
            if is_boundary:
                all_boundaries_waypoints.append(processed_waypoints)
            else:
                all_lanes_waypoints.append(processed_waypoints)

            temp_waypoints = temp_waypoints[-1:]
        else:
            temp_waypoints.append(sample)

    is_boundary = check_is_boundary(temp_waypoints[(len(temp_waypoints) - 1) // 2], sign)
    processed_waypoints = [
        location_to_list(lateral_shift(sample.transform, sign * 0.5 * sample.lane_width))
        for sample in temp_waypoints
    ]
    if is_boundary:
        all_boundaries_waypoints.append(processed_waypoints)
    else:
        all_lanes_waypoints.append(processed_waypoints)
    return all_lanes_waypoints, all_boundaries_waypoints


def to_left(waypoint: carla.Waypoint) -> carla.Waypoint:
    init_sign = waypoint.lane_id

    use_method = "get_left_lane" if init_sign * waypoint.lane_id > 0 else "get_right_lane"
    while getattr(waypoint, use_method)() is not None:
        next_waypoint = getattr(waypoint, use_method)()
        if next_waypoint is None:
            break
        if next_waypoint.lane_type != carla.LaneType.Driving:
            break
        waypoint = next_waypoint
        use_method = "get_left_lane" if init_sign * waypoint.lane_id > 0 else "get_right_lane"
    return waypoint


def to_right(waypoint: carla.Waypoint) -> carla.Waypoint:
    init_sign = waypoint.lane_id
    use_method = "get_right_lane" if init_sign * waypoint.lane_id > 0 else "get_left_lane"
    while getattr(waypoint, use_method)() is not None:
        next_waypoint = getattr(waypoint, use_method)()
        if next_waypoint is None:
            break
        if next_waypoint.lane_type != carla.LaneType.Driving:
            break
        waypoint = next_waypoint
        use_method = "get_right_lane" if init_sign * waypoint.lane_id > 0 else "get_left_lane"
    return waypoint


def save_map_info(carla_manager: CarlaManager, target_out_dir: pathlib.Path, precision: float = 1.0):
    topology = carla_manager.world_manager.map.get_topology()
    topology = sorted([x[0] for x in topology], key=lambda x: x.transform.location.z)
    set_waypoints: list[list[carla.Waypoint]] = []
    for waypoint in topology:
        waypoints = [waypoint]

        next_waypoint = waypoint.next(precision)
        if len(next_waypoint) > 0:
            next_waypoint = next_waypoint[0]
            while next_waypoint.road_id == waypoint.road_id:
                waypoints.append(next_waypoint)
                next_waypoint = next_waypoint.next(precision)
                if len(next_waypoint) > 0:
                    next_waypoint = next_waypoint[0]
                else:
                    break
        set_waypoints.append(waypoints)

    boundaries = {"labels": []}
    lanelines = {"labels": []}
    for waypoints in set_waypoints:
        left_lanes_waypoints_list, left_boundaries_waypoints_list = get_single_side_points(waypoints, sign=-1)
        for left_waypoints in left_lanes_waypoints_list:
            lanelines["labels"].append(
                {
                    "labelData": {
                        "shape3d": {"polyline3d": {"vertices": left_waypoints}},
                    }
                }
            )

        for left_waypoints in left_boundaries_waypoints_list:
            boundaries["labels"].append(
                {
                    "labelData": {
                        "shape3d": {"polyline3d": {"vertices": left_waypoints}},
                    }
                }
            )
        right_lanes_waypoints_list, right_boundaries_waypoints_list = get_single_side_points(
            waypoints, sign=1
        )
        for right_waypoints in right_lanes_waypoints_list:
            lanelines["labels"].append(
                {
                    "labelData": {
                        "shape3d": {"polyline3d": {"vertices": right_waypoints}},
                    }
                }
            )
        for right_waypoints in right_boundaries_waypoints_list:
            boundaries["labels"].append(
                {
                    "labelData": {
                        "shape3d": {"polyline3d": {"vertices": right_waypoints}},
                    }
                }
            )

    with open(target_out_dir / "hdmap" / "road_boundaries.json", "w") as fp:
        json.dump(boundaries, fp)
    with open(target_out_dir / "hdmap" / "lanelines.json", "w") as fp:
        json.dump(lanelines, fp)

    # ── save poles
    poles = {"labels": []}  # polylines
    pole_bbs = carla_manager.world_manager.world.get_level_bbs(carla.CityObjectLabel.Poles)
    for pole_bb in pole_bbs:
        center = pole_bb.location
        extent = pole_bb.extent
        verts = [
            [center.x, -center.y, center.z - extent.z],
            [center.x, -center.y, center.z + extent.z],
        ]
        poles["labels"].append(
            {
                "labelData": {
                    "shape3d": {"polyline3d": {"vertices": verts}},
                }
            }
        )
    with open(target_out_dir / "hdmap" / "poles.json", "w") as fp:
        json.dump(poles, fp)

    # ── save wait lines
    wait_lines = {"labels": []}
    for traffic_light in carla_manager.world_manager.actors.filter("traffic.traffic_light"):
        wait_points = traffic_light.get_stop_waypoints()
        if len(wait_points) == 0:
            continue
        wait_point = wait_points[0]

        right_point, left_point = wait_point, wait_point
        while right_point.get_right_lane() is not None:
            right_right_point = right_point.get_right_lane()
            if right_right_point.lane_type != carla.LaneType.Driving:
                break
            right_point = right_right_point
        while left_point.get_left_lane() is not None:
            left_left_point = left_point.get_left_lane()
            if (
                left_left_point.lane_type != carla.LaneType.Driving
                or left_left_point.lane_id * wait_point.lane_id < 0
            ):
                break
            left_point = left_left_point
        right_point = lateral_shift(right_point.transform, 0.5 * right_point.lane_width)
        left_point = lateral_shift(left_point.transform, -0.5 * left_point.lane_width)

        wait_lines["labels"].append(
            {
                "labelData": {
                    "shape3d": {
                        "polyline3d": {
                            "vertices": [
                                location_to_list(left_point),
                                location_to_list(right_point),
                            ]
                        }
                    },
                }
            }
        )
    with open(target_out_dir / "hdmap" / "wait_lines.json", "w") as fp:
        json.dump(wait_lines, fp)

    # ── save traffic lights
    traffic_lights = {"labels": []}  # cuboids
    traffic_light_bbs = carla_manager.world_manager.world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    for traffic_light_bb in traffic_light_bbs:
        center = traffic_light_bb.location
        extent = traffic_light_bb.extent
        verts = [
            [center.x - extent.x, -(center.y - extent.y), center.z + extent.z],
            [center.x + extent.x, -(center.y - extent.y), center.z + extent.z],
            [center.x + extent.x, -(center.y + extent.y), center.z + extent.z],
            [center.x - extent.x, -(center.y + extent.y), center.z + extent.z],
            [center.x - extent.x, -(center.y - extent.y), center.z - extent.z],
            [center.x + extent.x, -(center.y - extent.y), center.z - extent.z],
            [center.x + extent.x, -(center.y + extent.y), center.z - extent.z],
            [center.x - extent.x, -(center.y + extent.y), center.z - extent.z],
        ]
        traffic_lights["labels"].append(
            {
                "labelData": {
                    "shape3d": {"boxes": {"vertices": verts}},
                }
            }
        )
    with open(target_out_dir / "hdmap" / "traffic_lights.json", "w") as fp:
        json.dump(traffic_lights, fp)

    # ── save traffic signs
    traffic_signs = {"labels": []}  # cuboids
    traffic_sign_bbs = carla_manager.world_manager.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
    for traffic_sign_bb in traffic_sign_bbs:
        center = traffic_sign_bb.location
        extent = traffic_sign_bb.extent
        verts = [
            [center.x - extent.x, -(center.y - extent.y), center.z + extent.z],
            [center.x + extent.x, -(center.y - extent.y), center.z + extent.z],
            [center.x + extent.x, -(center.y + extent.y), center.z + extent.z],
            [center.x - extent.x, -(center.y + extent.y), center.z + extent.z],
            [center.x - extent.x, -(center.y - extent.y), center.z - extent.z],
            [center.x + extent.x, -(center.y - extent.y), center.z - extent.z],
            [center.x + extent.x, -(center.y + extent.y), center.z - extent.z],
            [center.x - extent.x, -(center.y + extent.y), center.z - extent.z],
        ]
        traffic_signs["labels"].append(
            {
                "labelData": {
                    "shape3d": {"boxes": {"vertices": verts}},
                }
            }
        )
    with open(target_out_dir / "hdmap" / "traffic_signs.json", "w") as fp:
        json.dump(traffic_signs, fp)

    # crosswalks will be processed with opendrive due to the issue of
    #  https://github.com/carla-simulator/carla/issues/5790


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
    cam_tf = carla.Transform(carla.Location(x=1, z=2))  # bonnet-edge mount
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
    save_map_info(carla_manager, target_out_dir)

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
        global_camera_location = ego.get_transform().location + cam_tf.location
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
        for actor in world_manager.actors:
            if not actor.is_alive:
                continue
            if actor.type_id.split(".")[0] not in CLASS_MAP:
                continue
            if actor.id == ego.id or not actor.bounding_box:
                continue
            if not is_actor_visible(
                world_manager.world,
                actor,
                global_camera_location,
                visible_distance_threshold=config.visible_distance_threshold,
            ):
                continue
            tf = actor.get_transform()
            bb = actor.bounding_box

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
