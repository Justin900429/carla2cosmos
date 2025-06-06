import pathlib
from typing import Literal
from attrs import define, field


@define
class LidarConfig:
    range: float = field(default=120.0)
    rotation_frequency: float = field(default=30.0)
    points_per_second: int = field(default=2200000)
    channels: int = field(default=64)
    upper_fov: float = field(default=10.0)
    lower_fov: float = field(default=-30.0)


@define
class CameraConfig:
    image_size_x: int = field(default=1920)
    image_size_y: int = field(default=1080)
    fov: float = field(default=90.0)
    sensor_tick: float = field(default=0.033333)


@define
class EgoConfig:
    vehicle_model: str = field(default="vehicle.lincoln.mkz_2017")
    autopilot: bool = field(default=True)


@define
class StreamConfig:
    # —— output
    out_dir: pathlib.Path = field(default=pathlib.Path("./data"))
    clip_id: int = field(default=0)
    make_video: bool = field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = field(default="INFO")

    # —— world
    host: str = field(default="localhost")
    port: int = field(default=2000)
    tm_port: int = field(default=8000)
    town: str = field(default="Town01")
    fps: int = field(default=30)
    num_frames: int = field(default=600)
    num_vehicles: int = field(default=20)
    num_walkers: int = field(default=30)

    # —— sensors
    lidar: LidarConfig = field(default=LidarConfig())
    camera: CameraConfig = field(default=CameraConfig())
    ego: EgoConfig = field(default=EgoConfig())
