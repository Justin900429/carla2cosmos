import logging
import carla

from manager.world_manager import WorldManager


class CarlaManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        logger: logging.Logger | None = None,
        fps: int = 30,
    ):
        self.logger = logger
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.fixed_delta_seconds = 1 / fps
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.world_manager = WorldManager(
            world=self.world,
            logger=logger,
            tm_port=tm_port,
        )
        self.original_settings = None

    def set_world(self):
        self.world = self.client.get_world()
        self.world_manager.world = self.world

    def load_town(self, town: str):
        if self.world is not None:
            self.clean()
        self.client.load_world(town)
        self.set_world()
        self.set_carla_sync_mode(True)
        self.world_manager.tick()

    def set_carla_sync_mode(self, sync: bool, fixed_delta_seconds: float = 1 / 30):
        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = fixed_delta_seconds if sync else None
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(sync)

    def clean(self):
        self.set_carla_sync_mode(False)

        if "sensor" in self.world_manager.agents:
            self.logger.debug("Destroying sensors...")
            for sensor in self.world_manager.agents["sensor"]:
                sensor.destroy()

        if "controller" in self.world_manager.agents:
            self.logger.debug("Stopping controllers...")
            for controller in self.world_manager.agents["controller"]:
                controller.stop()
                controller.destroy()

        if "vehicle" in self.world_manager.agents:
            self.logger.debug("Destroying vehicles...")
            self.client.apply_batch(
                [carla.command.DestroyActor(vehicle.id) for vehicle in self.world_manager.agents["vehicle"]]
            )

        if "walker" in self.world_manager.agents:
            self.logger.debug("Destroying walkers...")
            self.client.apply_batch(
                [carla.command.DestroyActor(walker.id) for walker in self.world_manager.agents["walker"]]
            )

    def __enter__(self):
        self.set_world()
        self.set_carla_sync_mode(True)
        return self

    def __exit__(self, *_):
        self.clean()
