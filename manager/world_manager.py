import random
from typing import Any, Callable, Literal, Optional
from collections import defaultdict
import carla
import logging

from manager.agent_model_manager import AgentModelManager


class WorldManager:
    def __init__(
        self,
        world: Optional[carla.World] = None,
        tm_port: int = 8000,
        logger: logging.Logger | None = None,
    ):
        self._world = world
        self.agents: dict[str, list[carla.Actor]] = defaultdict(list)
        self.tm_port = tm_port
        self.logger = logger
        self.agent_model_manager = AgentModelManager(blueprint=self.world.get_blueprint_library())

    @property
    def actors(self) -> carla.ActorList:
        return self.world.get_actors()

    @property
    def map(self):
        return self.world.get_map()

    @property
    def snapshot(self):
        return self.world.get_snapshot()

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, world: carla.World):
        self._world = world

    def set_attribute(self, attribute: str, value: Any):
        getattr(self.world, attribute)(value)

    def on_tick(self, callable: Callable):
        self.world.on_tick(callable)

    def tick(self, seconds: float = 10):
        self.world.tick(seconds)

    def destroy_all_actors(self):
        # reverse order to destroy sensors before destroying attached actors and
        #  stop controllers before destroying attached walkers
        for agent in self.agents[::-1]:
            type_id = agent.type_id
            self.logger.debug(f"destroying actor {type_id}")
            if agent.is_alive and type_id.startswith("controller"):
                agent.stop()
            elif agent.is_alive:
                agent.destroy()

    def find_blueprint(self, blueprint: str | carla.ActorBlueprint) -> carla.ActorBlueprint:
        if isinstance(blueprint, carla.ActorBlueprint):
            return blueprint
        return self.agent_model_manager.blueprint.find(blueprint)

    def spawn_actor(
        self,
        blueprint: carla.ActorBlueprint,
        transform: carla.Transform,
        attach_to: carla.Actor | None = None,
    ) -> carla.Actor:
        actor = self.world.try_spawn_actor(blueprint, transform, attach_to)
        if actor is not None:
            self.logger.debug(f"spawned actor {actor.type_id}")
            type_id_prefix = actor.type_id.split(".")[0]
            self.agents[type_id_prefix].append(actor)
        return actor

    def random_spawn(
        self,
        blueprint: str | None = None,
        category: Literal["car", "pedestrian"] = "car",
        autopilot: bool = True,
    ) -> carla.Actor:
        spawn_points = self.map.get_spawn_points()
        ego_tf = random.choice(spawn_points)
        if blueprint is not None:
            ego_bp = self.find_blueprint(blueprint)
        else:
            ego_bp_list = self.agent_model_manager.categories[category]
            ego_bp = random.choice(ego_bp_list)
        agent = self.spawn_actor(ego_bp, ego_tf)
        if agent is not None:
            if autopilot and agent.type_id.startswith("vehicle"):
                agent.set_autopilot(True, self.tm_port)
            elif autopilot and agent.type_id.startswith("walker"):
                walker_controller_bp = self.find_blueprint("controller.ai.walker")
                walker_controller = self.spawn_actor(
                    walker_controller_bp, agent.get_transform(), attach_to=agent
                )
                if walker_controller is not None:
                    walker_controller.start()
                    walker_controller.go_to_location(self.world.get_random_location_from_navigation())
                    walker_controller.set_max_speed(1 + random.random())
                    self.logger.debug(f"spawned walker controller {walker_controller.type_id}")
        return agent

    def random_spawn_with_nums(
        self,
        blueprint: str | None = None,
        category: Literal["car", "pedestrian"] = "car",
        autopilot: bool = True,
        spawn_nums: int = 1,
    ) -> list[carla.Actor]:
        if blueprint is not None:
            bp = self.find_blueprint(blueprint)
            bp_list = [bp]
        else:
            bp_list = self.agent_model_manager.categories[category]
        actors = []

        while len(actors) < spawn_nums:
            bp = random.choice(bp_list)
            actor = self.random_spawn(blueprint=bp, autopilot=autopilot)
            if actor is not None:
                actors.append(actor)
        return actors
