import random
from collections import defaultdict
import carla
import logging

from manager.agent_model_manager import AgentModelManager


DISTANCE_TO_OTHERS = 10


def proximity_to_ea(location: carla.Location, existing_agents: list[carla.Location]) -> bool:
    for ea_loc in existing_agents:
        if ea_loc.distance(location) < DISTANCE_TO_OTHERS:
            return True
    return False


class WorldManager:
    def __init__(
        self,
        world: carla.World | None = None,
        tm_port: int = 8000,
        logger: logging.Logger | None = None,
    ):
        self.world = world
        self.agents: dict[str, list[carla.Actor]] = defaultdict(list)
        self.tm_port = tm_port
        self.logger = logger
        self.agent_model_manager = AgentModelManager(blueprint=self.world.get_blueprint_library())

    @property
    def agents(self) -> dict[str, list[carla.Actor]]:
        return self._agents

    @agents.setter
    def agents(self, agents: dict[str, list[carla.Actor]]):
        self._agents = agents

    @property
    def actors(self) -> list[carla.Actor]:
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

    def set_attribute(self, attribute: str, value: any):
        getattr(self.world, attribute)(value)

    def on_tick(self, callable: callable):
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

    def random_spawn_car(
        self,
        blueprint: str | carla.ActorBlueprint | None = None,
        autopilot: bool = True,
        existing_agents: list[carla.Location] | None = None,
    ) -> carla.Actor:
        if blueprint is not None:
            bp = self.find_blueprint(blueprint)
        else:
            bp = self.agent_model_manager.categories["car"]
            bp = random.choice(bp)

        spawn_points = self.map.get_spawn_points()

        spawn_point = random.choice(spawn_points)
        if existing_agents is not None:
            is_proximity_to_ea = proximity_to_ea(spawn_point.location, existing_agents)
            while is_proximity_to_ea:
                spawn_point = random.choice(spawn_points)
                if spawn_point is not None:
                    is_proximity_to_ea = proximity_to_ea(spawn_point.location, existing_agents)

        agent = self.spawn_actor(bp, spawn_point)
        if agent is not None:
            if autopilot:
                agent.set_autopilot(True, self.tm_port)
            if existing_agents is not None:
                existing_agents.append(agent.get_transform().location)
        return agent

    def random_spawn_cars_with_nums(
        self,
        blueprint: str | carla.ActorBlueprint | None = None,
        autopilot: bool = True,
        existing_agents: list[carla.Location] | None = None,
        spawn_nums: int = 1,
    ) -> list[carla.Actor]:
        if blueprint is not None:
            bp = self.find_blueprint(blueprint)
            bp_list = [bp]
        else:
            bp_list = self.agent_model_manager.categories["car"]

        actors = []
        world_spawn_points = self.map.get_spawn_points()
        while len(actors) < spawn_nums:
            spawn_points = []
            for _ in range(spawn_nums - len(actors)):
                spawn_point = random.choice(world_spawn_points)
                if existing_agents is not None:
                    is_proximity_to_ea = proximity_to_ea(spawn_point.location, existing_agents)
                    while is_proximity_to_ea:
                        spawn_point = random.choice(world_spawn_points)
                        if spawn_point is not None:
                            is_proximity_to_ea = proximity_to_ea(spawn_point.location, existing_agents)
                spawn_points.append(spawn_point)
                if existing_agents is not None:
                    existing_agents.append(spawn_point.location)

            for spawn_point in spawn_points:
                bp = random.choice(bp_list)
                actor = self.spawn_actor(bp, spawn_point)
                if actor is not None:
                    if autopilot:
                        actor.set_autopilot(True, self.tm_port)
                    actors.append(actor)
        return actors

    def random_spawn_walker(
        self,
        blueprint: str | carla.ActorBlueprint | None = None,
        autopilot: bool = True,
        existing_agents: list[carla.Location] | None = None,
    ) -> carla.Actor:
        spawn_point = self.world.get_random_location_from_navigation()
        if existing_agents is not None:
            is_proximity_to_ea = proximity_to_ea(spawn_point, existing_agents)
            while is_proximity_to_ea:
                spawn_point = self.world.get_random_location_from_navigation()
                if spawn_point is not None:
                    is_proximity_to_ea = proximity_to_ea(spawn_point, existing_agents)

        spawn_point = carla.Transform(location=spawn_point + carla.Location(z=2))

        if blueprint is not None:
            bp = self.find_blueprint(blueprint)
        else:
            bp = self.agent_model_manager.categories["pedestrian"]
            bp = random.choice(bp)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        agent = self.spawn_actor(bp, spawn_point)
        if agent is not None:
            agent.set_collisions(True)
            agent.set_simulate_physics(True)
            if autopilot:
                walker_controller_bp = self.find_blueprint("controller.ai.walker")
                walker_controller = self.spawn_actor(
                    walker_controller_bp, agent.get_transform(), attach_to=agent
                )
                if walker_controller is not None:
                    walker_controller.start()
                    walker_controller.go_to_location(self.world.get_random_location_from_navigation())
                    walker_controller.set_max_speed(1 + random.random())
            if existing_agents is not None:
                existing_agents.append(agent.get_transform().location)
        return agent

    def random_spawn_walkers_with_nums(
        self,
        blueprint: str | carla.ActorBlueprint | None = None,
        existing_agents: list[carla.Location] | None = None,
        spawn_nums: int = 1,
    ) -> list[carla.Actor]:
        if blueprint is not None:
            bp = self.find_blueprint(blueprint)
            bp_list = [bp]
        else:
            bp_list = self.agent_model_manager.categories["pedestrian"]
        actors = []
        while len(actors) < spawn_nums:
            spawn_points = []
            for _ in range(spawn_nums - len(actors)):
                spawn_point = self.world.get_random_location_from_navigation()
                if existing_agents is not None:
                    is_proximity_to_ea = proximity_to_ea(spawn_point, existing_agents)
                    while is_proximity_to_ea:
                        spawn_point = self.world.get_random_location_from_navigation()
                        if spawn_point is not None:
                            is_proximity_to_ea = proximity_to_ea(spawn_point, existing_agents)
                spawn_points.append(carla.Transform(location=spawn_point + carla.Location(z=1)))
                if existing_agents is not None:
                    existing_agents.append(spawn_point)
            for spawn_point in spawn_points:
                walker_bp = random.choice(bp_list)
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                actor = self.spawn_actor(walker_bp, spawn_point)
                if actor is not None:
                    actor.set_collisions(True)
                    actor.set_simulate_physics(True)
                    actors.append(actor)
        return actors

    def set_ai_walkers(self, walkers: list[carla.Actor]):
        for walker in walkers:
            walker_controller_bp = self.find_blueprint("controller.ai.walker")
            walker_controller = self.spawn_actor(
                walker_controller_bp, walker.get_transform(), attach_to=walker
            )
            if walker_controller is not None:
                walker_controller.start()
                walker_controller.go_to_location(self.world.get_random_location_from_navigation())
                walker_controller.set_max_speed(1 + random.random())
