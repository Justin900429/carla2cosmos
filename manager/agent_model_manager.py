import random

import carla


class AgentModelManager:
    def __init__(self, blueprint: carla.ActorBlueprint | None = None):
        self.blueprint = blueprint
        self.categories = {
            "ambulance": [],
            "police": [],
            "firetruck": [],
            "bus": [],
            "truck": [],
            "motorcycle": [],
            "car": [],
            "bicycle": [],
            "pedestrian": [],
        }
        if self.blueprint is not None:
            self.classify_blueprint()

    def classify_blueprint(self):
        for actor in self.blueprint:
            action_name = actor.id

            if action_name.startswith("vehicle"):
                if actor.has_attribute("base_type"):
                    base_type = actor.get_attribute("base_type").as_str()
                else:
                    base_type = None
                if "ambulance" in action_name:
                    self.categories["ambulance"].append(actor)
                elif "police" in action_name:
                    self.categories["police"].append(actor)
                elif "firetruck" in action_name:
                    self.categories["firetruck"].append(actor)
                elif base_type is not None and base_type == "Bus":
                    self.categories["bus"].append(actor)
                elif base_type is not None and base_type == "truck":
                    self.categories["truck"].append(actor)
                elif base_type is not None and base_type == "motorcycle":
                    self.categories["motorcycle"].append(actor)
                elif base_type is not None and base_type == "bicycle":
                    self.categories["bicycle"].append(actor)
                else:
                    self.categories["car"].append(actor)
            elif action_name.startswith("walker"):
                self.categories["pedestrian"].append(actor)

    def get_blueprint_from_type(self, type_name: str) -> carla.ActorBlueprint:
        type_name = type_name if type_name in self.categories else "car"
        return random.choice(self.categories[type_name])

    def get_blueprint_from_name(self, model_name: str) -> carla.ActorBlueprint:
        return self.blueprint.find(model_name)

    def set_blueprint(self, blueprint: carla.BlueprintLibrary):
        self.blueprint = blueprint
        for key in self.categories.keys():
            self.categories[key].clear()
        self.classify_blueprint()

    def __str__(self) -> str:
        category_info = []
        for key, value in self.categories.items():
            category_info.append(f"{key}={len(value)}")
        return f"AgentModelManager({', '.join(category_info)})"
