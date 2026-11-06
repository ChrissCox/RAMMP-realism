"""Puppy Playground engine. Scaffold: every action fails with "not_yet" until its logic lands."""

from capability_engine.engine import Action, CapabilityEngine, EngineError, run

STAGE_PATH = "/data/stage_path.json"
CANDIDATE_PATH = "/data/stage_path.candidate.json"
LOG_PATH = "/data/playground.jsonl"

_SIZE = {"type": "number", "minimum": 0.25, "maximum": 1.0, "default": 0.25}


def _schema(**properties):
    return {"type": "object", "properties": properties}


class PuppyPlayground(CapabilityEngine):
    display_name = "Puppy Playground"
    required_resources = ["arm"]
    actions = [
        Action("plan_stage", "Plan the stage path (no motion)", params_schema=_schema()),
        Action("stage", "Move from rest to the stage pose", params_schema=_schema()),
        Action("rest", "Move from the stage pose back to rest", params_schema=_schema()),
        Action("recover", "Return to rest from near the stage path", params_schema=_schema()),
        Action(
            "express",
            "Play one expressive behavior",
            params_schema=_schema(
                behavior={"type": "string"},
                size=_SIZE,
                intensity={"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
            ),
        ),
        Action(
            "combo",
            "Play a sequence of behaviors",
            params_schema=_schema(
                behaviors={"type": "array", "items": {"type": "string"}}, size=_SIZE
            ),
        ),
        Action(
            "auto",
            "Play behaviors on its own whenever the arm is idle",
            params_schema=_schema(size=_SIZE, duration_s={"type": "number", "minimum": 0}),
        ),
    ]

    def __init__(self, engine_id="puppy_playground", **node_kwargs):
        super().__init__(engine_id, **node_kwargs)
        self.stage_path = self.declare_parameter("playground.stage_path", STAGE_PATH).value
        self.candidate_path = self.declare_parameter(
            "playground.candidate_path", CANDIDATE_PATH
        ).value
        self.log_path = self.declare_parameter("playground.log", LOG_PATH).value

    def execute(self, action_id, params, ctx):
        raise EngineError("not_yet", f"{action_id} is not implemented yet")


def main():
    run(PuppyPlayground)


if __name__ == "__main__":
    main()
