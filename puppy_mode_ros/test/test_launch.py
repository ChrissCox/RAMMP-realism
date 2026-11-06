"""The launcher imports <module>:<Class> and <module>:<main>, installs the hook on the class and
calls main with the remaining argv. A dummy module stands in for an engine (no ROS needed)."""

import sys
import textwrap

import pytest

from puppy_mode_ros import hook, launch

DUMMY = textwrap.dedent(
    """
    import sys

    seen = {}

    class Engine:
        def __init__(self):
            self.params = {}

        def has_parameter(self, name):
            return name in self.params

        def declare_parameter(self, name, value):
            self.params[name] = value

        def get_parameter(self, name):
            return type("P", (), {"value": self.params[name]})

        def execute(self, action_id, params, ctx):
            return {"ran": action_id}

    def main():
        seen["argv"] = list(sys.argv)
        seen["installed"] = getattr(Engine.execute, "_puppy_wrapped", False)
        e = Engine()
        seen["params"] = dict(e.params)
        seen["result"] = e.execute("go", {}, None)
        return seen.get("rc", 0)
    """
)


@pytest.fixture
def dummy(tmp_path, monkeypatch):
    (tmp_path / "dummy_engine_mod.py").write_text(DUMMY)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "argv", sys.argv[:])
    yield
    sys.modules.pop("dummy_engine_mod", None)


def test_launch_installs_then_runs_main_with_remaining_argv(dummy):
    rc = launch.main(["dummy_engine_mod:Engine", "dummy_engine_mod:main", "--ros-args", "-p", "x:=1"])
    assert rc == 0
    seen = sys.modules["dummy_engine_mod"].seen
    assert seen["installed"] is True
    assert seen["argv"] == ["dummy_engine_mod:main", "--ros-args", "-p", "x:=1"]
    assert seen["params"] == hook.PARAM_DEFAULTS
    assert seen["result"] == {"ran": "go"}  # puppy.enabled false: the original execute


def test_launch_exit_code_follows_the_engine_entry_point(dummy):
    import importlib

    mod = importlib.import_module("dummy_engine_mod")
    mod.seen["rc"] = None
    assert launch.main(["dummy_engine_mod:Engine", "dummy_engine_mod:main"]) == 0
    mod.seen["rc"] = {"not": "an int"}
    assert launch.main(["dummy_engine_mod:Engine", "dummy_engine_mod:main"]) == 0
    mod.seen["rc"] = 3
    assert launch.main(["dummy_engine_mod:Engine", "dummy_engine_mod:main"]) == 3


def test_launch_rejects_bad_specs(dummy):
    with pytest.raises(SystemExit):
        launch.main(["dummy_engine_mod", "dummy_engine_mod:main"])
    with pytest.raises(SystemExit):
        launch.main(["dummy_engine_mod:Engine"])
