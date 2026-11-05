import numpy as np
import pytest

import runs_helper


@pytest.fixture(scope="session")
def functional():
    return runs_helper.functional_track(runs_helper.RUN_NAMES[0])


@pytest.fixture(scope="session")
def home():
    return np.array(runs_helper.HOME_JOINTS)


@pytest.fixture(scope="session")
def chain():
    import json
    import os

    from puppy_motion.urdf import Chain

    with open(os.path.join(runs_helper.DATA, runs_helper.RUN_NAMES[0] + ".json")) as f:
        return Chain.from_urdf(json.load(f)["urdf"])


@pytest.fixture(scope="session")
def scene():
    import os

    from puppy_motion.geometry import Scene

    with open(os.path.join(runs_helper.DATA, "spike_person.json")) as f:
        return Scene.from_param(f.read())
