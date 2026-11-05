import numpy as np
import pytest

import runs_helper


@pytest.fixture(scope="session")
def functional():
    return runs_helper.functional_track(runs_helper.RUN_NAMES[0])


@pytest.fixture(scope="session")
def home():
    return np.array(runs_helper.HOME_JOINTS)
