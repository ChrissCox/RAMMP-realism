import numpy as np
import pytest

from puppy_sim import runs, scene


@pytest.fixture(scope="session")
def sc():
    return scene.build()


@pytest.fixture()
def kin(sc):
    return sc.kin()


@pytest.fixture(scope="session")
def run():
    return runs.load_run(scene.REFERENCE_RUN)


@pytest.fixture(scope="session")
def functional(sc, run):
    return runs.functional_track(run, sc.kin())


@pytest.fixture(scope="session")
def home():
    return np.array(runs.HOME_JOINTS)
