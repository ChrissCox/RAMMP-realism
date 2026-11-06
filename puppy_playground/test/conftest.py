"""CI tests the installed puppy_motion package, not the repo source, so a packaging break shows up.

With the repo mounted at the working directory, its top-level puppy_motion/ folder is only a
namespace-package candidate, and the installed regular package wins over it. When
PLAYGROUND_REQUIRE_INSTALLED_PUPPY_MOTION is set (the ros_tests workflow does), the session fails
unless puppy_motion really came from an installed distribution.
"""

import os
import sys


def pytest_sessionstart(session):
    if not os.environ.get("PLAYGROUND_REQUIRE_INSTALLED_PUPPY_MOTION"):
        return
    import puppy_motion

    where = os.path.realpath(puppy_motion.__file__)
    if "site-packages" not in where and "dist-packages" not in where:
        sys.exit(f"puppy_motion was imported from the source tree ({where}), not the installed package")
