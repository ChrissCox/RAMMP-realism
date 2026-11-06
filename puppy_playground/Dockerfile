# syntax=docker/dockerfile:1

# arm64 only (rammp-base is Jetson-only). Built from the REPO ROOT:
#   docker build -f puppy_playground/Dockerfile -t puppy-playground .
# Same base and build pattern as rammp-kitchen-capability-engines' capability images
# (capabilities/example, capabilities/grab_container): pinned nightly base, pybullet from source,
# rammp_curobo_interfaces from a pinned RAMMP-CuRobo commit, colcon overlay at /module_ws.
# nightly-20261030 carries the arm, common and orchestrator interfaces (73e7ed1716bf).
FROM ghcr.io/rammp-org/rammp-base:nightly-20261030

WORKDIR /module_ws

# PyPI ships no arm64 pybullet wheel: compiled from source; 3.2.7 matches the SheppyArm goldens.
RUN pip install --no-cache-dir pybullet==3.2.7

ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/fd8b89cb18fd854f3add36ab9b1b72933dbabf76.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz

# capability_engine from rammp-kitchen-capability-engines, at the commit the CI image tests against
# (docker/ros-test.Dockerfile RAMMP_REF).
ARG RAMMP_REF=c3989c6
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone -q --filter=blob:none https://github.com/rammp-org/rammp-kitchen-capability-engines /tmp/rammp \
    && git -C /tmp/rammp checkout -q ${RAMMP_REF} \
    && cp -r /tmp/rammp/capability_engine src/capability_engine \
    && rm -rf /tmp/rammp /var/lib/apt/lists/*

# puppy_motion is a plain Python library (numpy and scipy come from the base / its pip deps).
# The base's pip (Ubuntu 22.04, 22.0.2) builds a pyproject-only package as an empty UNKNOWN-0.0.0
# wheel, so upgrade pip first. numpy/scipy are pinned to the versions CI tests with (the base has
# numpy 1.21.5, below puppy_motion's floor; unpinned, pip would pull numpy 2, which the base's
# compiled packages do not support). /usr/local comes before the apt dist-packages on sys.path.
COPY puppy_motion/ /tmp/puppy_motion/
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir numpy==1.26.4 scipy==1.14.0 /tmp/puppy_motion \
    && rm -rf /tmp/puppy_motion

COPY puppy_playground/ src/puppy_playground/

# rosdep installs capability_engine's apt dependencies (kortex_description, robotiq_description, scipy)
# and puppy_playground's; those are not known to be in the base, so this is the same step the
# example/grab_container Dockerfiles run. Nothing here is a pip dependency: pybullet and puppy_motion
# (with its numpy/scipy/pyyaml requirements) are installed explicitly above.
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf build log

# Fail the build, not the robot run, if anything the nodes import is missing or on the wrong numpy.
RUN bash -c '. /opt/ros/humble/setup.sh && . /ros2_ws/install/setup.sh && . /module_ws/install/setup.sh \
    && python3 -W ignore -c "import numpy, scipy, pybullet, rclpy, capability_engine, puppy_motion.validate, \
puppy_playground.engine, puppy_playground.ui_node, puppy_playground.pipeline; \
assert numpy.__version__ == \"1.26.4\", numpy.__version__; print(\"imports ok\", numpy.__version__, scipy.__version__)"'

# The entrypoint sources /opt/ros/humble, the base's /ros2_ws overlay and this /module_ws overlay,
# then execs the command (so `docker run IMAGE python3 ...` works too).
COPY puppy_playground/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# /data holds stage_path.json, stage_path.candidate.json and playground.jsonl (mount a host dir).
VOLUME /data
EXPOSE 8080

# `ros2 launch` on Humble has no SIGTERM handler; SIGINT stops its nodes cleanly.
STOPSIGNAL SIGINT
ENTRYPOINT ["/entrypoint.sh"]
# Launch arguments go after the launch file: docker run ... IMAGE ros2 launch puppy_playground
# playground.launch.py mock:=true   (BRINGUP.md has the exact commands)
CMD ["ros2", "launch", "puppy_playground", "playground.launch.py", "use_orchestrator:=false"]
