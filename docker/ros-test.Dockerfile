# x86 ROS 2 Humble test image for puppy_playground.
# Base: rammp-org/rammp-kitchen-capability-engines docker/test.Dockerfile at c3989c6, copied
# verbatim below (lines up to the PYTHONPATH block). Added at the end: capability_engine from that
# same commit (+ its test helper fake_driver.py) and puppy_motion from this repo, copied in at
# build time, so the build context is the repo root:
#   docker build -f docker/ros-test.Dockerfile -t puppy-ros-test .
#   docker run --rm -v "$PWD":/repo puppy-ros-test pytest puppy_playground/test -q
# puppy_playground itself is not installed: the mounted /repo/puppy_playground is on PYTHONPATH.
FROM ros:humble-ros-base

# Pins. Interfaces and curobo are full SHAs; kinova_gen3_description is dev.
ARG INTERFACES_REF=73e7ed1716bf8a2df0c4f1614d8c360bde20d009
ARG CUROBO_REF=fd8b89cb18fd854f3add36ab9b1b72933dbabf76
ARG KINOVA_REF=2546edbab9cda031cdfe82b33289f05e01341f88
ARG DEPLOYMENTS_REF=9e0cd960fd7e1876a8a592b8fdb33a27da82a734
# feeding-deployment microwave-wip: the students' Cornell arm model.
ARG CORNELL_REF=7bc328aef5433b1617b61b30758d7d626b90af8b

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git python3-dev python3-pip python3-colcon-common-extensions \
        ros-humble-kortex-description ros-humble-robotiq-description ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

# numpy, scipy and pybullet are pinned to the versions the SheppyArm goldens were recorded with
# (pybullet 3.2.7 from PyPI reproduces the recorded pybullet-arm64 3.2.8 output exactly).
# hadolint ignore=DL3013
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.14.0 pybullet==3.2.7 pytest

# Fetch commit $2 of GitHub repo $1 into dir $3 (full SHA required for a shallow fetch).
# hadolint ignore=SC2016
RUN printf '%s\n' '#!/bin/sh' 'set -e' \
        'git init -q "$3" && cd "$3"' \
        'git fetch -q --depth 1 "https://github.com/$1.git" "$2" && git checkout -q FETCH_HEAD' \
        > /usr/local/bin/fetch-commit \
    && chmod +x /usr/local/bin/fetch-commit

WORKDIR /ws
RUN mkdir src \
    && fetch-commit rammp-org/rammp-interfaces-ros2 ${INTERFACES_REF} /tmp/interfaces \
    && cp -r /tmp/interfaces/rammp_arm_interfaces /tmp/interfaces/rammp_common_interfaces \
        /tmp/interfaces/rammp_orchestrator_interfaces src/ \
    && fetch-commit rammp-org/RAMMP-CuRobo ${CUROBO_REF} /tmp/curobo \
    && cp -r /tmp/curobo/rammp_curobo_interfaces src/ \
    && fetch-commit rammp-org/kinova-gen3-ros2 ${KINOVA_REF} /tmp/kinova \
    && cp -r /tmp/kinova/kinova_gen3_description src/ \
    && rm -rf /tmp/interfaces /tmp/curobo /tmp/kinova

RUN . /opt/ros/humble/setup.sh && colcon build --event-handlers console_direct-

# Cornell arm model (URDF + meshes), straight from the students' repo.
RUN git clone -q --filter=blob:none --no-checkout https://github.com/dhyi-svg/feeding-deployment /tmp/fd \
    && git -C /tmp/fd sparse-checkout set src/feeding_deployment/assets/robot \
    && git -C /tmp/fd checkout -q ${CORNELL_REF} \
    && mkdir -p /fixtures \
    && cp -r /tmp/fd/src/feeding_deployment/assets/robot /fixtures/cornell \
    && rm -rf /tmp/fd

# One URDF per robot, rendered from its mount.yaml exactly as description.launch.py does.
RUN . /ws/install/setup.sh && fetch-commit rammp-org/rammp-deployments ${DEPLOYMENTS_REF} /tmp/dep \
    && mkdir -p /fixtures/urdf \
    && for r in dojo flatbot gen_1_5; do \
        python3 /ws/src/kinova_gen3_description/scripts/kinova_gen3_render.py \
            --mount /tmp/dep/december_2026/config/robots/$r/mount.yaml \
            --out /fixtures/urdf/$r.urdf; \
    done \
    && rm -rf /tmp/dep

# capability_engine from RAMMP at the same commit (its test dir has fake_driver.py).
ARG RAMMP_REF=c3989c6
RUN git clone -q --filter=blob:none https://github.com/rammp-org/rammp-kitchen-capability-engines /tmp/rammp \
    && git -C /tmp/rammp checkout -q ${RAMMP_REF} \
    && cp -r /tmp/rammp/capability_engine /opt/capability_engine && rm -rf /tmp/rammp


ENV PYTHONPATH=/opt/capability_engine:/opt/capability_engine/test:/repo/puppy_playground
RUN printf '%s\n' '#!/bin/bash' \
        '. /opt/ros/humble/setup.bash' '. /ws/install/setup.bash' 'exec "$@"' \
        > /entrypoint.sh \
    && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /repo
