#!/bin/bash
# Source ROS, the base image's interface overlay and this image's workspace, then run the command.
set -e
for f in /opt/ros/humble/setup.bash /ros2_ws/install/setup.bash /module_ws/install/setup.bash; do
  # shellcheck disable=SC1090
  [ -f "$f" ] && . "$f"
done
exec "$@"
