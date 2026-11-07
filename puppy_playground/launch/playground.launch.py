"""Playground engine + testing UI, standalone (no orchestrator). Never starts anything else.

    ros2 launch puppy_playground playground.launch.py                  # real arm
    ros2 launch puppy_playground playground.launch.py plan_only:=true  # every action but plan_stage refuses
    ros2 launch puppy_playground playground.launch.py stage_floor_cap_m:=0.02  # attended, nobody in the chair
    ros2 launch puppy_playground playground.launch.py mock:=true mock_urdf:=/data/flatbot.urdf \
        mock_joints:='[0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725]'  # dry run

In mock mode (capability_engine arm.py, MOCK_* constants) the arm interface creates no ROS entity: a
motion logs "<label>: mock: not sent", succeeds, and becomes the state; the state starts at
`mock_joints` (arm.mock_joints, default below) and the model is `mock_urdf` (arm.mock_urdf).
"""

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# capability_engine.arm.MOCK_JOINTS (c3989c6): where a mock arm starts unless told otherwise
UPSTREAM_MOCK_JOINTS = "[0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]"


def generate_launch_description():
    def arg(name):
        return LaunchConfiguration(name)

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_orchestrator", default_value="false"),
            DeclareLaunchArgument("mock", default_value="false"),
            DeclareLaunchArgument("plan_only", default_value="false"),
            # 0.0 = off. 0.02 = ATTENDED floor cap, for attended tests with NOBODY in the chair; any other
            # value makes the engine refuse every action (bad_config). See BRINGUP.md.
            DeclareLaunchArgument("stage_floor_cap_m", default_value="0.0"),
            DeclareLaunchArgument("mock_urdf", default_value=""),
            DeclareLaunchArgument("mock_joints", default_value=UPSTREAM_MOCK_JOINTS),
            DeclareLaunchArgument("ui_port", default_value="8080"),
            Node(
                package="puppy_playground",
                executable="puppy_playground_node",
                name="puppy_playground",
                parameters=[
                    {
                        "use_orchestrator": ParameterValue(arg("use_orchestrator"), value_type=bool),
                        "mock": ParameterValue(arg("mock"), value_type=bool),
                        "plan_only": ParameterValue(arg("plan_only"), value_type=bool),
                        "playground.stage_floor_cap_m": ParameterValue(arg("stage_floor_cap_m"), value_type=float),
                        "arm.mock_urdf": ParameterValue(arg("mock_urdf"), value_type=str),
                        "arm.mock_joints": ParameterValue(arg("mock_joints"), value_type=List[float]),
                    }
                ],
                output="screen",
            ),
            Node(
                package="puppy_playground",
                executable="puppy_playground_ui",
                name="puppy_playground_ui",
                parameters=[{"playground.ui_port": ParameterValue(arg("ui_port"), value_type=int)}],
                output="screen",
            ),
        ]
    )
