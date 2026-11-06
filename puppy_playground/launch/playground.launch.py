"""Playground engine + testing UI, standalone (no orchestrator). Never starts anything else.

    ros2 launch puppy_playground playground.launch.py                 # real arm
    ros2 launch puppy_playground playground.launch.py mock:=true      # dry run: no goal is sent
    ros2 launch puppy_playground playground.launch.py mock:=true mock_urdf:=/data/flatbot.urdf
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_orchestrator", default_value="false"),
            DeclareLaunchArgument("mock", default_value="false"),
            # arm.mock_urdf: the URDF the arm model is built from in mock runs ("" loads none)
            DeclareLaunchArgument("mock_urdf", default_value=""),
            Node(
                package="puppy_playground",
                executable="puppy_playground_node",
                name="puppy_playground",
                parameters=[
                    {
                        "use_orchestrator": ParameterValue(LaunchConfiguration("use_orchestrator"), value_type=bool),
                        "mock": ParameterValue(LaunchConfiguration("mock"), value_type=bool),
                        "arm.mock_urdf": ParameterValue(LaunchConfiguration("mock_urdf"), value_type=str),
                    }
                ],
                output="screen",
            ),
            Node(
                package="puppy_playground",
                executable="puppy_playground_ui",
                name="puppy_playground_ui",
                output="screen",
            ),
        ]
    )
