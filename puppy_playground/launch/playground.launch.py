"""Playground engine + testing UI, standalone (no orchestrator). Never starts anything else."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="puppy_playground",
                executable="puppy_playground_node",
                name="puppy_playground",
                parameters=[{"use_orchestrator": False}],
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
