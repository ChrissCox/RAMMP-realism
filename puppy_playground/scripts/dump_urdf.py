"""Print the arm driver's URDF (/arm/robot_description, latched) to stdout; read-only.

    python3 dump_urdf.py > flatbot.urdf"""

import sys

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def main():
    rclpy.init()
    node = rclpy.create_node("puppy_dump_urdf")
    got = []
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, "/arm/robot_description", lambda m: got.append(m.data), qos)
    for _ in range(300):
        rclpy.spin_once(node, timeout_sec=0.1)
        if got:
            break
    node.destroy_node()
    rclpy.shutdown()
    if not got:
        sys.exit("no /arm/robot_description in 30 s: wrong ROS_DOMAIN_ID / RMW, or the arm driver is down")
    sys.stdout.write(got[0])


if __name__ == "__main__":
    main()
