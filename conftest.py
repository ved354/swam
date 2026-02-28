"""
VayuSwarm conftest — Block ROS2 pytest plugins from interfering.
"""

import sys

# Block ROS2 packages from being imported during testing
_ROS_BLOCKED = [
    "launch_testing",
    "launch_testing_ros",
    "launch_testing_ros_pytest_entrypoint",
    "launch",
    "ament_copyright",
    "ament_xmllint",
    "ament_lint",
    "ament_flake8",
    "ament_pep257",
]

for mod in _ROS_BLOCKED:
    if mod not in sys.modules:
        sys.modules[mod] = None  # type: ignore
