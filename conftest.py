"""
VayuSwarm conftest — Block ROS2 pytest plugins from interfering.
"""

import sys

# Block ROS2 modules from being imported by anything downstream
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
    sys.modules.setdefault(mod, None)  # type: ignore


# ROS2 Jazzy registers pytest plugins via entry points (from /opt/ros/jazzy/),
# which load BEFORE conftest.py. We unregister them here via pytest_configure,
# which fires right after all plugins are loaded but before collection.
_ROS_PLUGIN_NAMES = [
    "launch_ros",                          # actual entry point name for launch_testing_ros_pytest_entrypoint
    "launch_testing_ros_pytest_entrypoint",
    "launch-testing-ros",
    "launch_testing",
    "ament_copyright",
    "ament_xmllint",
    "ament_lint",
    "ament_flake8",
    "ament_pep257",
]


def pytest_configure(config):
    """Unregister ROS2 pytest plugins before they can cause hook validation errors."""
    pm = config.pluginmanager
    for plugin_name in _ROS_PLUGIN_NAMES:
        plugin = pm.get_plugin(plugin_name)
        if plugin is not None:
            pm.unregister(plugin)
