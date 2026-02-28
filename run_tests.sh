#!/bin/bash
# VayuSwarm Test Runner
# Runs pytest with ROS2 system plugins disabled (they conflict with venv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Run pytest with PYTHONDONTWRITEBYTECODE and override site-packages
# to prevent ROS2 launch_testing plugins from loading
PYTHONDONTWRITEBYTECODE=1 \
  python -c "
import sys
# Block ROS2 modules before pytest loads
for mod_name in ['launch_testing', 'launch_testing_ros', 'launch_testing_ros_pytest_entrypoint', 'launch', 'lark']:
    sys.modules[mod_name] = type(sys)('blocked')
    sys.modules[mod_name].__path__ = []
    sys.modules[mod_name].__file__ = ''

# Now run pytest
from pytest import console_main
console_main()
" "$@"
