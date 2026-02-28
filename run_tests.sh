#!/bin/bash
# VayuSwarm Test Runner
# Strips /opt/ros from PYTHONPATH so ROS entry-point plugins never load.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Strip any /opt/ros paths from PYTHONPATH (ROS entry points use these to
# register pytest plugins before any Python code can block them).
CLEAN_PYTHONPATH=""
if [ -n "$PYTHONPATH" ]; then
    while IFS=: read -ra PARTS; do
        for p in "${PARTS[@]}"; do
            case "$p" in
                /opt/ros/*) ;;   # skip ROS paths
                *) CLEAN_PYTHONPATH="${CLEAN_PYTHONPATH:+$CLEAN_PYTHONPATH:}$p" ;;
            esac
        done
    done <<< "$PYTHONPATH"
fi

echo "🚀 Running VayuSwarm tests (ROS plugins excluded)..."
PYTHONPATH="$CLEAN_PYTHONPATH" \
AMENT_PREFIX_PATH="" \
ROS_PACKAGE_PATH="" \
    python -m pytest "$@"
