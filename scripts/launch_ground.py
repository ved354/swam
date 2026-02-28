#!/usr/bin/env python3
"""
VayuSwarm — Launch the ground station.

Usage:
    python scripts/launch_ground.py
    python scripts/launch_ground.py --drones drone_01,drone_02,drone_03
"""

import asyncio
import sys
from pathlib import Path

import click
import yaml
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Config file")
@click.option("--drones", default="drone_01,drone_02,drone_03", help="Comma-separated drone IDs")
def main(config_path, drones):
    """Launch VayuSwarm ground station."""

    config = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)

    drone_ids = [d.strip() for d in drones.split(",")]

    from src.ground.station import GroundStation

    click.echo("🏢 Launching VayuSwarm Ground Station")
    click.echo(f"   Drones: {', '.join(drone_ids)}")

    station = GroundStation(config=config)

    try:
        asyncio.run(station.start(drone_ids=drone_ids))
    except KeyboardInterrupt:
        click.echo("\n🛑 Ground station shutting down...")
        asyncio.run(station.stop())


if __name__ == "__main__":
    main()
