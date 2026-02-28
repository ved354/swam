#!/usr/bin/env python3
"""
VayuSwarm — Launch dashboard.

Usage:
    python scripts/launch_dashboard.py
    python scripts/launch_dashboard.py --port 8080
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
@click.option("--host", default="0.0.0.0", help="Dashboard host")
@click.option("--port", default=8080, help="Dashboard port")
def main(config_path, host, port):
    """Launch VayuSwarm dashboard."""

    config = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)

    from src.dashboard.server import DashboardServer

    click.echo(f"📊 Launching VayuSwarm Dashboard")
    click.echo(f"   URL: http://{host}:{port}")

    server = DashboardServer(host=host, port=port)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        click.echo("\n📊 Dashboard shutting down...")


if __name__ == "__main__":
    main()
