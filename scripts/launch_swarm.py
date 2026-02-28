#!/usr/bin/env python3
"""
VayuSwarm — Launch full swarm simulation.

Starts ground station + N drone agents in simulation mode.

Usage:
    python scripts/launch_swarm.py --drones 3
    python scripts/launch_swarm.py --drones 5 --simulation
"""

import asyncio
import copy
import os
import signal
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

logger = structlog.get_logger(__name__)


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Config file")
@click.option("--drones", "drone_count", default=3, help="Number of drones")
@click.option("--simulation/--no-simulation", default=True, help="Simulation mode")
@click.option("--dashboard/--no-dashboard", default=True, help="Launch dashboard")
@click.option("--dashboard-port", default=8080, help="Dashboard port")
def main(config_path, drone_count, simulation, dashboard, dashboard_port):
    """Launch VayuSwarm full swarm simulation."""

    config = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)

    config.setdefault("system", {})["simulation"] = simulation

    drone_ids = [f"drone_{i+1:02d}" for i in range(drone_count)]

    click.echo("═" * 60)
    click.echo("  🚀 VAYUSWARM — SWARM SIMULATION")
    click.echo("═" * 60)
    click.echo(f"  Drones:      {drone_count}")
    click.echo(f"  Simulation:  {simulation}")
    click.echo(f"  Dashboard:   {'http://localhost:' + str(dashboard_port) if dashboard else 'disabled'}")
    click.echo(f"  Drone IDs:   {', '.join(drone_ids)}")
    click.echo("═" * 60)

    asyncio.run(_run_swarm(config, drone_ids, dashboard, dashboard_port))


async def _run_swarm(config, drone_ids, launch_dashboard, dashboard_port):
    """Run the full swarm simulation."""
    from src.ground.station import GroundStation
    from src.drone.agent import DroneAgent
    from src.dashboard.server import DashboardServer

    tasks = []
    agents = []

    # 1. Start ground station
    station = GroundStation(config=config)
    ground_task = asyncio.create_task(_run_ground(station, drone_ids))
    tasks.append(ground_task)

    # Wait for ground to bind
    await asyncio.sleep(1.0)

    # 2. Start drones
    comms_cfg = config.get("comms", {})
    ground_pub_port = comms_cfg.get("ground_pub_port", 5555)
    ground_addr = f"tcp://localhost:{ground_pub_port}"

    for i, drone_id in enumerate(drone_ids):
        # Offset home positions for visual separation
        drone_config = copy.deepcopy(config)
        drone_config.setdefault("drone", {}).setdefault("home", {})
        drone_config["drone"]["home"]["lat"] = 17.385 + (i * 0.002)
        drone_config["drone"]["home"]["lon"] = 78.487 + (i * 0.003)

        agent = DroneAgent(drone_id=drone_id, config=drone_config)
        agents.append(agent)
        task = asyncio.create_task(_run_drone(agent, ground_addr, drone_id))
        tasks.append(task)
        await asyncio.sleep(0.5)  # Stagger launches

    # 3. Start dashboard
    if launch_dashboard:
        dashboard_server = DashboardServer(
            ground_station=station,
            port=dashboard_port,
        )
        dashboard_task = asyncio.create_task(dashboard_server.start())
        tasks.append(dashboard_task)

    logger.info("swarm.all_systems_launched",
                drones=len(drone_ids),
                dashboard=launch_dashboard)

    # Graceful shutdown on SIGINT / SIGTERM
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler():
        logger.info("swarm.signal_received, initiating shutdown")
        shutdown_event.set()
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Wait for all tasks or shutdown
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("swarm.shutting_down")
        for agent in agents:
            try:
                await agent.stop()
            except Exception as e:
                logger.error("swarm.agent_stop_error", error=str(e))
        try:
            await station.stop()
        except Exception as e:
            logger.error("swarm.station_stop_error", error=str(e))
        logger.info("swarm.shutdown_complete")


async def _run_ground(station, drone_ids):
    """Run ground station."""
    try:
        await station.start(drone_ids=drone_ids)
    except Exception as e:
        logger.error("swarm.ground_error", error=str(e))


async def _run_drone(agent, ground_addr, drone_id):
    """Run a single drone agent."""
    try:
        await agent.start(ground_address=ground_addr)
    except Exception as e:
        logger.error("swarm.drone_error", drone_id=drone_id, error=str(e))


if __name__ == "__main__":
    main()
