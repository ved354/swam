#!/usr/bin/env python3
"""
VayuSwarm — Launch a single drone agent.

Usage:
    python scripts/launch_drone.py --id drone_01
    python scripts/launch_drone.py --id drone_01 --ground tcp://192.168.1.10:5555
    python scripts/launch_drone.py --test-vision --input sample.jpg
"""

import asyncio
import sys
from pathlib import Path

import click
import yaml
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)


@click.command()
@click.option("--id", "drone_id", default="drone_01", help="Drone ID")
@click.option("--config", "config_path", default="config/default.yaml", help="Config file path")
@click.option("--ground", "ground_addr", default="tcp://localhost:5555", help="Ground station address")
@click.option("--simulation/--no-simulation", default=True, help="Simulation mode")
@click.option("--test-vision", is_flag=True, help="Test vision pipeline only")
@click.option("--input", "input_image", default=None, help="Test image for vision pipeline")
def main(drone_id, config_path, ground_addr, simulation, test_vision, input_image):
    """Launch a VayuSwarm drone agent."""

    # Load config
    config = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)

    config.setdefault("system", {})["simulation"] = simulation

    if test_vision:
        _test_vision(config, input_image)
        return

    from src.drone.agent import DroneAgent

    click.echo(f"🛸 Launching drone agent: {drone_id}")
    click.echo(f"   Ground station: {ground_addr}")
    click.echo(f"   Simulation: {simulation}")
    click.echo(f"   Config: {config_path}")

    agent = DroneAgent(drone_id=drone_id, config=config)

    try:
        asyncio.run(agent.start(ground_address=ground_addr))
    except KeyboardInterrupt:
        click.echo(f"\n🛬 Drone {drone_id} shutting down...")
        asyncio.run(agent.stop())


def _test_vision(config, input_image):
    """Test the vision pipeline on a single image."""
    import numpy as np
    from src.vision.yolo_detector import YOLODetector
    from src.vision.thermal_model import ThermalModel
    from src.vision.sensor_fusion import SensorFusion
    from src.vision.behavior_analyzer import BehaviorAnalyzer
    from proto.messages import GeoPoint

    click.echo("🔍 Testing vision pipeline...")

    # Load image or create test frame
    if input_image:
        import cv2
        frame = cv2.imread(input_image)
        if frame is None:
            click.echo(f"❌ Could not load image: {input_image}")
            return
    else:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    thermal = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    position = GeoPoint(lat=17.385, lon=78.487, alt=50)

    # Run pipeline
    yolo = YOLODetector()
    yolo.load()
    detections = yolo.detect(frame, position)
    click.echo(f"   YOLO detections: {len(detections)}")
    for d in detections:
        click.echo(f"     • {d.detection_class.value}: {d.confidence:.2f}")

    thermal_model = ThermalModel()
    thermal_model.load()
    thermal_dets = thermal_model.detect(thermal, position)
    click.echo(f"   Thermal detections: {len(thermal_dets)}")

    fusion = SensorFusion()
    fused = fusion.fuse(detections, thermal_dets, position)
    click.echo(f"   Fused events: {len(fused)}")
    for f in fused:
        click.echo(f"     • {f.to_llm_text()}")

    analyzer = BehaviorAnalyzer()
    analyzer.load()
    analyzed = analyzer.analyze(fused)
    click.echo(f"   Behavior analyzed: {len(analyzed)} events")

    click.echo("✅ Vision pipeline test complete!")


if __name__ == "__main__":
    main()
