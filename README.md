# VayuSwarm — AI-Powered Swarm Drone Software

An autonomous swarm drone system with a hierarchical AI architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GROUND STATION                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Ground LLM  (70B class)                     │  │
│  │  • Strategic decisions across entire swarm               │  │
│  │  • Full mission context & history                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────┬──────────────────────────────┬─────────────┘
                    │  (ZeroMQ pub/sub)             │
          ┌─────────▼──────────┐          ┌────────▼───────────┐
          │      DRONE 1       │◄────────►│      DRONE 2       │
          │  Local LLM (3B)    │  mesh    │  Local LLM (3B)    │
          │  Safety Layer      │  comms   │  Safety Layer      │
          │  Vision Pipeline   │          │  Vision Pipeline   │
          │  PX4 (MAVLink)     │          │  PX4 (MAVLink)     │
          └────────────────────┘          └────────────────────┘
```

## Vision Pipeline (Per Drone)

```
Camera Frame → YOLOv8 → Thermal Model → Sensor Fusion → Behavior Analyzer → LLM → Safety → PX4
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Launch simulated swarm (3 drones)
python scripts/launch_swarm.py --drones 3 --simulation

# Launch dashboard
python scripts/launch_dashboard.py
# Open http://localhost:8080
```

## Project Structure

```
drones/
├── config/          # YAML configuration files
├── proto/           # Message protocol schemas (Pydantic)
├── src/
│   ├── comms/       # ZeroMQ message bus + mesh network
│   ├── vision/      # YOLOv8, thermal, fusion, behavior
│   ├── drone/       # Drone agent, LLM, safety, PX4
│   ├── ground/      # Ground station, strategic LLM, fleet
│   └── dashboard/   # Real-time web dashboard
├── scripts/         # Launch scripts
└── tests/           # Test suite
```
