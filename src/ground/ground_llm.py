"""
VayuSwarm — Ground LLM Interface (70B-class)

Strategic LLM running at the ground station.
Sees aggregated reports from ALL drones, has full mission context,
and makes high-level strategic decisions for the entire swarm.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import structlog

from proto.messages import (
    CommandPriority,
    CommandType,
    DroneReport,
    GeoPoint,
    GroundCommand,
    LLMDecision,
    MissionDefinition,
    ThreatLevel,
)

logger = structlog.get_logger(__name__)

GROUND_SYSTEM_PROMPT = """You are the Strategic AI Commander for VayuSwarm — a swarm drone surveillance system.

## Your Role
You manage {drone_count} surveillance drones. You receive aggregated intelligence reports from ALL drones and make strategic decisions for the entire swarm.

## Current Mission
{mission_context}

## Swarm Status
{swarm_status}

## Decision History (last {history_count} decisions)
{decision_history}

## Available Commands
For each drone, you can issue:
- REPOSITION: Send drone to new coordinates
- INVESTIGATE: Direct drone to examine specific area/target
- TRACK: Assign target tracking to a drone
- RTL: Order return to launch
- HOLD: Order hold position
- FORMATION: Change swarm formation
- REALLOCATE: Reassign drones between zones

## Output Format
Respond in JSON:
{{
    "commands": [
        {{
            "drone_id": "<drone_id>",
            "command": "<COMMAND>",
            "waypoint": {{"lat": <float>, "lon": <float>, "alt": <float>}} or null,
            "priority": "LOW|NORMAL|HIGH|CRITICAL",
            "message": "<directive for drone's local LLM>"
        }}
    ],
    "strategic_assessment": "<brief situation assessment>",
    "threat_summary": "<overall threat picture>",
    "changes_to_roe": {{}} or null
}}

## Rules
1. Maintain coverage — never leave an area unmonitored.
2. Prioritize CRITICAL threats above mission objectives.
3. If drone battery < 30%, plan for replacement coverage.
4. Correlate detections across drones for better intelligence.
5. Communicate clear, concise directives.
"""


class GroundLLM:
    """
    70B-class strategic LLM for the ground station.
    
    Makes high-level swarm management decisions based on 
    aggregated intelligence from all drones.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3:70b",
        endpoint: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout_s: float = 30.0,
    ):
        self._provider = provider
        self._model = model
        self._endpoint = endpoint
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._client = None
        self._decision_history: list[dict] = []
        self._decision_count = 0

    async def connect(self) -> None:
        """Initialize the HTTP client (skipped for mock provider)."""
        if self._provider == "mock":
            logger.info("ground_llm.mock_mode",
                         note="Using rule-based mock decisions (no LLM)")
            return
        import httpx
        self._client = httpx.AsyncClient(timeout=self._timeout)

        # Verify LLM server is reachable; fall back to mock if not
        try:
            resp = await self._client.get(self._endpoint, timeout=3.0)
            resp.raise_for_status()
            logger.info("ground_llm.connected", provider=self._provider, model=self._model)
        except Exception as e:
            logger.warning("ground_llm.server_unreachable",
                           endpoint=self._endpoint,
                           error=str(e),
                           fallback="mock")
            await self._client.aclose()
            self._client = None
            self._provider = "mock"

    async def strategize(
        self,
        drone_reports: dict[str, DroneReport],
        mission: Optional[MissionDefinition] = None,
    ) -> list[GroundCommand]:
        """
        Make strategic decisions based on aggregated drone intelligence.
        
        Args:
            drone_reports: Dict of drone_id → latest DroneReport
            mission: Current mission definition
            
        Returns:
            List of GroundCommand to send to individual drones
        """
        start_time = time.time()
        self._decision_count += 1

        # Build context
        system_prompt = self._build_system_prompt(drone_reports, mission)
        user_message = self._build_intelligence_summary(drone_reports)

        try:
            if self._client and self._provider == "ollama":
                response = await self._call_ollama(system_prompt, user_message)
            elif self._client:
                response = await self._call_openai_compatible(system_prompt, user_message)
            else:
                response = self._mock_strategize(drone_reports)

            commands = self._parse_response(response, drone_reports)

            latency_ms = (time.time() - start_time) * 1000
            logger.info("ground_llm.decision",
                        commands_issued=len(commands),
                        latency_ms=round(latency_ms, 1))

            # Record in history
            self._decision_history.append({
                "timestamp": time.time(),
                "commands": len(commands),
                "drones": len(drone_reports),
            })
            if len(self._decision_history) > 50:
                self._decision_history = self._decision_history[-50:]

            return commands

        except Exception as e:
            logger.error("ground_llm.strategize_error", error=str(e))
            return []

    async def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        """Call Ollama API."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        timeout = max(self._timeout, 120.0) if self._decision_count <= 1 else self._timeout
        response = await self._client.post(
            f"{self._endpoint}/api/chat", json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    async def _call_openai_compatible(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI-compatible API."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        response = await self._client.post(f"{self._endpoint}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _build_system_prompt(
        self,
        reports: dict[str, DroneReport],
        mission: Optional[MissionDefinition],
    ) -> str:
        """Build the strategic system prompt with current context."""
        # Swarm status
        status_lines = []
        for drone_id, report in reports.items():
            t = report.telemetry
            status_lines.append(
                f"  {drone_id}: State={t.state.value}, Pos=({t.position.lat:.4f},{t.position.lon:.4f}), "
                f"Alt={t.position.alt:.0f}m, Battery={t.battery_pct:.0f}%, "
                f"Events={len(report.fused_events)}, Action={report.current_action}"
            )
        swarm_status = "\n".join(status_lines) if status_lines else "  No drones reporting"

        # Mission context
        mission_ctx = "Standard surveillance patrol"
        if mission:
            mission_ctx = f"Mission: {mission.name}\nObjectives:\n"
            for obj in mission.objectives:
                status = "✓" if obj.completed else "○"
                mission_ctx += f"  {status} {obj.description}\n"

        # Decision history
        history = ""
        for entry in self._decision_history[-5:]:
            history += f"  [{entry['timestamp']:.0f}] {entry['commands']} commands to {entry['drones']} drones\n"

        return GROUND_SYSTEM_PROMPT.format(
            drone_count=len(reports),
            mission_context=mission_ctx,
            swarm_status=swarm_status,
            history_count=len(self._decision_history),
            decision_history=history or "  No previous decisions",
        )

    def _build_intelligence_summary(self, reports: dict[str, DroneReport]) -> str:
        """Build intelligence summary from all drone reports."""
        lines = [f"=== INTELLIGENCE BRIEFING ({len(reports)} drones reporting) ===\n"]

        all_events = []
        for drone_id, report in reports.items():
            lines.append(f"── {drone_id} ──")
            lines.append(f"  State: {report.telemetry.state.value}")
            lines.append(f"  Battery: {report.telemetry.battery_pct:.0f}%")
            lines.append(f"  Position: ({report.telemetry.position.lat:.6f}, {report.telemetry.position.lon:.6f})")

            if report.fused_events:
                lines.append(f"  Detections ({len(report.fused_events)}):")
                for event in report.fused_events:
                    lines.append(f"    • {event.to_llm_text()}")
                    all_events.append(event)
            else:
                lines.append("  Detections: None")

            if report.safety_vetoes:
                lines.append(f"  Safety vetoes: {', '.join(report.safety_vetoes)}")
            lines.append("")

        # Threat summary
        critical = [e for e in all_events if e.threat_level == ThreatLevel.CRITICAL]
        high = [e for e in all_events if e.threat_level == ThreatLevel.HIGH]
        if critical or high:
            lines.append(f"⚠ PRIORITY: {len(critical)} CRITICAL, {len(high)} HIGH threats detected")

        lines.append("\n=== END BRIEFING ===")
        lines.append("\nProvide strategic commands for the swarm. Respond in JSON format.")

        return "\n".join(lines)

    def _parse_response(self, raw: str, reports: dict[str, DroneReport]) -> list[GroundCommand]:
        """Parse LLM response into GroundCommand objects."""
        commands = []

        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(raw[json_start:json_end])

                for cmd_data in data.get("commands", []):
                    drone_id = cmd_data.get("drone_id", "")
                    if drone_id not in reports:
                        continue

                    cmd_type = self._map_command_type(cmd_data.get("command", "HOLD"))
                    waypoint = None
                    if cmd_data.get("waypoint"):
                        wp = cmd_data["waypoint"]
                        waypoint = GeoPoint(
                            lat=wp.get("lat", 0),
                            lon=wp.get("lon", 0),
                            alt=wp.get("alt", 50),
                        )

                    priority_str = cmd_data.get("priority", "NORMAL")
                    try:
                        priority = CommandPriority(priority_str)
                    except ValueError:
                        priority = CommandPriority.NORMAL

                    commands.append(GroundCommand(
                        source_id="ground",
                        target_drone_id=drone_id,
                        command_type=cmd_type,
                        priority=priority,
                        waypoint=waypoint,
                        message=cmd_data.get("message", ""),
                    ))

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("ground_llm.parse_error", error=str(e))

        return commands

    @staticmethod
    def _map_command_type(command: str) -> CommandType:
        """Map LLM command string to CommandType enum."""
        command_map = {
            "REPOSITION": CommandType.GOTO_WAYPOINT,
            "GOTO": CommandType.GOTO_WAYPOINT,
            "INVESTIGATE": CommandType.INVESTIGATE_TARGET,
            "TRACK": CommandType.TRACK_TARGET,
            "RTL": CommandType.RETURN_TO_LAUNCH,
            "HOLD": CommandType.HOLD_POSITION,
            "LAND": CommandType.LAND,
            "FORMATION": CommandType.FORMATION_CHANGE,
        }
        return command_map.get(command.upper(), CommandType.HOLD_POSITION)

    def _mock_strategize(self, reports: dict[str, DroneReport]) -> str:
        """Mock strategic decision for testing."""
        commands = []
        for drone_id, report in reports.items():
            # If drone has high threats, tell it to track
            high_threats = [e for e in report.fused_events
                           if e.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH)]

            if high_threats:
                event = high_threats[0]
                wp = None
                if event.geo_position:
                    wp = {"lat": event.geo_position.lat,
                          "lon": event.geo_position.lon,
                          "alt": 60}
                commands.append({
                    "drone_id": drone_id,
                    "command": "TRACK",
                    "waypoint": wp,
                    "priority": "HIGH",
                    "message": f"Track {event.detection_class.value} target. Maintain safe distance.",
                })
            elif report.telemetry.battery_pct < 30:
                commands.append({
                    "drone_id": drone_id,
                    "command": "RTL",
                    "waypoint": None,
                    "priority": "HIGH",
                    "message": "Battery low, return to base.",
                })

        return json.dumps({
            "commands": commands,
            "strategic_assessment": "Active surveillance in progress",
            "threat_summary": f"{len(reports)} drones operating nominally",
            "changes_to_roe": None,
        })

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()

    @property
    def stats(self) -> dict:
        return {
            "decisions": self._decision_count,
            "model": self._model,
            "history_size": len(self._decision_history),
        }
