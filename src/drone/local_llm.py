"""
VayuSwarm — Local LLM Interface (3B-class)

Interfaces with a local LLM (via Ollama or OpenAI-compatible API) to make
tactical decisions based on fused sensor data, mission context, and ROE.

The LLM receives structured text (never raw pixels) like:
    "person detected, 87% confidence, thermal+rgb, HIGH threat, in NGZ"
"""

from __future__ import annotations

import json
import time
from typing import Optional

import structlog
from pydantic import BaseModel

from proto.messages import (
    DroneState,
    FusedEvent,
    GeoPoint,
    LLMDecision,
    ThreatLevel,
)

logger = structlog.get_logger(__name__)

# ─── System Prompt ──────────────────────────────────────────────────────────────

DRONE_SYSTEM_PROMPT = """You are the tactical AI for an autonomous surveillance drone (ID: {drone_id}).

## Your Role
You make real-time tactical decisions based on sensor data. You receive structured detection events (never raw images) and output action decisions.

## Current Mission Context
- State: {state}
- Position: ({lat:.6f}, {lon:.6f}) at {alt:.0f}m
- Battery: {battery_pct:.0f}%
- Mission: {mission_description}

## Rules of Engagement (ROE)
{roe_text}

## Available Actions
- CONTINUE: Keep current behavior
- INVESTIGATE: Move closer to examine a detection (provide waypoint)
- TRACK: Follow a target at safe distance
- ALERT: Flag for ground station attention (HIGH priority uplink)
- AVOID: Change course to avoid area
- RTL: Return to launch
- HOLD: Hover in place and observe

## Output Format
Respond in JSON:
{{
    "action": "<ACTION>",
    "reasoning": "<brief explanation>",
    "confidence": <0.0-1.0>,
    "waypoint": {{"lat": <float>, "lon": <float>, "alt": <float>}} or null,
    "alert_ground": <true/false>,
    "priority": "LOW|NORMAL|HIGH|CRITICAL"
}}

## Rules
1. NEVER engage. You are surveillance only.
2. Safety is absolute. If uncertain, HOLD and ALERT.
3. Prioritize threats by level: CRITICAL > HIGH > MEDIUM > LOW.
4. If battery < 25%, RTL unless a CRITICAL threat is being tracked.
5. Always explain your reasoning clearly.
"""


class LocalLLM:
    """
    3B-class Local LLM for tactical drone decisions.
    
    Interfaces with Ollama or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        drone_id: str,
        provider: str = "ollama",
        model: str = "phi3:3.8b",
        endpoint: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 5.0,
    ):
        self._drone_id = drone_id
        self._provider = provider
        self._model = model
        self._endpoint = endpoint
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._client = None
        self._decision_count = 0
        self._total_latency_ms = 0.0

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        import httpx
        self._client = httpx.AsyncClient(timeout=self._timeout)
        logger.info("local_llm.connected",
                     drone_id=self._drone_id,
                     provider=self._provider,
                     model=self._model)

    async def decide(
        self,
        fused_events: list[FusedEvent],
        drone_state: DroneState,
        drone_position: GeoPoint,
        battery_pct: float,
        mission_description: str = "Surveillance patrol",
        roe: Optional[dict] = None,
    ) -> LLMDecision:
        """
        Make a tactical decision based on current sensor data.
        
        Args:
            fused_events: Current fused detections
            drone_state: Current drone state
            drone_position: Current GPS position
            battery_pct: Current battery percentage
            mission_description: Text description of current mission
            roe: Rules of engagement dict
            
        Returns:
            LLMDecision with action, reasoning, and optional waypoint
        """
        start_time = time.time()
        self._decision_count += 1

        # Build system prompt with current context
        roe = roe or {}
        roe_text = self._format_roe(roe)
        system_prompt = DRONE_SYSTEM_PROMPT.format(
            drone_id=self._drone_id,
            state=drone_state.value,
            lat=drone_position.lat,
            lon=drone_position.lon,
            alt=drone_position.alt,
            battery_pct=battery_pct,
            mission_description=mission_description,
            roe_text=roe_text,
        )

        # Build user message with sensor data
        user_message = self._build_sensor_message(fused_events)

        try:
            if self._client and self._provider == "ollama":
                response = await self._call_ollama(system_prompt, user_message)
            elif self._client:
                response = await self._call_openai_compatible(system_prompt, user_message)
            else:
                response = self._mock_decide(fused_events, battery_pct)

            decision = self._parse_response(response)
            latency_ms = (time.time() - start_time) * 1000
            self._total_latency_ms += latency_ms

            logger.info("local_llm.decision",
                        drone_id=self._drone_id,
                        action=decision.action,
                        confidence=decision.confidence,
                        latency_ms=round(latency_ms, 1))

            return decision

        except Exception as e:
            logger.error("local_llm.decide_error", error=str(e))
            # Default safe action
            return LLMDecision(
                source="local_llm",
                action="HOLD",
                reasoning=f"LLM error: {str(e)}. Defaulting to HOLD.",
                confidence=0.5,
            )

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

        response = await self._client.post(
            f"{self._endpoint}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def _call_openai_compatible(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI-compatible API (vLLM, LiteLLM, etc.)."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        response = await self._client.post(
            f"{self._endpoint}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _build_sensor_message(self, events: list[FusedEvent]) -> str:
        """Convert fused events to structured text for the LLM."""
        if not events:
            return "No detections in current frame. Area appears clear."

        lines = [f"=== SENSOR REPORT ({len(events)} detections) ===\n"]
        for i, event in enumerate(events, 1):
            lines.append(f"[{i}] {event.to_llm_text()}")
        lines.append("\n=== END REPORT ===")
        lines.append("\nWhat action should the drone take? Respond in JSON format.")
        return "\n".join(lines)

    def _format_roe(self, roe: dict) -> str:
        """Format ROE dict into readable text."""
        if not roe:
            return "- Standard surveillance ROE\n- No engagement authorized\n- Alert ground on HIGH+ threats"

        lines = []
        for key, value in roe.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"- {formatted_key}: {value}")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> LLMDecision:
        """Parse LLM JSON response into LLMDecision."""
        try:
            # Try to extract JSON from response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
                data = json.loads(json_str)

                waypoint = None
                if data.get("waypoint"):
                    wp = data["waypoint"]
                    waypoint = GeoPoint(
                        lat=wp.get("lat", 0),
                        lon=wp.get("lon", 0),
                        alt=wp.get("alt", 50),
                    )

                return LLMDecision(
                    source="local_llm",
                    action=data.get("action", "HOLD"),
                    reasoning=data.get("reasoning", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    suggested_waypoint=waypoint,
                    raw_response=raw,
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("local_llm.parse_error", error=str(e))

        # Fallback: extract action keyword
        raw_upper = raw.upper()
        for action in ["INVESTIGATE", "TRACK", "ALERT", "AVOID", "RTL", "HOLD", "CONTINUE"]:
            if action in raw_upper:
                return LLMDecision(
                    source="local_llm",
                    action=action,
                    reasoning=raw[:200],
                    confidence=0.4,
                    raw_response=raw,
                )

        return LLMDecision(
            source="local_llm",
            action="HOLD",
            reasoning="Could not parse LLM response",
            confidence=0.3,
            raw_response=raw,
        )

    def _mock_decide(self, events: list[FusedEvent], battery_pct: float) -> str:
        """Mock LLM decision for testing without an actual LLM."""
        if battery_pct < 20:
            return json.dumps({
                "action": "RTL",
                "reasoning": "Battery critically low, returning to launch",
                "confidence": 0.95,
                "waypoint": None,
                "alert_ground": True,
                "priority": "HIGH",
            })

        if not events:
            return json.dumps({
                "action": "CONTINUE",
                "reasoning": "No detections, continuing patrol",
                "confidence": 0.8,
                "waypoint": None,
                "alert_ground": False,
                "priority": "LOW",
            })

        # Find highest threat
        highest_threat = max(events, key=lambda e: list(ThreatLevel).index(e.threat_level))

        if highest_threat.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH):
            action = "ALERT"
            reasoning = f"HIGH/CRITICAL threat detected: {highest_threat.to_llm_text()}"
            wp = None
            if highest_threat.geo_position:
                wp = {"lat": highest_threat.geo_position.lat,
                      "lon": highest_threat.geo_position.lon,
                      "alt": 60}
        elif highest_threat.threat_level == ThreatLevel.MEDIUM:
            action = "INVESTIGATE"
            reasoning = f"MEDIUM threat, investigating: {highest_threat.to_llm_text()}"
            wp = None
            if highest_threat.geo_position:
                wp = {"lat": highest_threat.geo_position.lat,
                      "lon": highest_threat.geo_position.lon,
                      "alt": 40}
        else:
            action = "CONTINUE"
            reasoning = "Low/no threat detections, continuing patrol"
            wp = None

        return json.dumps({
            "action": action,
            "reasoning": reasoning,
            "confidence": 0.7,
            "waypoint": wp,
            "alert_ground": highest_threat.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH),
            "priority": "HIGH" if highest_threat.threat_level.value >= ThreatLevel.HIGH.value else "NORMAL",
        })

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()

    @property
    def stats(self) -> dict:
        avg_latency = (self._total_latency_ms / self._decision_count) if self._decision_count > 0 else 0
        return {
            "decisions": self._decision_count,
            "avg_latency_ms": round(avg_latency, 1),
            "model": self._model,
        }
