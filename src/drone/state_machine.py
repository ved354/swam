"""
VayuSwarm — Drone State Machine

Finite State Machine (FSM) managing drone lifecycle and state transitions.

States: IDLE → PREFLIGHT → TAKEOFF → PATROL → INVESTIGATE → TRACK → RTL → LAND
        EMERGENCY is reachable from any state.

Each state has entry/exit actions and allowed transitions.
Transitions are triggered by LLM decisions, safety vetoes, or ground commands.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import structlog

from proto.messages import DroneState

logger = structlog.get_logger(__name__)


class StateTransition:
    """Represents an allowed state transition."""

    def __init__(
        self,
        from_state: DroneState,
        to_state: DroneState,
        condition: Optional[Callable[[], bool]] = None,
        on_transition: Optional[Callable[[], None]] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition or (lambda: True)
        self.on_transition = on_transition


class DroneStateMachine:
    """
    Finite State Machine for drone lifecycle.
    
    State diagram:
    
    IDLE → PREFLIGHT → TAKEOFF → PATROL ─┬─→ INVESTIGATE → TRACK ──┐
      ↑                          ↑        │                          │
      │                          └────────┴──────────────────────────┘
      │                                   │
      ├──── LAND ←── RTL ←───────────────┘
      │
      └──── EMERGENCY (from any state)
    """

    # Allowed transitions
    TRANSITIONS = {
        DroneState.IDLE: [DroneState.PREFLIGHT, DroneState.EMERGENCY],
        DroneState.PREFLIGHT: [DroneState.TAKEOFF, DroneState.IDLE, DroneState.EMERGENCY],
        DroneState.TAKEOFF: [DroneState.PATROL, DroneState.RTL, DroneState.EMERGENCY],
        DroneState.PATROL: [
            DroneState.INVESTIGATE, DroneState.TRACK, DroneState.RTL,
            DroneState.LAND, DroneState.EMERGENCY,
        ],
        DroneState.INVESTIGATE: [
            DroneState.TRACK, DroneState.PATROL, DroneState.RTL,
            DroneState.EMERGENCY,
        ],
        DroneState.TRACK: [
            DroneState.INVESTIGATE, DroneState.PATROL, DroneState.RTL,
            DroneState.EMERGENCY,
        ],
        DroneState.ENGAGE: [
            DroneState.TRACK, DroneState.PATROL, DroneState.RTL,
            DroneState.EMERGENCY,
        ],
        DroneState.RTL: [DroneState.LAND, DroneState.PATROL, DroneState.EMERGENCY],
        DroneState.LAND: [DroneState.IDLE, DroneState.EMERGENCY],
        DroneState.EMERGENCY: [DroneState.LAND, DroneState.IDLE],
        DroneState.OFFLINE: [DroneState.IDLE],
    }

    def __init__(self, drone_id: str, initial_state: DroneState = DroneState.IDLE):
        self._drone_id = drone_id
        self._state = initial_state
        self._previous_state: Optional[DroneState] = None
        self._state_enter_time = time.time()
        self._transition_history: list[dict] = []
        self._on_enter_callbacks: dict[DroneState, list[Callable]] = {}
        self._on_exit_callbacks: dict[DroneState, list[Callable]] = {}
        self._on_any_transition: list[Callable] = []

    @property
    def state(self) -> DroneState:
        return self._state

    @property
    def previous_state(self) -> Optional[DroneState]:
        return self._previous_state

    @property
    def time_in_state(self) -> float:
        """Time spent in current state (seconds)."""
        return time.time() - self._state_enter_time

    def can_transition(self, target: DroneState) -> bool:
        """Check if a transition to the target state is allowed."""
        if target == DroneState.EMERGENCY:
            return True  # EMERGENCY is always reachable
        allowed = self.TRANSITIONS.get(self._state, [])
        return target in allowed

    def transition(self, target: DroneState, reason: str = "") -> bool:
        """
        Attempt a state transition.
        
        Returns True if the transition was successful, False if not allowed.
        """
        if not self.can_transition(target):
            logger.warning(
                "fsm.transition_denied",
                drone_id=self._drone_id,
                current=self._state.value,
                target=target.value,
                reason=reason,
            )
            return False

        old_state = self._state

        # Exit callbacks
        for cb in self._on_exit_callbacks.get(old_state, []):
            try:
                cb(old_state, target)
            except Exception as e:
                logger.error("fsm.exit_callback_error", error=str(e))

        # Transition
        self._previous_state = old_state
        self._state = target
        self._state_enter_time = time.time()

        # Record history
        record = {
            "from": old_state.value,
            "to": target.value,
            "reason": reason,
            "timestamp": time.time(),
        }
        self._transition_history.append(record)

        logger.info(
            "fsm.transition",
            drone_id=self._drone_id,
            from_state=old_state.value,
            to_state=target.value,
            reason=reason,
        )

        # Enter callbacks
        for cb in self._on_enter_callbacks.get(target, []):
            try:
                cb(old_state, target)
            except Exception as e:
                logger.error("fsm.enter_callback_error", error=str(e))

        # Any-transition callbacks
        for cb in self._on_any_transition:
            try:
                cb(old_state, target, reason)
            except Exception as e:
                logger.error("fsm.transition_callback_error", error=str(e))

        return True

    def on_enter(self, state: DroneState, callback: Callable) -> None:
        """Register a callback for when entering a state."""
        if state not in self._on_enter_callbacks:
            self._on_enter_callbacks[state] = []
        self._on_enter_callbacks[state].append(callback)

    def on_exit(self, state: DroneState, callback: Callable) -> None:
        """Register a callback for when exiting a state."""
        if state not in self._on_exit_callbacks:
            self._on_exit_callbacks[state] = []
        self._on_exit_callbacks[state].append(callback)

    def on_transition(self, callback: Callable) -> None:
        """Register a callback for any state transition."""
        self._on_any_transition.append(callback)

    def action_to_state(self, action: str) -> Optional[DroneState]:
        """Map an LLM action string to a DroneState."""
        action_map = {
            "CONTINUE": None,          # Stay in current state
            "HOLD": None,              # Stay in current state
            "INVESTIGATE": DroneState.INVESTIGATE,
            "TRACK": DroneState.TRACK,
            "ALERT": None,             # Alert doesn't change state
            "AVOID": DroneState.PATROL,
            "RTL": DroneState.RTL,
            "LAND": DroneState.LAND,
            "TAKEOFF": DroneState.TAKEOFF,
            "PATROL": DroneState.PATROL,
            "EMERGENCY": DroneState.EMERGENCY,
        }
        return action_map.get(action.upper())

    @property
    def history(self) -> list[dict]:
        """Get transition history (last 20)."""
        return self._transition_history[-20:]

    def __repr__(self) -> str:
        return f"DroneStateMachine(drone={self._drone_id}, state={self._state.value})"
