"""
VayuSwarm — Drone Agent Tests
"""

import pytest
from proto.messages import DroneState
from src.drone.state_machine import DroneStateMachine


class TestStateMachine:
    def test_initial_state(self):
        fsm = DroneStateMachine("test_drone")
        assert fsm.state == DroneState.IDLE

    def test_valid_transition(self):
        fsm = DroneStateMachine("test_drone")
        assert fsm.transition(DroneState.PREFLIGHT, reason="test")
        assert fsm.state == DroneState.PREFLIGHT

    def test_invalid_transition(self):
        fsm = DroneStateMachine("test_drone")
        assert not fsm.transition(DroneState.PATROL, reason="test")
        assert fsm.state == DroneState.IDLE  # Unchanged

    def test_emergency_from_any_state(self):
        for state in DroneState:
            if state == DroneState.OFFLINE:
                continue
            fsm = DroneStateMachine("test_drone", initial_state=state)
            assert fsm.can_transition(DroneState.EMERGENCY)

    def test_full_lifecycle(self):
        fsm = DroneStateMachine("test_drone")
        assert fsm.transition(DroneState.PREFLIGHT)
        assert fsm.transition(DroneState.TAKEOFF)
        assert fsm.transition(DroneState.PATROL)
        assert fsm.transition(DroneState.INVESTIGATE)
        assert fsm.transition(DroneState.TRACK)
        assert fsm.transition(DroneState.RTL)
        assert fsm.transition(DroneState.LAND)
        assert fsm.transition(DroneState.IDLE)

    def test_history(self):
        fsm = DroneStateMachine("test_drone")
        fsm.transition(DroneState.PREFLIGHT)
        fsm.transition(DroneState.TAKEOFF)
        assert len(fsm.history) == 2

    def test_action_to_state_mapping(self):
        fsm = DroneStateMachine("test_drone")
        assert fsm.action_to_state("INVESTIGATE") == DroneState.INVESTIGATE
        assert fsm.action_to_state("RTL") == DroneState.RTL
        assert fsm.action_to_state("CONTINUE") is None
        assert fsm.action_to_state("HOLD") is None

    def test_callbacks(self):
        fsm = DroneStateMachine("test_drone")
        entered = []
        exited = []

        fsm.on_enter(DroneState.PREFLIGHT, lambda a, b: entered.append(b))
        fsm.on_exit(DroneState.IDLE, lambda a, b: exited.append(a))

        fsm.transition(DroneState.PREFLIGHT)
        assert DroneState.PREFLIGHT in entered
        assert DroneState.IDLE in exited
