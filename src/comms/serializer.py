"""
VayuSwarm — Message Serializer

Handles serialization/deserialization of Pydantic message models
to/from JSON bytes for ZeroMQ transport.
"""

from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class MessageSerializer:
    """Serialize Pydantic models for ZeroMQ transport."""

    # Topic separator in ZeroMQ multipart messages
    TOPIC_SEP = b"|"

    @staticmethod
    def serialize(msg: BaseModel, topic: str = "") -> list[bytes]:
        """
        Serialize a Pydantic message into ZeroMQ multipart frames.

        Returns [topic_frame, type_frame, payload_frame].
        """
        type_name = type(msg).__name__
        payload = msg.model_dump_json().encode("utf-8")

        return [
            topic.encode("utf-8"),
            type_name.encode("utf-8"),
            payload,
        ]

    @staticmethod
    def deserialize(frames: list[bytes], model_registry: dict[str, Type[BaseModel]]) -> tuple[str, BaseModel]:
        """
        Deserialize ZeroMQ multipart frames back into a Pydantic model.

        Args:
            frames: [topic_frame, type_frame, payload_frame]
            model_registry: mapping of type name -> Pydantic model class

        Returns:
            (topic, model_instance)
        """
        if len(frames) < 3:
            raise ValueError(f"Expected 3 frames, got {len(frames)}")

        topic = frames[0].decode("utf-8")
        type_name = frames[1].decode("utf-8")
        payload = frames[2].decode("utf-8")

        model_class = model_registry.get(type_name)
        if model_class is None:
            raise ValueError(f"Unknown message type: {type_name}")

        return topic, model_class.model_validate_json(payload)

    @staticmethod
    def serialize_simple(msg: BaseModel) -> bytes:
        """Serialize to raw JSON bytes (for mesh / simple comms)."""
        wrapper = {
            "_type": type(msg).__name__,
            "_data": json.loads(msg.model_dump_json()),
        }
        return json.dumps(wrapper).encode("utf-8")

    @staticmethod
    def deserialize_simple(data: bytes, model_registry: dict[str, Type[BaseModel]]) -> BaseModel:
        """Deserialize raw JSON bytes back into a Pydantic model."""
        wrapper = json.loads(data.decode("utf-8"))
        type_name = wrapper["_type"]
        model_class = model_registry.get(type_name)
        if model_class is None:
            raise ValueError(f"Unknown message type: {type_name}")
        return model_class.model_validate(wrapper["_data"])


# ─── Default Registry ──────────────────────────────────────────────────────────

def build_message_registry() -> dict[str, Type[BaseModel]]:
    """Build the default message type registry from proto.messages."""
    from proto.messages import (
        DroneTelemetry,
        DetectionEvent,
        ThermalDetection,
        FusedEvent,
        DroneReport,
        GroundCommand,
        SwarmMessage,
        SafetyVeto,
        MissionDefinition,
        LLMDecision,
    )

    return {
        cls.__name__: cls
        for cls in [
            DroneTelemetry,
            DetectionEvent,
            ThermalDetection,
            FusedEvent,
            DroneReport,
            GroundCommand,
            SwarmMessage,
            SafetyVeto,
            MissionDefinition,
            LLMDecision,
        ]
    }
