"""
VayuSwarm — ZeroMQ Message Bus

Pub/Sub message bus for ground station ↔ drone communication.
Supports topic-based filtering, async operation, and heartbeat monitoring.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Optional

import zmq
import zmq.asyncio
import structlog
from pydantic import BaseModel

from src.comms.serializer import MessageSerializer, build_message_registry

logger = structlog.get_logger(__name__)


class Publisher:
    """
    ZeroMQ PUB socket wrapper.
    
    Used by:
    - Ground station to publish commands to drones (bind mode)
    - Drones to publish reports to ground station (connect mode)
    """

    def __init__(self, address: str, bind: bool = True):
        self._address = address
        self._bind = bind
        self._ctx = zmq.asyncio.Context()
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._running = False

    async def start(self) -> None:
        """Start the PUB socket (bind or connect)."""
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1000)
        self._socket.setsockopt(zmq.LINGER, 0)
        if self._bind:
            self._socket.bind(self._address)
        else:
            self._socket.connect(self._address)
        self._running = True
        logger.info("publisher.started", address=self._address,
                     mode="bind" if self._bind else "connect")
        # Small delay for ZMQ slow-joiner problem
        await asyncio.sleep(0.2)

    async def publish(self, topic: str, msg: BaseModel) -> None:
        """Publish a message on a topic."""
        if not self._socket or not self._running:
            raise RuntimeError("Publisher not started")
        frames = MessageSerializer.serialize(msg, topic)
        await self._socket.send_multipart(frames)
        logger.debug("publisher.sent", topic=topic, type=type(msg).__name__)

    async def stop(self) -> None:
        """Close the PUB socket."""
        self._running = False
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("publisher.stopped")


class Subscriber:
    """
    ZeroMQ SUB socket wrapper with async message handling.
    
    Used by:
    - Ground station to subscribe to drone reports
    - Drones to subscribe to ground commands
    """

    def __init__(self, address: str, topics: Optional[list[str]] = None, bind: bool = False):
        self._address = address
        self._topics = topics or [""]  # empty string = subscribe to all
        self._bind = bind
        self._ctx = zmq.asyncio.Context()
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._running = False
        self._handlers: dict[str, list[Callable]] = {}
        self._registry = build_message_registry()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the SUB socket (bind or connect) and begin receiving."""
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1000)
        self._socket.setsockopt(zmq.LINGER, 0)

        for topic in self._topics:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)

        if self._bind:
            self._socket.bind(self._address)
        else:
            self._socket.connect(self._address)
        self._running = True
        self._task = asyncio.create_task(self._receive_loop())
        logger.info("subscriber.started", address=self._address,
                     mode="bind" if self._bind else "connect", topics=self._topics)

    def on_message(self, topic: str, handler: Callable[[str, BaseModel], Coroutine]) -> None:
        """Register an async handler for a topic prefix."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

    async def _receive_loop(self) -> None:
        """Main receive loop — dispatches messages to registered handlers."""
        while self._running:
            try:
                frames = await self._socket.recv_multipart()
                topic, msg = MessageSerializer.deserialize(frames, self._registry)

                # Find matching handlers
                matched = False
                for prefix, handlers in self._handlers.items():
                    if topic.startswith(prefix):
                        for handler in handlers:
                            try:
                                await handler(topic, msg)
                            except Exception as e:
                                logger.error("subscriber.handler_error",
                                             topic=topic, error=str(e))
                        matched = True

                if not matched:
                    logger.debug("subscriber.no_handler", topic=topic)

            except zmq.ZMQError as e:
                if self._running:
                    logger.error("subscriber.zmq_error", error=str(e))
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("subscriber.error", error=str(e))
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Close the SUB socket."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("subscriber.stopped")


class HeartbeatMonitor:
    """
    Tracks heartbeats from connected nodes.
    Reports nodes that have gone silent.
    """

    def __init__(self, timeout_s: float = 10.0):
        self._timeout_s = timeout_s
        self._last_seen: dict[str, float] = {}
        self._alive: set[str] = set()
        self._callbacks: list[Callable[[str, bool], Coroutine]] = []

    def register_callback(self, callback: Callable[[str, bool], Coroutine]) -> None:
        """Register callback(node_id, is_alive) for heartbeat state changes."""
        self._callbacks.append(callback)

    def beat(self, node_id: str) -> None:
        """Record a heartbeat from a node."""
        self._last_seen[node_id] = time.time()
        if node_id not in self._alive:
            self._alive.add(node_id)
            logger.info("heartbeat.node_online", node_id=node_id)

    async def check(self) -> list[str]:
        """
        Check for timed-out nodes. Returns list of lost node IDs.
        """
        now = time.time()
        lost = []
        for node_id, last in list(self._last_seen.items()):
            if now - last > self._timeout_s and node_id in self._alive:
                self._alive.discard(node_id)
                lost.append(node_id)
                logger.warning("heartbeat.node_lost", node_id=node_id,
                               last_seen_s=round(now - last, 1))
                for cb in self._callbacks:
                    try:
                        await cb(node_id, False)
                    except Exception as e:
                        logger.error("heartbeat.callback_error", error=str(e))
        return lost

    def get_alive_nodes(self) -> set[str]:
        """Get set of currently alive node IDs."""
        return self._alive.copy()

    def is_alive(self, node_id: str) -> bool:
        """Check if a specific node is alive."""
        return node_id in self._alive


class MessageBus:
    """
    High-level message bus combining Publisher + Subscriber + Heartbeat.
    
    Usage for Ground Station (bind both):
        bus = MessageBus(role="ground", pub_addr="tcp://*:5555",
                         sub_addr="tcp://*:5556", pub_bind=True, sub_bind=True)
        
    Usage for Drone (connect both):
        bus = MessageBus(role="drone", pub_addr="tcp://ground:5556",
                         sub_addr="tcp://ground:5555", pub_bind=False, sub_bind=False)
    """

    def __init__(
        self,
        role: str,
        pub_addr: str,
        sub_addr: str,
        node_id: str = "",
        topics: Optional[list[str]] = None,
        pub_bind: bool = True,
        sub_bind: bool = False,
        heartbeat_interval_s: float = 2.0,
        heartbeat_timeout_s: float = 10.0,
    ):
        self.role = role
        self.node_id = node_id
        self.publisher = Publisher(pub_addr, bind=pub_bind)
        self.subscriber = Subscriber(sub_addr, topics, bind=sub_bind)
        self.heartbeat = HeartbeatMonitor(heartbeat_timeout_s)
        self._heartbeat_interval = heartbeat_interval_s
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start publisher, subscriber, and heartbeat."""
        await self.publisher.start()
        await self.subscriber.start()
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("message_bus.started", role=self.role, node_id=self.node_id)

    async def publish(self, topic: str, msg: BaseModel) -> None:
        """Publish a message."""
        await self.publisher.publish(topic, msg)

    def on_message(self, topic: str, handler: Callable) -> None:
        """Register a message handler."""
        self.subscriber.on_message(topic, handler)

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeats and check for lost nodes."""
        from proto.messages import DroneTelemetry, GeoPoint, DroneState

        while self._running:
            try:
                # Send our heartbeat
                if self.node_id:
                    heartbeat = DroneTelemetry(
                        source_id=self.node_id,
                        drone_id=self.node_id,
                        position=GeoPoint(lat=0, lon=0, alt=0),
                        heading=0,
                        speed=0,
                        battery_pct=100,
                        state=DroneState.IDLE,
                    )
                    await self.publish(f"heartbeat/{self.node_id}", heartbeat)

                # Check for lost nodes
                await self.heartbeat.check()

            except Exception as e:
                logger.error("heartbeat_loop.error", error=str(e))

            await asyncio.sleep(self._heartbeat_interval)

    async def stop(self) -> None:
        """Stop everything."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.subscriber.stop()
        await self.publisher.stop()
        logger.info("message_bus.stopped", role=self.role)
