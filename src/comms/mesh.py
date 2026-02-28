"""
VayuSwarm — Drone-to-Drone Mesh Network

Peer-to-peer mesh for low-latency swarm coordination.
Each drone can send/receive messages directly to/from neighbors.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Coroutine, Optional

import zmq
import zmq.asyncio
import structlog
from pydantic import BaseModel

from src.comms.serializer import MessageSerializer, build_message_registry

logger = structlog.get_logger(__name__)


class MeshPeer:
    """Represents a known peer drone in the mesh."""

    def __init__(self, peer_id: str, address: str):
        self.peer_id = peer_id
        self.address = address
        self.last_seen: float = 0.0
        self.connected: bool = False


class MeshNetwork:
    """
    Drone-to-drone mesh network using ZeroMQ ROUTER/DEALER pattern.
    
    Each drone runs a ROUTER socket (server) and connects DEALER sockets
    to known peers. This allows bidirectional, non-blocking communication.
    """

    def __init__(
        self,
        drone_id: str,
        bind_port: int,
        host: str = "0.0.0.0",
    ):
        self._drone_id = drone_id
        self._bind_addr = f"tcp://{host}:{bind_port}"
        self._ctx = zmq.asyncio.Context()
        self._router: Optional[zmq.asyncio.Socket] = None
        self._peers: dict[str, MeshPeer] = {}
        self._peer_sockets: dict[str, zmq.asyncio.Socket] = {}
        self._handlers: list[Callable[[str, BaseModel], Coroutine]] = []
        self._registry = build_message_registry()
        self._running = False
        self._recv_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the mesh network — bind ROUTER and start receiving."""
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt_string(zmq.IDENTITY, self._drone_id)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.bind(self._bind_addr)
        self._running = True
        self._recv_task = asyncio.create_task(self._receive_loop())
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info("mesh.started", drone_id=self._drone_id, address=self._bind_addr)

    def add_peer(self, peer_id: str, address: str) -> None:
        """Add a known peer to the mesh."""
        if peer_id == self._drone_id:
            return
        self._peers[peer_id] = MeshPeer(peer_id, address)
        logger.info("mesh.peer_added", peer_id=peer_id, address=address)

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the mesh."""
        if peer_id in self._peers:
            del self._peers[peer_id]
        if peer_id in self._peer_sockets:
            self._peer_sockets[peer_id].close()
            del self._peer_sockets[peer_id]
        logger.info("mesh.peer_removed", peer_id=peer_id)

    async def _ensure_peer_connected(self, peer_id: str) -> Optional[zmq.asyncio.Socket]:
        """Ensure a DEALER socket is connected to the peer."""
        if peer_id in self._peer_sockets:
            return self._peer_sockets[peer_id]

        peer = self._peers.get(peer_id)
        if not peer:
            return None

        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self._drone_id)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(peer.address)
        self._peer_sockets[peer_id] = sock
        peer.connected = True
        logger.info("mesh.peer_connected", peer_id=peer_id, address=peer.address)
        await asyncio.sleep(0.1)
        return sock

    async def send_to_peer(self, peer_id: str, msg: BaseModel) -> bool:
        """Send a message directly to a specific peer drone."""
        sock = await self._ensure_peer_connected(peer_id)
        if not sock:
            logger.warning("mesh.peer_not_found", peer_id=peer_id)
            return False

        try:
            data = MessageSerializer.serialize_simple(msg)
            await sock.send(data)
            logger.debug("mesh.sent", to=peer_id, type=type(msg).__name__)
            return True
        except zmq.ZMQError as e:
            logger.error("mesh.send_error", peer_id=peer_id, error=str(e))
            return False

    async def broadcast(self, msg: BaseModel, exclude: Optional[set[str]] = None) -> int:
        """Broadcast a message to all peers. Returns number of successful sends."""
        exclude = exclude or set()
        sent = 0
        for peer_id in self._peers:
            if peer_id not in exclude:
                if await self.send_to_peer(peer_id, msg):
                    sent += 1
        return sent

    def on_message(self, handler: Callable[[str, BaseModel], Coroutine]) -> None:
        """Register a handler for incoming mesh messages."""
        self._handlers.append(handler)

    async def _receive_loop(self) -> None:
        """Receive messages on the ROUTER socket."""
        while self._running:
            try:
                frames = await self._router.recv_multipart()
                if len(frames) < 2:
                    continue

                sender_id = frames[0].decode("utf-8")
                data = frames[1]

                msg = MessageSerializer.deserialize_simple(data, self._registry)

                # Update peer last seen
                if sender_id in self._peers:
                    self._peers[sender_id].last_seen = time.time()

                # Dispatch to handlers
                for handler in self._handlers:
                    try:
                        await handler(sender_id, msg)
                    except Exception as e:
                        logger.error("mesh.handler_error", sender=sender_id, error=str(e))

            except zmq.ZMQError as e:
                if self._running:
                    logger.error("mesh.recv_error", error=str(e))
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("mesh.error", error=str(e))
                await asyncio.sleep(0.1)

    async def _discovery_loop(self) -> None:
        """Periodically check peer health and discover new peers."""
        while self._running:
            now = time.time()
            for peer_id, peer in list(self._peers.items()):
                if peer.connected and peer.last_seen > 0:
                    if now - peer.last_seen > 30.0:
                        logger.warning("mesh.peer_timeout", peer_id=peer_id)
            await asyncio.sleep(5.0)

    def get_peers(self) -> dict[str, MeshPeer]:
        """Get all known peers."""
        return self._peers.copy()

    def get_connected_peers(self) -> list[str]:
        """Get list of connected peer IDs."""
        return [pid for pid, p in self._peers.items() if p.connected]

    async def stop(self) -> None:
        """Stop the mesh network."""
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        for sock in self._peer_sockets.values():
            sock.close()
        self._peer_sockets.clear()
        if self._router:
            self._router.close()
        logger.info("mesh.stopped", drone_id=self._drone_id)
