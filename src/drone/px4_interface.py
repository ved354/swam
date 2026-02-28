"""
VayuSwarm — PX4/MAVLink Flight Control Interface

Interfaces with PX4 autopilot via MAVLink protocol.
Supports both real hardware and SITL (Software-In-The-Loop) simulation.

Commands: arm, takeoff, land, goto, set velocity, RTL, mode switching.
Telemetry: GPS, attitude, battery, airspeed.

PX4 SITL Usage:
    1. Start PX4 SITL:  make px4_sitl gazebo  (or jmavsim)
    2. SITL listens on udp:127.0.0.1:14540 by default
    3. Set simulation: false in config and run launch_drone.py
"""

from __future__ import annotations

import asyncio
import functools
import math
import time
from typing import Any, Dict, Optional

import structlog

from proto.messages import DroneTelemetry, DroneState, GeoPoint

logger = structlog.get_logger(__name__)

# ─── PX4 Custom Mode Constants ──────────────────────────────────────────────────
# PX4 uses MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (bit 0 of base_mode = 1)
# The custom_mode is a 32-bit value encoding main_mode (upper 8 bits) and
# sub_mode (middle 8 bits).
#
# Main modes:
PX4_MAIN_MODE_MANUAL     = 1
PX4_MAIN_MODE_ALTCTL     = 2
PX4_MAIN_MODE_POSCTL     = 3
PX4_MAIN_MODE_AUTO       = 4
PX4_MAIN_MODE_ACRO       = 5
PX4_MAIN_MODE_OFFBOARD   = 6
PX4_MAIN_MODE_STABILIZED = 7
PX4_MAIN_MODE_RATTITUDE  = 8

# Auto sub-modes:
PX4_AUTO_SUB_MODE_READY     = 1
PX4_AUTO_SUB_MODE_TAKEOFF   = 2
PX4_AUTO_SUB_MODE_LOITER    = 3
PX4_AUTO_SUB_MODE_MISSION   = 4
PX4_AUTO_SUB_MODE_RTL       = 5
PX4_AUTO_SUB_MODE_LAND      = 6
PX4_AUTO_SUB_MODE_FOLLOW    = 8

# Map friendly names → (main_mode, sub_mode)
PX4_MODE_MAP: Dict[str, tuple] = {
    "MANUAL":     (PX4_MAIN_MODE_MANUAL, 0),
    "ALTCTL":     (PX4_MAIN_MODE_ALTCTL, 0),
    "POSCTL":     (PX4_MAIN_MODE_POSCTL, 0),
    "STABILIZED": (PX4_MAIN_MODE_STABILIZED, 0),
    "OFFBOARD":   (PX4_MAIN_MODE_OFFBOARD, 0),
    "ACRO":       (PX4_MAIN_MODE_ACRO, 0),
    "AUTO.READY":   (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_READY),
    "AUTO.TAKEOFF": (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_TAKEOFF),
    "AUTO.LOITER":  (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_LOITER),
    "AUTO.MISSION": (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_MISSION),
    "AUTO.RTL":     (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_RTL),
    "AUTO.LAND":    (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_LAND),
    "AUTO.FOLLOW":  (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_FOLLOW),
    # Convenience aliases used by VayuSwarm state machine
    "GUIDED":       (PX4_MAIN_MODE_OFFBOARD, 0),
    "LOITER":       (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_LOITER),
    "RTL":          (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_RTL),
    "LAND":         (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_LAND),
    "TAKEOFF":      (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_TAKEOFF),
    "MISSION":      (PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_MISSION),
}


def _px4_custom_mode(main_mode: int, sub_mode: int = 0) -> int:
    """Encode PX4 main_mode + sub_mode into a 32-bit custom_mode value."""
    return (main_mode << 16) | (sub_mode << 24)


class PX4Interface:
    """
    MAVLink interface to PX4 autopilot.

    In simulation mode (default), provides mock telemetry and
    simulates flight dynamics without a real PX4 connection.

    In real/SITL mode, communicates with PX4 over MAVLink UDP.
    Supports PX4 custom modes, OFFBOARD control, and AUTO sub-modes.
    """

    def __init__(
        self,
        drone_id: str,
        connection_string: str = "udp:127.0.0.1:14540",
        source_system: int = 255,
        source_component: int = 0,
        simulation: bool = True,
    ):
        self._drone_id = drone_id
        self._connection_string = connection_string
        self._source_system = source_system
        self._source_component = source_component
        self._simulation = simulation
        self._mav: Any = None
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Background tasks for real MAVLink
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._telemetry_task: Optional[asyncio.Task] = None
        self._stopping = False

        # Cached real telemetry (updated by background receiver)
        self._cached_position = GeoPoint(lat=0.0, lon=0.0, alt=0.0)
        self._cached_heading = 0.0
        self._cached_speed = 0.0
        self._cached_vspeed = 0.0
        self._cached_battery_pct = 100.0
        self._cached_battery_voltage = 16.8
        self._cached_gps_fix = 0
        self._cached_satellites = 0
        self._cached_armed = False
        self._cached_mode = "UNKNOWN"

        # Simulated state
        self._sim_lat = 17.385044
        self._sim_lon = 78.486671
        self._sim_alt = 0.0
        self._sim_heading = 0.0
        self._sim_speed = 0.0
        self._sim_vspeed = 0.0
        self._sim_battery = 100.0
        self._sim_armed = False
        self._sim_mode = "STABILIZE"
        self._sim_target: Optional[GeoPoint] = None
        self._sim_flying = False
        self._last_update = time.time()

    # ─── Connection ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to PX4 autopilot (or start simulated mode)."""
        if self._simulation:
            self._connected = True
            logger.info("px4.connected_simulation", drone_id=self._drone_id)
            return

        self._loop = asyncio.get_running_loop()

        try:
            from pymavlink import mavutil

            # Run blocking mavlink_connection in executor
            self._mav = await self._loop.run_in_executor(
                None,
                functools.partial(
                    mavutil.mavlink_connection,
                    self._connection_string,
                    source_system=self._source_system,
                    source_component=self._source_component,
                ),
            )

            # Wait for heartbeat (blocking → executor)
            heartbeat = await self._loop.run_in_executor(
                None,
                functools.partial(self._mav.wait_heartbeat, timeout=15),
            )
            if heartbeat is None:
                raise TimeoutError("No heartbeat from PX4 within 15 s")

            self._connected = True
            logger.info(
                "px4.connected",
                drone_id=self._drone_id,
                target_system=self._mav.target_system,
                target_component=self._mav.target_component,
            )

            # Request data streams (PX4 uses MAVLink 2 by default but we
            # request explicitly for compatibility)
            self._request_data_streams()

            # Start background heartbeat + telemetry receiver
            self._stopping = False
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._telemetry_task = asyncio.create_task(self._telemetry_receiver())

        except Exception as e:
            logger.error("px4.connect_error", error=str(e))
            # Fall back to simulation
            self._simulation = True
            self._connected = True
            logger.info("px4.fallback_simulation", drone_id=self._drone_id)

    def _request_data_streams(self) -> None:
        """Request telemetry data streams from PX4."""
        if not self._mav:
            return
        try:
            # Request all data streams at 10 Hz
            self._mav.mav.request_data_stream_send(
                self._mav.target_system,
                self._mav.target_component,
                0,   # MAV_DATA_STREAM_ALL
                10,  # rate Hz
                1,   # start
            )
        except Exception:
            pass  # Non-critical

    async def _heartbeat_loop(self) -> None:
        """Send GCS heartbeats to PX4 at 1 Hz (required to keep link alive)."""
        try:
            from pymavlink import mavutil
        except ImportError:
            return

        while not self._stopping and self._mav:
            try:
                self._mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
            except Exception as e:
                logger.debug("px4.heartbeat_send_error", error=str(e))
            await asyncio.sleep(1.0)

    async def _telemetry_receiver(self) -> None:
        """Background loop that polls MAVLink messages and caches telemetry."""
        while not self._stopping and self._mav:
            try:
                # Run blocking recv_match in executor with short timeout
                msg = await self._loop.run_in_executor(
                    None,
                    functools.partial(
                        self._mav.recv_match,
                        blocking=True,
                        timeout=0.1,
                    ),
                )
                if msg is None:
                    continue

                msg_type = msg.get_type()
                self._process_mavlink_message(msg_type, msg)

            except Exception as e:
                if not self._stopping:
                    logger.debug("px4.recv_error", error=str(e))
                await asyncio.sleep(0.05)

    def _process_mavlink_message(self, msg_type: str, msg: Any) -> None:
        """Process a single MAVLink message and update cached state."""
        if msg_type == "GLOBAL_POSITION_INT":
            self._cached_position = GeoPoint(
                lat=msg.lat / 1e7,
                lon=msg.lon / 1e7,
                alt=msg.relative_alt / 1000.0,
            )
            self._cached_heading = msg.hdg / 100.0 if msg.hdg != 65535 else self._cached_heading
            self._cached_vspeed = msg.vz / 100.0  # cm/s → m/s, NED

        elif msg_type == "VFR_HUD":
            self._cached_speed = msg.groundspeed
            self._cached_heading = msg.heading

        elif msg_type == "SYS_STATUS":
            self._cached_battery_pct = float(
                msg.battery_remaining if msg.battery_remaining >= 0 else 0
            )
            self._cached_battery_voltage = msg.voltage_battery / 1000.0  # mV → V

        elif msg_type == "GPS_RAW_INT":
            self._cached_gps_fix = msg.fix_type
            self._cached_satellites = msg.satellites_visible

        elif msg_type == "HEARTBEAT":
            # Decode armed state from base_mode
            armed = bool(msg.base_mode & 0x80)  # MAV_MODE_FLAG_SAFETY_ARMED
            self._cached_armed = armed
            # Decode PX4 custom mode to friendly name
            self._cached_mode = self._decode_px4_mode(msg.custom_mode)

    @staticmethod
    def _decode_px4_mode(custom_mode: int) -> str:
        """Decode PX4 custom_mode bitfield into a human-readable string."""
        main_mode = (custom_mode >> 16) & 0xFF
        sub_mode = (custom_mode >> 24) & 0xFF

        # Reverse lookup
        for name, (mm, sm) in PX4_MODE_MAP.items():
            if mm == main_mode and sm == sub_mode and "." in name:
                return name
            if mm == main_mode and sm == 0 and sub_mode == 0 and "." not in name:
                return name

        return f"CUSTOM({main_mode},{sub_mode})"

    # ─── Commands ────────────────────────────────────────────────────────────

    async def arm(self) -> bool:
        """Arm the drone motors."""
        if self._simulation:
            self._sim_armed = True
            logger.info("px4.armed", drone_id=self._drone_id, simulation=True)
            return True

        try:
            # MAV_CMD_COMPONENT_ARM_DISARM (command 400), param1=1 → arm
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                400,  # MAV_CMD_COMPONENT_ARM_DISARM
                0,    # confirmation
                1,    # param1: 1=arm
                0, 0, 0, 0, 0, 0,
            )
            # Wait for ACK
            ack = await self._wait_command_ack(400, timeout=5.0)
            if ack:
                logger.info("px4.armed", drone_id=self._drone_id)
                return True
            else:
                logger.warning("px4.arm_no_ack", drone_id=self._drone_id)
                return False
        except Exception as e:
            logger.error("px4.arm_error", error=str(e))
            return False

    async def disarm(self) -> bool:
        """Disarm the drone motors."""
        if self._simulation:
            self._sim_armed = False
            self._sim_flying = False
            logger.info("px4.disarmed", drone_id=self._drone_id)
            return True

        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                400,  # MAV_CMD_COMPONENT_ARM_DISARM
                0,
                0,    # param1: 0=disarm
                0, 0, 0, 0, 0, 0,
            )
            ack = await self._wait_command_ack(400, timeout=5.0)
            if ack:
                logger.info("px4.disarmed", drone_id=self._drone_id)
            return ack
        except Exception as e:
            logger.error("px4.disarm_error", error=str(e))
            return False

    async def takeoff(self, altitude_m: float = 50.0) -> bool:
        """Command takeoff to specified altitude."""
        if self._simulation:
            self._sim_armed = True
            self._sim_flying = True
            self._sim_mode = "GUIDED"
            self._sim_target = GeoPoint(lat=self._sim_lat, lon=self._sim_lon, alt=altitude_m)
            logger.info("px4.takeoff", drone_id=self._drone_id, altitude=altitude_m)
            return True

        try:
            # Set AUTO.TAKEOFF mode, then send takeoff command
            await self._set_px4_mode("TAKEOFF")
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                22,  # MAV_CMD_NAV_TAKEOFF
                0,   # confirmation
                0,   # param1: min pitch
                0,   # param2: empty
                0,   # param3: empty
                0,   # param4: yaw angle (NaN for current)
                0,   # param5: lat (0 = current)
                0,   # param6: lon (0 = current)
                altitude_m,  # param7: altitude
            )
            ack = await self._wait_command_ack(22, timeout=10.0)
            logger.info("px4.takeoff", drone_id=self._drone_id,
                        altitude=altitude_m, acked=ack)
            return ack
        except Exception as e:
            logger.error("px4.takeoff_error", error=str(e))
            return False

    async def land(self) -> bool:
        """Command landing at current position."""
        if self._simulation:
            self._sim_target = GeoPoint(lat=self._sim_lat, lon=self._sim_lon, alt=0)
            self._sim_mode = "LAND"
            logger.info("px4.landing", drone_id=self._drone_id)
            return True

        try:
            ok = await self._set_px4_mode("LAND")
            logger.info("px4.landing", drone_id=self._drone_id, mode_set=ok)
            return ok
        except Exception as e:
            logger.error("px4.land_error", error=str(e))
            return False

    async def goto(self, waypoint: GeoPoint, speed_ms: float = 10.0) -> bool:
        """Navigate to a waypoint using OFFBOARD position control."""
        if self._simulation:
            self._sim_target = waypoint
            self._sim_speed = speed_ms
            self._sim_mode = "GUIDED"
            logger.info("px4.goto",
                        drone_id=self._drone_id,
                        lat=waypoint.lat, lon=waypoint.lon, alt=waypoint.alt)
            return True

        try:
            # Switch to OFFBOARD mode (PX4 requires setpoint stream first)
            self._mav.mav.set_position_target_global_int_send(
                0,  # time_boot_ms
                self._mav.target_system,
                self._mav.target_component,
                6,  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
                0b0000_1111_1111_1000,  # type_mask: position only
                int(waypoint.lat * 1e7),
                int(waypoint.lon * 1e7),
                waypoint.alt,
                0, 0, 0,  # velocity
                0, 0, 0,  # acceleration
                0, 0,     # yaw, yaw_rate
            )
            await self._set_px4_mode("OFFBOARD")
            logger.info("px4.goto",
                        drone_id=self._drone_id,
                        lat=waypoint.lat, lon=waypoint.lon, alt=waypoint.alt)
            return True
        except Exception as e:
            logger.error("px4.goto_error", error=str(e))
            return False

    async def set_velocity(self, vx: float, vy: float, vz: float) -> bool:
        """Set velocity in NED frame (m/s). Requires OFFBOARD mode."""
        if self._simulation:
            self._sim_speed = math.sqrt(vx * vx + vy * vy)
            self._sim_vspeed = -vz  # NED to up
            self._sim_heading = math.degrees(math.atan2(vy, vx)) % 360
            return True

        try:
            self._mav.mav.set_position_target_local_ned_send(
                0,
                self._mav.target_system,
                self._mav.target_component,
                1,  # MAV_FRAME_LOCAL_NED
                0b0000_1111_1100_0111,  # type_mask: velocity only
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, 0,
            )
            return True
        except Exception as e:
            logger.error("px4.velocity_error", error=str(e))
            return False

    async def return_to_launch(self) -> bool:
        """Command Return to Launch (RTL)."""
        if self._simulation:
            self._sim_target = GeoPoint(lat=17.385044, lon=78.486671, alt=50)
            self._sim_mode = "RTL"
            logger.info("px4.rtl", drone_id=self._drone_id)
            return True

        try:
            ok = await self._set_px4_mode("RTL")
            logger.info("px4.rtl", drone_id=self._drone_id, mode_set=ok)
            return ok
        except Exception as e:
            logger.error("px4.rtl_error", error=str(e))
            return False

    async def hold_position(self) -> bool:
        """Hold current position (loiter)."""
        if self._simulation:
            self._sim_target = None
            self._sim_speed = 0
            self._sim_mode = "LOITER"
            return True

        try:
            ok = await self._set_px4_mode("LOITER")
            return ok
        except Exception as e:
            logger.error("px4.hold_error", error=str(e))
            return False

    # ─── PX4 Mode Setting ───────────────────────────────────────────────────

    async def _set_px4_mode(self, mode_name: str) -> bool:
        """
        Set PX4 flight mode using custom_mode encoding.

        PX4 does NOT use the simple mode_mapping that ArduPilot provides.
        Instead we encode (main_mode, sub_mode) into a 32-bit custom_mode
        and send SET_MODE with MAV_MODE_FLAG_CUSTOM_MODE_ENABLED.
        """
        if not self._mav:
            return False

        entry = PX4_MODE_MAP.get(mode_name.upper())
        if entry is None:
            logger.warning("px4.unknown_mode", mode=mode_name)
            return False

        main_mode, sub_mode = entry
        custom = _px4_custom_mode(main_mode, sub_mode)
        base_mode = 1  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

        # If armed, preserve the armed flag
        if self._cached_armed:
            base_mode |= 0x80  # MAV_MODE_FLAG_SAFETY_ARMED

        try:
            self._mav.mav.set_mode_send(
                self._mav.target_system,
                base_mode,
                custom,
            )
            # Wait briefly for mode change ACK
            ack = await self._wait_command_ack(176, timeout=3.0)  # 176 = SET_MODE
            return ack
        except Exception as e:
            logger.error("px4.set_mode_error", mode=mode_name, error=str(e))
            return False

    async def _wait_command_ack(self, command_id: int, timeout: float = 5.0) -> bool:
        """Wait for COMMAND_ACK from PX4 for a given command."""
        if not self._mav or not self._loop:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = await self._loop.run_in_executor(
                    None,
                    functools.partial(
                        self._mav.recv_match,
                        type="COMMAND_ACK",
                        blocking=True,
                        timeout=0.5,
                    ),
                )
                if msg and msg.command == command_id:
                    # result == 0 means MAV_RESULT_ACCEPTED
                    return msg.result == 0
            except Exception:
                pass
        return False

    # ─── Telemetry ───────────────────────────────────────────────────────────

    async def get_telemetry(self) -> DroneTelemetry:
        """Get current telemetry data."""
        if self._simulation:
            self._update_simulation()
            return DroneTelemetry(
                source_id=self._drone_id,
                drone_id=self._drone_id,
                position=GeoPoint(
                    lat=self._sim_lat,
                    lon=self._sim_lon,
                    alt=self._sim_alt,
                ),
                heading=self._sim_heading,
                speed=self._sim_speed,
                vertical_speed=self._sim_vspeed,
                battery_pct=self._sim_battery,
                battery_voltage=self._sim_battery * 0.168,  # Approximate
                gps_fix=3,
                satellites=12,
                armed=self._sim_armed,
                state=self._get_sim_state(),
                mode=self._sim_mode,
                signal_strength=95.0,
            )

        # Real MAVLink telemetry — read from cached values
        return DroneTelemetry(
            source_id=self._drone_id,
            drone_id=self._drone_id,
            position=self._cached_position,
            heading=self._cached_heading,
            speed=self._cached_speed,
            vertical_speed=self._cached_vspeed,
            battery_pct=self._cached_battery_pct,
            battery_voltage=self._cached_battery_voltage,
            gps_fix=self._cached_gps_fix,
            satellites=self._cached_satellites,
            armed=self._cached_armed,
            mode=self._cached_mode,
            state=self._mavlink_to_drone_state(),
        )

    def _mavlink_to_drone_state(self) -> DroneState:
        """Map current PX4 mode + armed state to VayuSwarm DroneState."""
        mode = self._cached_mode.upper()
        if not self._cached_armed:
            return DroneState.IDLE
        if "RTL" in mode:
            return DroneState.RTL
        if "LAND" in mode:
            return DroneState.LAND
        if "TAKEOFF" in mode:
            return DroneState.TAKEOFF
        if self._cached_position.alt < 3.0:
            return DroneState.TAKEOFF
        return DroneState.PATROL

    # ─── Simulation ─────────────────────────────────────────────────────────

    def _update_simulation(self) -> None:
        """Update simulated position and state."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        if dt > 1.0:
            dt = 0.1  # Cap delta time

        # Battery drain
        if self._sim_flying:
            self._sim_battery -= dt * 0.05  # ~0.05%/s
        self._sim_battery = max(0, self._sim_battery)

        # Move toward target
        if self._sim_target and self._sim_flying:
            dlat = self._sim_target.lat - self._sim_lat
            dlon = self._sim_target.lon - self._sim_lon
            dalt = self._sim_target.alt - self._sim_alt

            dist_h = math.sqrt(dlat ** 2 + dlon ** 2) * 111319.9  # Approx meters
            dist_v = abs(dalt)

            if dist_h > 1.0:
                speed = min(self._sim_speed or 10.0, dist_h)
                ratio = (speed * dt) / max(dist_h, 0.001)
                ratio = min(ratio, 1.0)
                self._sim_lat += dlat * ratio
                self._sim_lon += dlon * ratio
                self._sim_heading = math.degrees(math.atan2(dlon, dlat)) % 360

            if dist_v > 0.5:
                v_speed = min(3.0, dist_v)
                ratio = (v_speed * dt) / max(dist_v, 0.001)
                ratio = min(ratio, 1.0)
                self._sim_alt += dalt * ratio
                self._sim_vspeed = dalt * ratio / max(dt, 0.001)

            # Check if arrived
            if dist_h < 2.0 and dist_v < 1.0:
                if self._sim_mode == "LAND" and self._sim_alt <= 1.0:
                    self._sim_flying = False
                    self._sim_armed = False
                    self._sim_alt = 0.0

    def _get_sim_state(self) -> DroneState:
        """Get simulated drone state."""
        if not self._sim_armed:
            return DroneState.IDLE
        if self._sim_mode == "RTL":
            return DroneState.RTL
        if self._sim_mode == "LAND":
            return DroneState.LAND
        if self._sim_alt < 5 and self._sim_flying:
            return DroneState.TAKEOFF
        if self._sim_flying:
            return DroneState.PATROL
        return DroneState.IDLE

    # ─── Utilities ───────────────────────────────────────────────────────────

    def set_home_position(self, lat: float, lon: float, alt: float = 0.0) -> None:
        """Set simulation home position."""
        self._sim_lat = lat
        self._sim_lon = lon
        self._sim_alt = alt

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulation(self) -> bool:
        return self._simulation

    async def disconnect(self) -> None:
        """Disconnect from PX4 and stop background tasks."""
        self._stopping = True
        self._connected = False

        # Cancel background tasks
        for task in (self._heartbeat_task, self._telemetry_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if self._mav:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

        logger.info("px4.disconnected", drone_id=self._drone_id)
