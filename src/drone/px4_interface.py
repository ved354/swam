"""
VayuSwarm — PX4/MAVLink Flight Control Interface

Interfaces with PX4 autopilot via MAVLink protocol.
Supports both real hardware and SITL (Software-In-The-Loop) simulation.

Commands: arm, takeoff, land, goto, set velocity, RTL, mode switching.
Telemetry: GPS, attitude, battery, airspeed.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Optional

import structlog

from proto.messages import DroneTelemetry, DroneState, GeoPoint

logger = structlog.get_logger(__name__)


class PX4Interface:
    """
    MAVLink interface to PX4 autopilot.
    
    In simulation mode (default), provides mock telemetry and
    simulates flight dynamics without a real PX4 connection.
    """

    def __init__(
        self,
        drone_id: str,
        connection_string: str = "udp:127.0.0.1:14540",
        source_system: int = 1,
        source_component: int = 1,
        simulation: bool = True,
    ):
        self._drone_id = drone_id
        self._connection_string = connection_string
        self._source_system = source_system
        self._source_component = source_component
        self._simulation = simulation
        self._mav = None
        self._connected = False

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

    async def connect(self) -> None:
        """Connect to PX4 autopilot."""
        if self._simulation:
            self._connected = True
            logger.info("px4.connected_simulation", drone_id=self._drone_id)
            return

        try:
            from pymavlink import mavutil
            self._mav = mavutil.mavlink_connection(
                self._connection_string,
                source_system=self._source_system,
                source_component=self._source_component,
            )
            self._mav.wait_heartbeat(timeout=10)
            self._connected = True
            logger.info("px4.connected",
                        drone_id=self._drone_id,
                        target_system=self._mav.target_system)
        except Exception as e:
            logger.error("px4.connect_error", error=str(e))
            # Fall back to simulation
            self._simulation = True
            self._connected = True
            logger.info("px4.fallback_simulation", drone_id=self._drone_id)

    async def arm(self) -> bool:
        """Arm the drone motors."""
        if self._simulation:
            self._sim_armed = True
            logger.info("px4.armed", drone_id=self._drone_id, simulation=True)
            return True

        try:
            self._mav.arducopter_arm()
            self._mav.motors_armed_wait()
            logger.info("px4.armed", drone_id=self._drone_id)
            return True
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
            self._mav.arducopter_disarm()
            return True
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
            self._set_mode("GUIDED")
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                22,  # MAV_CMD_NAV_TAKEOFF
                0, 0, 0, 0, 0, 0, 0, altitude_m,
            )
            return True
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
            self._set_mode("LAND")
            return True
        except Exception as e:
            logger.error("px4.land_error", error=str(e))
            return False

    async def goto(self, waypoint: GeoPoint, speed_ms: float = 10.0) -> bool:
        """Navigate to a waypoint."""
        if self._simulation:
            self._sim_target = waypoint
            self._sim_speed = speed_ms
            self._sim_mode = "GUIDED"
            logger.info("px4.goto",
                        drone_id=self._drone_id,
                        lat=waypoint.lat, lon=waypoint.lon, alt=waypoint.alt)
            return True

        try:
            self._set_mode("GUIDED")
            self._mav.mav.set_position_target_global_int_send(
                0,  # time_boot_ms
                self._mav.target_system,
                self._mav.target_component,
                6,  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
                0b0000111111111000,  # type_mask (position only)
                int(waypoint.lat * 1e7),
                int(waypoint.lon * 1e7),
                waypoint.alt,
                0, 0, 0,  # velocity
                0, 0, 0,  # acceleration
                0, 0,     # yaw, yaw_rate
            )
            return True
        except Exception as e:
            logger.error("px4.goto_error", error=str(e))
            return False

    async def set_velocity(self, vx: float, vy: float, vz: float) -> bool:
        """Set velocity in NED frame (m/s)."""
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
                8,  # MAV_FRAME_BODY_NED
                0b0000111111000111,  # type_mask (velocity only)
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
            self._set_mode("RTL")
            return True
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
            self._set_mode("LOITER")
            return True
        except Exception as e:
            logger.error("px4.hold_error", error=str(e))
            return False

    def _set_mode(self, mode: str) -> None:
        """Set flight mode."""
        if self._mav:
            mode_id = self._mav.mode_mapping().get(mode, 0)
            self._mav.set_mode(mode_id)

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

        # Real MAVLink telemetry
        try:
            msg = self._mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            bat = self._mav.recv_match(type="SYS_STATUS", blocking=False)
            att = self._mav.recv_match(type="ATTITUDE", blocking=False)

            lat = msg.lat / 1e7 if msg else 0.0
            lon = msg.lon / 1e7 if msg else 0.0
            alt = msg.relative_alt / 1000.0 if msg else 0.0
            hdg = att.yaw * 180 / math.pi % 360 if att else 0.0
            bat_pct = bat.battery_remaining if bat else 0

            return DroneTelemetry(
                source_id=self._drone_id,
                drone_id=self._drone_id,
                position=GeoPoint(lat=lat, lon=lon, alt=alt),
                heading=hdg,
                speed=0.0,
                battery_pct=float(bat_pct),
                armed=self._mav.motors_armed(),
                mode=self._mav.flightmode,
            )
        except Exception as e:
            logger.error("px4.telemetry_error", error=str(e))
            return DroneTelemetry(
                source_id=self._drone_id,
                drone_id=self._drone_id,
                position=GeoPoint(lat=0, lon=0, alt=0),
                heading=0,
                speed=0,
                battery_pct=0,
                state=DroneState.OFFLINE,
            )

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
        """Disconnect from PX4."""
        self._connected = False
        if self._mav:
            self._mav.close()
        logger.info("px4.disconnected", drone_id=self._drone_id)
