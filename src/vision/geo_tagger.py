"""
VayuSwarm — GPS Geo-Tagging Module

Converts pixel bounding-box centres into GPS (lat/lon/alt) coordinates.
Uses a simplified pinhole camera model assuming a nadir (straight-down)
or tilted camera mounted on the drone.

Maths (nadir camera, flat-earth approximation valid for altitudes <1 km):
  cx_norm, cy_norm  = bbox centre in [0, 1] normalised pixel coords
  H   = altitude AGL in metres
  HFOV, VFOV = camera horizontal / vertical field-of-view (degrees)

  ground_east_m  = (cx_norm - 0.5) * 2 * H * tan(HFOV/2)
  ground_north_m = (0.5 - cy_norm) * 2 * H * tan(VFOV/2)
                   (note: image y+ is downward, so flip sign for north)

If the drone has a compass heading θ, the local East/North offsets are
rotated into geographic North/East:
  east_geo  =  ground_east_m * cos(θ) + ground_north_m * sin(θ)
  north_geo = -ground_east_m * sin(θ) + ground_north_m * cos(θ)

Then:
  delta_lat = north_geo / 111_111           (≈ 1 degree lat = 111 111 m)
  delta_lon = east_geo  / (111_111 * cos(lat))
"""

from __future__ import annotations

import math
from typing import Optional

import structlog

from proto.messages import BoundingBox, GeoPoint

logger = structlog.get_logger(__name__)

# Earth constants
_METRES_PER_DEG_LAT: float = 111_111.0
_MIN_ALTITUDE_M: float = 1.0   # below this, geo-tagging is meaningless


class GeoTagger:
    """
    Converts normalised pixel coordinates to GPS positions.

    Parameters
    ----------
    hfov_deg : float
        Camera horizontal field-of-view in degrees (default 90°).
    vfov_deg : float
        Camera vertical field-of-view in degrees (default 60°).
    tilt_deg : float
        Camera tilt below horizontal, degrees (0 = nadir / straight down,
        90 = forward-looking).  Only nadir (0) is fully supported; non-zero
        tilt applies a first-order forward-shift correction.
    min_confidence : float
        Minimum detection confidence required before geo-tagging is attempted
        (very-low-confidence detections get drone position as fallback).
    """

    def __init__(
        self,
        hfov_deg: float = 90.0,
        vfov_deg: float = 60.0,
        tilt_deg: float = 0.0,
        min_confidence: float = 0.3,
    ) -> None:
        self._htan = math.tan(math.radians(hfov_deg / 2.0))
        self._vtan = math.tan(math.radians(vfov_deg / 2.0))
        self._tilt_rad = math.radians(tilt_deg)
        self._min_confidence = min_confidence
        self._tag_count = 0
        self._fallback_count = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def pixel_to_gps(
        self,
        bbox: BoundingBox,
        drone_pos: GeoPoint,
        heading_deg: float = 0.0,
        confidence: float = 1.0,
    ) -> GeoPoint:
        """
        Convert a normalised bounding-box centre to a GPS position.

        Parameters
        ----------
        bbox : BoundingBox
            Normalised bounding box (x/y values in [0, 1]).
        drone_pos : GeoPoint
            Current drone GPS position; ``alt`` must be altitude AGL in metres.
        heading_deg : float
            Drone compass heading in degrees (0 = North, 90 = East).
        confidence : float
            Detection confidence — low-confidence detections fall back to
            drone position.

        Returns
        -------
        GeoPoint
            Estimated GPS position of the detected object.  Falls back to
            ``drone_pos`` whenever geo-tagging cannot be performed reliably.
        """
        # Guard: no useful altitude → return drone position
        alt = drone_pos.alt if drone_pos.alt is not None else 0.0
        if alt < _MIN_ALTITUDE_M or confidence < self._min_confidence:
            self._fallback_count += 1
            return drone_pos

        # Bounding-box centre in normalised [0, 1] coordinates
        cx_norm = (bbox.x_min + bbox.x_max) / 2.0
        cy_norm = (bbox.y_min + bbox.y_max) / 2.0

        # ── Tilt correction ──────────────────────────────────────────────
        # For a tilted camera, the nadir point shifts forward by:
        #   forward_shift_m = alt * tan(tilt)
        # We add this to the north offset (assumes gimbal faces drone heading).
        forward_shift_m = alt * math.tan(self._tilt_rad)

        # ── Ground-plane offsets in camera frame (East +, North +) ───────
        ground_east_m  = (cx_norm - 0.5) * 2.0 * alt * self._htan
        ground_north_m = (0.5 - cy_norm) * 2.0 * alt * self._vtan + forward_shift_m

        # ── Rotate by drone heading into geographic North/East ────────────
        theta = math.radians(heading_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        east_geo  =  ground_east_m * cos_t + ground_north_m * sin_t
        north_geo = -ground_east_m * sin_t + ground_north_m * cos_t

        # ── Convert metre offsets → degree offsets ────────────────────────
        delta_lat = north_geo / _METRES_PER_DEG_LAT
        lat_rad   = math.radians(drone_pos.lat)
        delta_lon = east_geo  / (_METRES_PER_DEG_LAT * math.cos(lat_rad) + 1e-10)

        self._tag_count += 1
        return GeoPoint(
            lat=drone_pos.lat + delta_lat,
            lon=drone_pos.lon + delta_lon,
            alt=0.0,    # detected object is on the ground
        )

    def tag_detections(
        self,
        detections: list,            # list[DetectionEvent]
        drone_pos: GeoPoint,
        heading_deg: float = 0.0,
    ) -> list:
        """
        Bulk-tag a list of DetectionEvent objects in-place.

        Each DetectionEvent's ``geo_position`` is replaced with the
        computed ground GPS position (or drone position as fallback).

        Returns the same list (mutated) for convenience.
        """
        if not drone_pos:
            return detections

        for det in detections:
            if det.bbox is None:
                det.geo_position = drone_pos
                continue

            det.geo_position = self.pixel_to_gps(
                bbox=det.bbox,
                drone_pos=drone_pos,
                heading_deg=heading_deg,
                confidence=getattr(det, "confidence", 1.0),
            )

        if detections:
            logger.debug(
                "geo_tagger.tagged",
                count=len(detections),
                alt_m=drone_pos.alt,
                heading=heading_deg,
            )

        return detections

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def haversine_distance(p1: GeoPoint, p2: GeoPoint) -> float:
        """
        Return the great-circle distance in metres between two GeoPoints.
        Useful for unit tests and sanity checks.
        """
        R = 6_371_000.0   # Earth radius metres
        dlat = math.radians(p2.lat - p1.lat)
        dlon = math.radians(p2.lon - p1.lon)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(p1.lat))
            * math.cos(math.radians(p2.lat))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    @property
    def stats(self) -> dict:
        """Return cumulative tagging statistics."""
        return {
            "tagged": self._tag_count,
            "fallbacks": self._fallback_count,
        }
