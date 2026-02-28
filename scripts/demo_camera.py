"""
VayuSwarm — Live Camera Demo

Runs the vision pipeline on a webcam feed:
  Webcam → YOLOv8 → Thermal (mock) → Fusion → Behavior → Display

Press 'q' to quit.

Usage:
    python scripts/demo_camera.py [--camera 0] [--model models/yolo/best.pt]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Adjust path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proto.messages import (
    DetectionClass,
    DetectionSource,
    GeoPoint,
    ThreatLevel,
)
from src.vision.yolo_detector import YOLODetector
from src.vision.thermal_model import ThermalModel
from src.vision.sensor_fusion import SensorFusion
from src.vision.behavior_analyzer import BehaviorAnalyzer


# ─── Colors ──────────────────────────────────────────────────────
COLORS = {
    DetectionClass.PERSON: (0, 255, 0),           # Green
    DetectionClass.VEHICLE: (255, 165, 0),         # Orange
    DetectionClass.VEHICLE_CAR: (255, 165, 0),
    DetectionClass.VEHICLE_TRUCK: (255, 140, 0),
    DetectionClass.VEHICLE_MOTORCYCLE: (255, 200, 0),
    DetectionClass.WEAPON_RIFLE: (0, 0, 255),      # Red
    DetectionClass.WEAPON_PISTOL: (0, 0, 255),
    DetectionClass.WEAPON_HANDGUN: (0, 0, 255),
    DetectionClass.WEAPON_KNIFE: (0, 50, 255),
    DetectionClass.DRONE: (255, 0, 255),           # Magenta
    DetectionClass.FIRE: (0, 100, 255),            # Orange-red
    DetectionClass.SUSPICIOUS_PACKAGE: (0, 255, 255),  # Yellow
    DetectionClass.ANIMAL: (200, 200, 0),          # Cyan-ish
    DetectionClass.UNKNOWN: (128, 128, 128),       # Gray
}

THREAT_COLORS = {
    ThreatLevel.NONE: (100, 100, 100),
    ThreatLevel.LOW: (0, 200, 0),
    ThreatLevel.MEDIUM: (0, 200, 255),
    ThreatLevel.HIGH: (0, 100, 255),
    ThreatLevel.CRITICAL: (0, 0, 255),
}


def draw_detections(frame, fused_events):
    """Draw bounding boxes and labels on frame."""
    h, w = frame.shape[:2]

    for event in fused_events:
        color = COLORS.get(event.detection_class, (128, 128, 128))

        # Draw bounding box if available
        if event.bbox:
            x1 = int(event.bbox.x_min * w)
            y1 = int(event.bbox.y_min * h)
            x2 = int(event.bbox.x_max * w)
            y2 = int(event.bbox.y_max * h)

            # Box border color based on threat level
            threat_color = THREAT_COLORS.get(event.threat_level, color)
            thickness = 3 if event.threat_level.value >= ThreatLevel.HIGH.value else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), threat_color, thickness)

            # Label background
            label = f"{event.detection_class.value} {event.class_confidence:.0%}"
            if event.armed:
                label += " [ARMED]"
            if event.behavior.value != "unknown":
                label += f" | {event.behavior.value}"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def draw_hud(frame, fps, n_detections, yolo_stats):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]

    # Top-left: FPS and detection count
    cv2.rectangle(frame, (0, 0), (280, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"VayuSwarm Vision Demo", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Detections: {n_detections}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Frames: {yolo_stats.get('frames_processed', 0)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Bottom: controls
    cv2.rectangle(frame, (0, h - 25), (200, h), (0, 0, 0), -1)
    cv2.putText(frame, "Press 'q' to quit", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="VayuSwarm Live Camera Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "yolo" / "best.pt"),
                        help="Path to YOLOv8 model (default: models/yolo/best.pt)")
    parser.add_argument("--thermal-model", type=str,
                        default=str(PROJECT_ROOT / "models" / "thermal" / "thermal_classifier.onnx"),
                        help="Path to thermal ONNX model")
    parser.add_argument("--behavior-model", type=str,
                        default=str(PROJECT_ROOT / "models" / "behavior" / "behavior_transformer.onnx"),
                        help="Path to behavior ONNX model")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    args = parser.parse_args()

    print(f"🎥 VayuSwarm Camera Demo")
    print(f"   Camera: {args.camera}")
    print(f"   Model:  {args.model}")
    print(f"   Conf:   {args.conf}")
    print()

    # ── Initialize pipeline ──
    print("Loading YOLOv8...")
    yolo = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )
    yolo.load()

    print("Loading thermal model...")
    thermal = ThermalModel(model_path=args.thermal_model)
    thermal.load()

    print("Loading behavior model...")
    behavior = BehaviorAnalyzer(model_path=args.behavior_model)
    behavior.load()

    fusion = SensorFusion()

    # ── Open camera ──
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("✅ Pipeline ready! Press 'q' to quit.\n")

    # ── Simulated drone position ──
    drone_pos = GeoPoint(lat=17.385044, lon=78.486671, alt=50.0)

    fps = 0.0
    frame_count = 0

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                print("⚠ Camera frame read failed, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # ── 1. YOLO Detection ──
            rgb_detections = yolo.detect(frame, drone_pos)

            # ── 2. Thermal (mock — no real thermal camera) ──
            thermal_detections = thermal.mock_detect(drone_pos)

            # ── 3. Sensor Fusion ──
            fused_events = fusion.fuse(rgb_detections, thermal_detections, drone_pos)

            # ── 4. Behavior Analysis ──
            if fused_events:
                fused_events = behavior.analyze(fused_events)

            # ── 5. Draw ──
            display = frame.copy()
            display = draw_detections(display, fused_events)
            display = draw_hud(display, fps, len(fused_events), yolo.stats)

            cv2.imshow("VayuSwarm Vision", display)

            # Compute FPS
            elapsed = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6))

            # ── Controls ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    except KeyboardInterrupt:
        print("\n⛔ Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final stats:")
        print(f"   Frames: {frame_count}")
        print(f"   Avg FPS: {fps:.1f}")
        print(f"   Total detections: {yolo.stats['total_detections']}")


if __name__ == "__main__":
    main()
