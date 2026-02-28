#!/usr/bin/env python3
"""
VayuSwarm — Live Camera Detection

Opens your laptop camera (or a video/image), runs YOLOv8 detection,
and shows results on screen with bounding boxes.

Usage:
    # Laptop camera (default)
    python3 simulate.py

    # Specific model + laptop camera
    python3 simulate.py --model models/yolo/best.pt

    # Image file
    python3 simulate.py --source /path/to/image.jpg

    # Video file
    python3 simulate.py --source /path/to/video.mp4

    # External USB camera
    python3 simulate.py --source 1

Controls:
    q / ESC  — quit
    s        — save screenshot
    p        — pause / resume
    +/-      — adjust confidence threshold
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# ─── Colors for different detection classes ─────────────────────────────────
CLASS_COLORS = {
    "person":              (0, 230, 118),   # green
    "vehicle":             (255, 167, 38),   # orange
    "vehicle_car":         (255, 167, 38),
    "vehicle_truck":       (255, 193, 7),
    "vehicle_motorcycle":  (255, 214, 0),
    "weapon_rifle":        (0, 0, 255),      # red
    "weapon_pistol":       (0, 0, 255),
    "weapon_handgun":      (0, 0, 255),
    "weapon_knife":        (0, 69, 255),
    "drone":               (255, 0, 255),    # magenta
    "fire":                (0, 100, 255),    # dark orange
    "suspicious_package":  (0, 255, 255),    # yellow
    "animal":              (180, 130, 70),   # teal
    "unknown":             (128, 128, 128),  # gray
}

DEFAULT_COLOR = (200, 200, 200)


def get_color(class_name: str):
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_detections(frame, detections, conf_threshold):
    """Draw bounding boxes and labels on the frame."""
    h, w = frame.shape[:2]
    count = 0

    for det in detections:
        conf = det.confidence
        if conf < conf_threshold:
            continue

        cls_name = det.detection_class.value
        color = get_color(cls_name)

        # Convert normalized bbox to pixel coords
        bbox = det.bbox
        if bbox is None:
            continue
        x1 = int(bbox.x_min * w)
        y1 = int(bbox.y_min * h)
        x2 = int(bbox.x_max * w)
        y2 = int(bbox.y_max * h)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        count += 1

    return count


def draw_hud(frame, fps, det_count, conf_threshold, model_name, paused):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 36), (10, 14, 23), -1)
    cv2.putText(frame, "VAYUSWARM", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 212, 255), 2, cv2.LINE_AA)

    status_text = f"FPS: {fps:.0f}  |  Detections: {det_count}  |  Conf: {conf_threshold:.0%}  |  Model: {model_name}"
    cv2.putText(frame, status_text, (160, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 200), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "PAUSED", (w - 100, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Bottom bar — controls
    cv2.rectangle(frame, (0, h - 28), (w, h), (10, 14, 23), -1)
    controls = "q: Quit  |  s: Screenshot  |  p: Pause  |  +/-: Confidence"
    cv2.putText(frame, controls, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 110, 120), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="VayuSwarm — Live Camera Detection")
    parser.add_argument("--model", default="models/yolo/best.pt",
                        help="Path to YOLOv8 model (default: models/yolo/best.pt — VisDrone patrol)")
    parser.add_argument("--source", default="0",
                        help="Camera device (0=laptop), or path to image/video")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--size", type=int, default=640,
                        help="Inference size (default: 640)")
    parser.add_argument("--save-dir", default="data/screenshots",
                        help="Directory to save screenshots")
    args = parser.parse_args()

    # ── Load YOLO model ──
    print(f"\n[VayuSwarm] Loading model: {args.model}")
    from src.vision.yolo_detector import YOLODetector
    from proto.messages import GeoPoint

    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=0.45,
        device="auto",
    )
    detector.load()

    model_name = Path(args.model).stem
    print(f"[VayuSwarm] Model loaded: {model_name}")

    # ── Determine source ──
    source = args.source
    is_image = False

    # Check if source is a number (camera device)
    try:
        source = int(source)
        print(f"[VayuSwarm] Opening camera device: {source}")
    except ValueError:
        # It's a file path
        if not os.path.exists(source):
            print(f"[VayuSwarm] ERROR: Source not found: {source}")
            sys.exit(1)

        ext = Path(source).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            is_image = True
            print(f"[VayuSwarm] Processing image: {source}")
        else:
            print(f"[VayuSwarm] Opening video: {source}")

    # ── Image mode ──
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[VayuSwarm] ERROR: Could not read image: {source}")
            sys.exit(1)

        detections = detector.detect(frame, drone_position=GeoPoint(lat=0, lon=0, alt=0))
        count = draw_detections(frame, detections, args.conf)
        draw_hud(frame, 0, count, args.conf, model_name, False)

        print(f"[VayuSwarm] {count} detections found")
        for det in detections:
            print(f"  - {det.detection_class.value}: {det.confidence:.0%}")

        cv2.imshow("VayuSwarm Detection", frame)
        print("\n[VayuSwarm] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Video / Camera mode ──
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[VayuSwarm] ERROR: Could not open source: {source}")
        sys.exit(1)

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[VayuSwarm] Resolution: {actual_w}x{actual_h}")

    conf_threshold = args.conf
    paused = False
    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    screenshot_count = 0

    print("[VayuSwarm] Running... Press 'q' to quit\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Video ended — loop or exit
                if isinstance(source, int):
                    print("[VayuSwarm] Camera disconnected")
                    break
                else:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Run detection
            detections = detector.detect(frame, drone_position=GeoPoint(lat=0, lon=0, alt=0))
            det_count = draw_detections(frame, detections, conf_threshold)

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            draw_hud(frame, fps, det_count, conf_threshold, model_name, paused)
            display_frame = frame
        else:
            # Paused — keep showing last frame
            draw_hud(display_frame, fps, det_count, conf_threshold, model_name, paused)

        cv2.imshow("VayuSwarm Detection", display_frame)

        # ── Handle key presses ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('s'):
            # Save screenshot
            os.makedirs(args.save_dir, exist_ok=True)
            screenshot_count += 1
            path = os.path.join(args.save_dir, f"screenshot_{screenshot_count:04d}.jpg")
            cv2.imwrite(path, display_frame)
            print(f"[VayuSwarm] Screenshot saved: {path}")
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("[VayuSwarm] Paused")
            else:
                print("[VayuSwarm] Resumed")
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"[VayuSwarm] Confidence: {conf_threshold:.0%}")
        elif key == ord('-') or key == ord('_'):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"[VayuSwarm] Confidence: {conf_threshold:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[VayuSwarm] Done.")


if __name__ == "__main__":
    main()
