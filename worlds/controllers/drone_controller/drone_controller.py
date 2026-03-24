"""
VayuSwarm — Webots Drone Controller
Runs inside each Webots drone node.

Responsibilities:
  1. Fly the drone using basic PD control (altitude + position hold)
  2. Accept MAVLink-style JSON commands from VayuSwarm backend over TCP
  3. Stream camera frames as JPEG over TCP to the camera bridge
"""

import argparse
import socket
import struct
import threading
import time
import json

import sys
import os

# ── Auto-detect and add Webots Python controller library path ──────────────
_WEBOTS_CANDIDATES = [
    os.environ.get('WEBOTS_HOME', ''),
    '/usr/local/webots',
    '/snap/webots/current',
    os.path.expanduser('~/webots'),
]
_CONTROLLER_ADDED = False
for _wb in _WEBOTS_CANDIDATES:
    if not _wb:
        continue
    _p = os.path.join(_wb, 'lib', 'controller', 'python')
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        _CONTROLLER_ADDED = True
        break

from controller import Robot, Camera, GPS, Gyro, InertialUnit, Motor

# ── Command server thread ──────────────────────────────────────────
class CommandServer(threading.Thread):
    def __init__(self, port, command_queue):
        super().__init__(daemon=True)
        self.port = port
        self.q = command_queue
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('0.0.0.0', port))
        self._sock.listen(5)

    def run(self):
        while True:
            try:
                conn, _ = self._sock.accept()
                data = b''
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                if data:
                    cmd = json.loads(data.decode())
                    self.q.append(cmd)
                conn.close()
            except Exception:
                pass


# ── Camera streamer thread ─────────────────────────────────────────
class CameraStreamer(threading.Thread):
    def __init__(self, camera, port, drone_id):
        super().__init__(daemon=True)
        self.camera = camera
        self.port = port
        self.drone_id = drone_id
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('0.0.0.0', port))
        self._server.listen(1)
        self._server.settimeout(1.0)
        print(f"[{drone_id}] Camera stream listening on port {port}")

    def update_frame(self, jpeg_bytes):
        with self._frame_lock:
            self._latest_frame = jpeg_bytes

    def run(self):
        while True:
            try:
                conn, addr = self._server.accept()
                print(f"[{self.drone_id}] Camera client connected: {addr}")
                conn.settimeout(2.0)
                while True:
                    with self._frame_lock:
                        frame = self._latest_frame
                    if frame is None:
                        time.sleep(0.033)
                        continue
                    try:
                        # Send: drone_id (32 bytes) + frame_len (4 bytes) + frame
                        id_bytes = self.drone_id.encode().ljust(32)[:32]
                        header = struct.pack('>I', len(frame))
                        conn.sendall(id_bytes + header + frame)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(0.1)  # ~10 FPS stream
                conn.close()
            except socket.timeout:
                continue
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drone-id', default='drone_01')
    parser.add_argument('--mavlink-port', type=int, default=14550)
    parser.add_argument('--camera-port', type=int, default=9001)
    parser.add_argument('--cmd-port', type=int, default=None)
    args = parser.parse_args()

    # cmd port defaults to mavlink-port + 100
    cmd_port = args.cmd_port if args.cmd_port else args.mavlink_port + 100

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ── Sensors ───────────────────────────────────────────────────
    imu = robot.getDevice('inertial unit')
    imu.enable(timestep)
    gps = robot.getDevice('gps')
    gps.enable(timestep)
    gyro = robot.getDevice('gyro')
    gyro.enable(timestep)
    camera = robot.getDevice(f'{args.drone_id}_camera')
    camera.enable(timestep)

    # ── Motors (4 rotors) ─────────────────────────────────────────
    m_names = ['front left propeller', 'front right propeller',
               'rear left propeller', 'rear right propeller']
    motors = [robot.getDevice(n) for n in m_names]
    for m in motors:
        m.setPosition(float('inf'))
        m.setVelocity(0.0)

    # ── Command queue ─────────────────────────────────────────────
    command_queue = []
    cmd_server = CommandServer(cmd_port, command_queue)
    cmd_server.start()

    # ── Camera streamer ───────────────────────────────────────────
    streamer = CameraStreamer(camera, args.camera_port, args.drone_id)
    streamer.start()

    print(f"[{args.drone_id}] Controller started | cmd:{cmd_port} | cam:{args.camera_port}")

    # ── Flight state ──────────────────────────────────────────────
    target_altitude = 0.0
    target_x = 0.0
    target_y = 0.0
    armed = False
    BASE_THROTTLE = 68.5  # hover RPM for Mavic 2 Pro

    K_alt_p = 10.0
    K_pos_p = 2.0
    K_att_p = 3.0

    frame_counter = 0

    while robot.step(timestep) != -1:
        # ── Process commands ──────────────────────────────────────
        while command_queue:
            cmd = command_queue.pop(0)
            action = cmd.get('action', '')
            if action == 'arm':
                armed = True
                print(f"[{args.drone_id}] ARMED")
            elif action == 'disarm':
                armed = False
                for m in motors:
                    m.setVelocity(0.0)
                print(f"[{args.drone_id}] DISARMED")
            elif action == 'takeoff':
                armed = True
                target_altitude = cmd.get('altitude', 15.0)
                print(f"[{args.drone_id}] TAKEOFF → {target_altitude}m")
            elif action == 'land':
                target_altitude = 0.3
                print(f"[{args.drone_id}] LANDING")
            elif action == 'goto':
                target_x = cmd.get('x', target_x)
                target_y = cmd.get('y', target_y)
                target_altitude = cmd.get('altitude', target_altitude)
                print(f"[{args.drone_id}] GOTO ({target_x}, {target_y}, {target_altitude})")
            elif action == 'hold':
                pos = gps.getValues()
                target_x, target_y = pos[0], pos[1]
                target_altitude = pos[2]
                print(f"[{args.drone_id}] HOLD at current position")
            elif action == 'rtl':
                target_x, target_y, target_altitude = 0.0, 5.0, 10.0
                print(f"[{args.drone_id}] RTL")

        if not armed:
            continue

        # ── Read sensors ──────────────────────────────────────────
        roll, pitch, yaw = imu.getRollPitchYaw()
        gpos = gps.getValues()
        cur_x, cur_y, cur_alt = gpos[0], gpos[1], gpos[2]
        gyro_vals = gyro.getValues()

        # ── Altitude PD controller ────────────────────────────────
        alt_err = target_altitude - cur_alt
        throttle = BASE_THROTTLE + K_alt_p * alt_err - gyro_vals[0] * 2.0

        # ── Position P controller (simplified) ────────────────────
        x_err = target_x - cur_x
        y_err = target_y - cur_y

        # Convert world XY error to body frame pitch/roll
        import math
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        target_pitch = K_pos_p * (cos_yaw * x_err + sin_yaw * y_err)
        target_roll  = K_pos_p * (-sin_yaw * x_err + cos_yaw * y_err)

        # Clamp
        target_pitch = max(-0.3, min(0.3, target_pitch))
        target_roll  = max(-0.3, min(0.3, target_roll))

        # ── Attitude PD ───────────────────────────────────────────
        roll_input  = K_att_p * (target_roll  - roll)  - gyro_vals[0]
        pitch_input = K_att_p * (target_pitch - pitch) - gyro_vals[1]
        yaw_input   = -gyro_vals[2] * 0.5

        # ── Motor mixing ──────────────────────────────────────────
        fl = throttle - roll_input + pitch_input - yaw_input
        fr = throttle + roll_input + pitch_input + yaw_input
        rl = throttle - roll_input - pitch_input + yaw_input
        rr = throttle + roll_input - pitch_input - yaw_input

        motors[0].setVelocity(max(0, fl))   # front left  (CCW)
        motors[1].setVelocity(-max(0, fr))  # front right (CW, negative)
        motors[2].setVelocity(-max(0, rl))  # rear left   (CW, negative)
        motors[3].setVelocity(max(0, rr))   # rear right  (CCW)

        # ── Camera capture every 3 steps (~10 FPS) ───────────────
        frame_counter += 1
        if frame_counter % 3 == 0:
            try:
                import cv2
                import numpy as np
                raw = camera.getImage()
                w, h = camera.getWidth(), camera.getHeight()
                img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                streamer.update_frame(jpeg.tobytes())
            except Exception as e:
                pass


if __name__ == '__main__':
    main()
