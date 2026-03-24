"""
VayuSwarm — Vehicle Cruise Controller (Webots)
Generic controller for cars, trucks, and motorbikes.
Each vehicle follows a simple preset route via waypoints.
"""

import argparse
import math
from controller import Robot, Motor, GPS

ROUTES = {
    "north_south":           [(-100, 120), (-100, -120), (-100, 120)],
    "east_west":             [(-120, 0),   (120, 0),     (-120, 0)],
    "north_south_secondary": [(60, 100),   (60, -100),   (60, 100)],
    "east_west_secondary":   [(-100, -60), (100, -60),   (-100, -60)],
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=7.0)
    parser.add_argument('--route', default='east_west')
    return parser.parse_args()


def main():
    args = get_args()
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Try to get GPS to read position
    try:
        gps = robot.getDevice('gps')
        gps.enable(timestep)
    except Exception:
        gps = None

    # Get wheels (different names per vehicle type)
    wheel_names = [
        ['front left wheel', 'front right wheel', 'rear left wheel', 'rear right wheel'],
        ['wheel1', 'wheel2', 'wheel3', 'wheel4'],
        ['wheel', 'wheel2'],
    ]
    wheels = []
    for names in wheel_names:
        found = []
        for n in names:
            try:
                m = robot.getDevice(n)
                m.setPosition(float('inf'))
                m.setVelocity(0)
                found.append(m)
            except Exception:
                pass
        if found:
            wheels = found
            break

    route = ROUTES.get(args.route, ROUTES['east_west'])
    wp_idx = 0
    speed = args.speed

    while robot.step(timestep) != -1:
        if not gps or not wheels:
            # Just spin wheels at speed if no GPS/wheels found
            for w in wheels:
                w.setVelocity(speed)
            continue

        pos = gps.getValues()
        cur_x, cur_y = pos[0], pos[1]

        target_x, target_y = route[wp_idx]
        dist = math.sqrt((target_x - cur_x)**2 + (target_y - cur_y)**2)

        if dist < 3.0:
            wp_idx = (wp_idx + 1) % len(route)

        for w in wheels:
            w.setVelocity(speed)


if __name__ == '__main__':
    main()
