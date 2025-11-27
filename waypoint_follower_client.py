#!/usr/bin/env python3
import os
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints as FollowWayPoints  # Humble uses FollowWaypoints
from rclpy.duration import Duration
from rclpy.time import Time

import PyKDL


def yaw_to_quat(yaw: float) -> Quaternion:
    qx, qy, qz, qw = PyKDL.Rotation.RPY(0.0, 0.0, yaw).GetQuaternion()
    q = Quaternion()
    q.x, q.y, q.z, q.w = qx, qy, qz, qw
    return q


class FollowWaypointsClient(Node):
    def __init__(self):
        super().__init__('follow_waypoints_client')

        # Parameters
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('server_name', 'follow_waypoints')
        self.declare_parameter('goal_timeout_sec', 600.0)

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.server_name = self.get_parameter('server_name').get_parameter_value().string_value
        self.goal_timeout = Duration(seconds=float(self.get_parameter('goal_timeout_sec').value))

        self.client = ActionClient(self, FollowWayPoints, self.server_name)
        self.get_logger().info(
            f'FollowWaypoints client -> action: {self.server_name}, frame: {self.global_frame}'
        )

    # -------- file parsing (same rules as before) --------
    def read_coordinates(self, file_path: str) -> List[Tuple[float, float]]:
        coords: List[Tuple[float, float]] = []
        if not os.path.isfile(file_path):
            self.get_logger().error(f"Coordinates file not found: {file_path}")
            return coords
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) < 2:
                    self.get_logger().warning(f"Ignoring invalid line: {line}")
                    continue
                try:
                    x, y = float(parts[0]), float(parts[1])
                except ValueError:
                    self.get_logger().warning(f"Ignoring non-numeric line: {line}")
                    continue
                coords.append((x, y))
        return coords

    def make_waypoint_poses(self, coords: List[Tuple[float, float]]) -> List[PoseStamped]:
        poses: List[PoseStamped] = []
        n = len(coords)
        if n == 0:
            return poses

        # Compute a facing yaw for each waypoint (face the next point; last one keeps previous yaw)
        yaws: List[float] = []
        for i in range(n):
            if i < n - 1:
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
                yaws.append(math.atan2(dy, dx))
            else:
                yaws.append(yaws[-1] if i > 0 else 0.0)

        now = self.get_clock().now().to_msg()
        for (x, y), yaw in zip(coords, yaws):
            p = PoseStamped()
            p.header.frame_id = self.global_frame
            p.header.stamp = now
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation = yaw_to_quat(yaw)
            poses.append(p)
        return poses

    def send_waypoints(self, poses: List[PoseStamped]) -> bool:
        # Wait for action server
        self.get_logger().info('Waiting for FollowWaypoints action server...')
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('FollowWaypoints server not available. Is nav2_waypoint_follower running?')
            return False

        goal = FollowWayPoints.Goal()
        goal.poses = poses

        def fb_cb(fb):
            # fb.feedback.current_waypoint is uint32 index
            idx = int(getattr(fb.feedback, 'current_waypoint', 0))
            self.get_logger().info(f'Waypoint follower feedback: currently at index {idx}')

        self.get_logger().info(f'Sending {len(poses)} waypoints...')
        send_future = self.client.send_goal_async(goal, feedback_callback=fb_cb)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        if not send_future.done():
            self.get_logger().error('Timeout sending goal to FollowWaypoints.')
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('FollowWaypoints goal was rejected.')
            return False

        self.get_logger().info('Goal accepted. Waiting for result...')
        get_res_future = goal_handle.get_result_async()

        start = self.get_clock().now()
        while rclpy.ok():
            if (self.get_clock().now() - start) > self.goal_timeout:
                self.get_logger().warn('Waypoint following timed out. Canceling...')
                try:
                    _ = goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return False
            rclpy.spin_once(self, timeout_sec=0.25)
            if get_res_future.done():
                break

        res = get_res_future.result()
        missed = list(getattr(res.result, 'missed_waypoints', []))
        if missed:
            self.get_logger().warn(f'Completed with missed waypoints: {missed}')
        else:
            self.get_logger().info('All waypoints reached successfully.')
        return True

def main(args=None):
    rclpy.init(args=args)
    node = FollowWaypointsClient()
    try:
        coords = node.read_coordinates('coordinates.txt')
        if not coords:
            node.get_logger().error('No coordinates found; exiting.')
            return

        poses = node.make_waypoint_poses(coords)
        ok = node.send_waypoints(poses)
        if ok:
            node.get_logger().info('FollowWaypoints goal finished.')
        else:
            node.get_logger().error('FollowWaypoints goal failed or timed out.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
