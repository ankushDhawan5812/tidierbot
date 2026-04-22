# Author: Jimmy Wu
# Date: October 2024
# Modified: Added execute_joint_position, execute_joint_trajectory, set_gripper for cuRobo integration
#
# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

import queue
import time
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from arm_controller import JointCompliantController
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from ik_solver import IKSolver
from kinova import TorqueControlledArm

class Arm:
    def __init__(self):
        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None
        self.ik_solver = IKSolver(ee_offset=0.12)

    def reset(self):
        try:
            # Stop low-level control
            if self.arm.cyclic_running:
                time.sleep(0.75)  # Wait for arm to stop moving
                self.arm.stop_cyclic()
            # Drain any stale commands from previous trajectory
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break
            # Clear faults
            self.arm.clear_faults()
            # Reset arm configuration
            self.arm.retract()
            # Create new instance of controller
            self.controller = JointCompliantController(self.command_queue)
            # Start low-level control
            self.arm.init_cyclic(self.controller.control_callback)
            while not self.arm.cyclic_running:
                time.sleep(0.01)
        except Exception as e:
            import traceback
            raise RuntimeError(f'Arm reset failed: {traceback.format_exc()}') from None

    def execute_action(self, action):
        qpos = self.ik_solver.solve(action['arm_pos'], action['arm_quat'], self.arm.q)
        self.command_queue.put((qpos, action['gripper_pos'].item()))

    def get_state(self):
        arm_pos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        state = {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([self.arm.gripper_pos]),
        }
        return state

    def execute_joint_position(self, qpos, gripper_pos=None):
        """Send a single joint position command.

        Args:
            qpos: List of 7 joint angles in radians.
            gripper_pos: Gripper position 0.0 (open) to 1.0 (closed).
                If None, maintain current gripper position.
        """
        if gripper_pos is None:
            gripper_pos = self.arm.gripper_pos
        self.command_queue.put((np.array(qpos), float(gripper_pos)))

    def execute_joint_trajectory(self, positions, dt, gripper_positions=None):
        """Execute a joint trajectory by streaming waypoints through the command queue.

        Args:
            positions: List of joint position arrays, each of length 7.
            dt: Time step between waypoints in seconds.
            gripper_positions: Optional gripper positions (0.0-1.0).
                Single float for constant grip, or list with one per waypoint.
                If None, maintain current gripper position.
        """
        n = len(positions)
        if gripper_positions is None:
            grip = [self.arm.gripper_pos] * n
        elif isinstance(gripper_positions, (int, float)):
            grip = [float(gripper_positions)] * n
        else:
            grip = list(gripper_positions)
        for i in range(n):
            next_time = time.perf_counter() + dt
            try:
                self.command_queue.put_nowait((np.array(positions[i]), grip[i]))
            except queue.Full:
                pass  # Skip waypoint if queue full — Ruckig is still tracking
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def set_gripper(self, gripper_pos, wait_time=1.0):
        """Set gripper position while keeping arm in place.

        Args:
            gripper_pos: 0.0 (open) to 1.0 (closed).
            wait_time: Seconds to wait for gripper to reach position.
        """
        self.command_queue.put((self.arm.q.copy(), float(gripper_pos)))
        time.sleep(wait_time)

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()
        self.arm.disconnect()

class ArmManager(MPBaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    server.serve_forever()
