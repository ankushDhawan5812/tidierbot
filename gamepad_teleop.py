# Author: Jimmy Wu
# Date: October 2024
#
# Note: This code is only intended for debugging the base controller.
# In external code, avoid directly importing Vehicle as done here.
# Instead, please use the RPC server in base_server.py, which runs the
# low-level controller in a dedicated, real-time process to help minimize
# unintended latency spikes caused by external code.

import os
import signal
import time
import numpy as np
import pygame
from pygame.joystick import Joystick
from base_controller import Vehicle

# --- CONTROLLER MAP ------------------------ -------------------
BTN_START   = 10
BTN_BACK    = 9
BTN_LB      = 5
BTN_RB      = 6

AX_LEFT_Y   = 1
AX_LEFT_X   = 0   # ← left stick X axis  (left/right)
AX_RIGHT_X  = 2   # ← right stick X axis (spin)
# ---------------------------------------------------------------

pygame.init()

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))

class GamepadTeleop:
    def __init__(self):
        self.joy = Joystick(0)  # Logitech F710
        self.vehicle = None

    def run(self):
        last_enabled = False
        frame = None
        print('Press the "Start" button on the gamepad to start control')
        while True:
            pygame.event.pump()

            # Start control
            if not self.vehicle and self.joy.get_button(BTN_START):  # Replaced by Ahmed, 6/16/25
                self.vehicle = Vehicle(max_vel=(1.0, 1.0, 3.14), max_accel=(0.5, 0.5, 2.36))
                self.vehicle.start_control()
                last_enabled = False
                frame = 'local'
                print('Control started')

            # Stop control
            if self.vehicle and self.joy.get_button(BTN_BACK):  # Replaced by Ahmed, 6/16/25
                self.vehicle.stop_control()
                self.vehicle = None
                print('Control stopped')

            if self.vehicle:
                # Hold down left/right bumper to enable control in local/global frame
                left_bumper = self.joy.get_button(BTN_LB)
                right_bumper = self.joy.get_button(BTN_RB)
                frame = 'local' if left_bumper else 'global'
                if left_bumper or right_bumper:
                    if not last_enabled:
                        print(f'Robot enabled ({frame} frame)')
                        last_enabled = True

                    # Compute unscaled target velocity
                    x = self.joy.get_axis(AX_LEFT_X)  # Left analog stick
                    y = -self.joy.get_axis(AX_LEFT_Y)  # Left analog stick
                    th = -self.joy.get_axis(AX_RIGHT_X)  # Right analog stick
                    target_velocity = np.array([x, y, th])


                    # Apply deadzone for joystick drift
                    target_velocity = apply_deadzone(target_velocity)

                    # Send command to robot
                    target_velocity = self.vehicle.max_vel * target_velocity

                    # print("Commanded velocity:", target_velocity.round(2))
                    # print("cmd:", target_velocity.round(2), "frame:", frame)
                    self.vehicle.set_target_velocity(target_velocity, frame=frame)
                    # self.vehicle.set_target_position(self.vehicle.x + 1.5 * target_velocity)



                elif last_enabled:
                    print('Robot disabled')
                    last_enabled = False

            time.sleep(0.01)

# Handle SIGTERM
def handler(signum, frame):
    os.kill(os.getpid(), signal.SIGINT)
signal.signal(signal.SIGTERM, handler)

if __name__ == '__main__':
    teleop = GamepadTeleop()
    try:
        teleop.run()
    finally:
        if teleop.vehicle:
            teleop.vehicle.stop_control()
