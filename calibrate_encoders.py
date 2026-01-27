#!/usr/bin/env python3
"""
Encoder calibration script for TidyBot2 base.

Instructions:
1. Power on the robot but keep it disabled (e-stop engaged or motors off)
2. Manually rotate each wheel to point STRAIGHT FORWARD (parallel to robot's forward direction)
3. Run this script: python calibrate_encoders.py
4. Copy the output ENCODER_MAGNET_OFFSETS line to constants.py
"""

import os
os.environ['CTR_TARGET'] = 'Hardware'

import time
from phoenix6 import hardware, configs

NUM_CASTERS = 4

def get_encoder_offsets():
    print("Reading encoder positions...")
    print("Make sure all wheels are pointing STRAIGHT FORWARD!\n")

    offsets = []
    for num in range(1, NUM_CASTERS + 1):
        cancoder = hardware.CANcoder(num)
        cancoder_cfg = configs.CANcoderConfiguration()

        # Read current config
        cancoder.configurator.refresh(cancoder_cfg)
        curr_offset = cancoder_cfg.magnet_sensor.magnet_offset

        # Wait for fresh reading
        position_signal = cancoder.get_absolute_position()
        position_signal.wait_for_update(0.1)
        curr_position = position_signal.value

        new_offset = curr_offset - curr_position
        offsets.append(f'{round(4096 * new_offset)}.0 / 4096')

        print(f'Caster {num}: current_position = {curr_position:.4f}, new_offset = {new_offset:.4f}')

    print(f'\n# Copy this line to constants.py:')
    print(f'ENCODER_MAGNET_OFFSETS = [{", ".join(offsets)}]')

if __name__ == '__main__':
    input("Press Enter when all wheels are aligned straight forward...")
    get_encoder_offsets()
