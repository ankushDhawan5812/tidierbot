# Test script to identify which caster is which by blinking LEDs
# Run with: python test_casters.py

import os
os.environ['CTR_TARGET'] = 'Hardware'

import time
from phoenix6 import hardware, controls, signals

def test_casters():
    print("Initializing motors...")

    # Create TalonFX motors (steer motors are odd: 1, 3, 5, 7)
    motors = []
    for caster_num in range(1, 5):
        steer_motor_id = 2 * caster_num - 1
        motor = hardware.TalonFX(steer_motor_id)
        motors.append((caster_num, steer_motor_id, motor))

    print("\nThis script will blink each caster's steer motor LED one at a time.")
    print("Watch which motor blinks to identify the caster number.\n")

    for caster_num, motor_id, motor in motors:
        input(f"Press Enter to blink caster {caster_num} (motor ID {motor_id})...")

        print(f"  Blinking caster {caster_num} for 3 seconds...")

        # Use music tone to make motor beep/blink
        music_request = controls.MusicTone(500)  # 500 Hz tone

        # Blink pattern: on/off
        for _ in range(6):
            motor.set_control(music_request)
            time.sleep(0.25)
            motor.set_control(controls.NeutralOut())
            time.sleep(0.25)

        print(f"  Caster {caster_num} done.\n")

    print("Test complete!")
    print("\nCaster positions based on h_x, h_y values:")
    print("  Caster 1 (index 0): h_x=+, h_y=-  -> Front-Right")
    print("  Caster 2 (index 1): h_x=+, h_y=+  -> Front-Left")
    print("  Caster 3 (index 2): h_x=-, h_y=+  -> Rear-Left")
    print("  Caster 4 (index 3): h_x=-, h_y=-  -> Rear-Right")

if __name__ == '__main__':
    test_casters()
