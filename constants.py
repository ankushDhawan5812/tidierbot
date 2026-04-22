import numpy as np

################################################################################
# Mobile base

# Vehicle center to steer axis (m)
h_x, h_y = 0.190150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.170150 * np.array([-1.0, 1.0, 1.0, -1.0])  # Kinova / Franka
# h_x, h_y = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])  # ARX5


# Encoder magnet offsets
# ENCODER_MAGNET_OFFSETS = [0.0 / 4096, 0.0 / 4096, 0.0 / 4096, 0.0 / 4096]  # TODO
# ENCODER_MAGNET_OFFSETS = [-804.0 / 4096, -1711.0 / 4096, 1396.0 / 4096, 677.0 / 4096]  # Calibrated with wheels straight forward
# ENCODER_MAGNET_OFFSETS = [-1948.0 / 4096, -2872.0 / 4096, 234.0 / 4096, -479.0 / 4096]
ENCODER_MAGNET_OFFSETS = [-819.0 / 4096, -5758.0 / 4096, 699.0 / 4096, -2687.0 / 4096]
# ENCODER_MAGNET_OFFSETS = [-2835.0 / 4096, -3736.0 / 4096, -618.0 / 4096, -1358.0 / 4096]
################################################################################
# Teleop and imitation learning

# Base and arm RPC servers
BASE_RPC_HOST = '100.87.168.15'
BASE_RPC_PORT = 50000
ARM_RPC_HOST = '100.87.168.15'
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b'secret password'

# Cameras
REALSENSE_SERIAL = None  # Set to specific serial if multiple RealSense cameras, or None for first available
# WRIST_CAMERA_SERIAL = 'TODO'  # Not used by Kinova wrist camera

# Policy
POLICY_SERVER_HOST = '100.87.168.15'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_IMAGE_WIDTH = 84
POLICY_IMAGE_HEIGHT = 84
