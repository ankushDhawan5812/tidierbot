from kinova import TorqueControlledArm

arm = TorqueControlledArm()
arm.zero_torque_offsets()
arm.disconnect()