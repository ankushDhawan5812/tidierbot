#!/usr/bin/env python3
import phoenix6
from phoenix6.hardware import TalonFX
import time

print("=" * 60)
print("Phoenix 6 Diagnostic Server Starting...")
print("=" * 60)

# Create a dummy device to force the diagnostic server to start
# This device doesn't need to exist on the CAN bus
dummy = TalonFX(0, "can0")  # Using device ID 0 on can0

print("Diagnostic server should now be running on port 1250")
print("Connect Phoenix Tuner X to: 10.32.38.133:1250")
print("Press Ctrl+C to stop")
print("=" * 60)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping diagnostic server...")