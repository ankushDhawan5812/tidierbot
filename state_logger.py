# State logger for tracking arm and base state over time
# Usage:
#   python state_logger.py              # Log to CSV file
#   python state_logger.py --plot       # Plot from existing log file

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np

LOG_DIR = Path('state_logs')


class StateLogger:
    """Logs arm and base state over time to a CSV file."""

    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.log_path = self.log_dir / f'state_log_{timestamp}.csv'

        # CSV headers
        self.headers = [
            'timestamp',
            'base_x', 'base_y', 'base_theta',
            'arm_x', 'arm_y', 'arm_z',
            'arm_qx', 'arm_qy', 'arm_qz', 'arm_qw',
            'gripper_pos'
        ]

        # Open file and write headers
        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)

        self.start_time = time.time()
        self.count = 0

        print(f'Logging state to: {self.log_path}')

    def log(self, obs):
        """Log a single observation."""
        timestamp = time.time() - self.start_time

        row = [
            f'{timestamp:.4f}',
            f'{obs["base_pose"][0]:.6f}',
            f'{obs["base_pose"][1]:.6f}',
            f'{obs["base_pose"][2]:.6f}',
            f'{obs["arm_pos"][0]:.6f}',
            f'{obs["arm_pos"][1]:.6f}',
            f'{obs["arm_pos"][2]:.6f}',
            f'{obs["arm_quat"][0]:.6f}',
            f'{obs["arm_quat"][1]:.6f}',
            f'{obs["arm_quat"][2]:.6f}',
            f'{obs["arm_quat"][3]:.6f}',
            f'{obs["gripper_pos"][0]:.6f}',
        ]

        self.writer.writerow(row)
        self.count += 1

        # Flush periodically
        if self.count % 100 == 0:
            self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.close()
        print(f'Saved {self.count} observations to {self.log_path}')


def load_log(log_path):
    """Load a state log file."""
    import pandas as pd
    return pd.read_csv(log_path)


def plot_log(log_path):
    """Plot state data from a log file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed. Install with: pip install matplotlib')
        return

    df = load_log(log_path)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'State Log: {Path(log_path).name}')

    # Base trajectory (x, y)
    ax = axes[0, 0]
    ax.plot(df['base_x'], df['base_y'], 'b-', linewidth=0.5)
    ax.scatter(df['base_x'].iloc[0], df['base_y'].iloc[0], c='green', s=100, label='Start', zorder=5)
    ax.scatter(df['base_x'].iloc[-1], df['base_y'].iloc[-1], c='red', s=100, label='End', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Base Trajectory (Top View)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)

    # Base pose over time
    ax = axes[0, 1]
    ax.plot(df['timestamp'], df['base_x'], label='x')
    ax.plot(df['timestamp'], df['base_y'], label='y')
    ax.plot(df['timestamp'], df['base_theta'], label='theta')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('Base Pose Over Time')
    ax.legend()
    ax.grid(True)

    # Arm position (3D trajectory)
    ax = axes[1, 0]
    ax.plot(df['timestamp'], df['arm_x'], label='x')
    ax.plot(df['timestamp'], df['arm_y'], label='y')
    ax.plot(df['timestamp'], df['arm_z'], label='z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Arm Position Over Time')
    ax.legend()
    ax.grid(True)

    # Arm trajectory (x, y) top view
    ax = axes[1, 1]
    ax.plot(df['arm_x'], df['arm_y'], 'b-', linewidth=0.5)
    ax.scatter(df['arm_x'].iloc[0], df['arm_y'].iloc[0], c='green', s=100, label='Start', zorder=5)
    ax.scatter(df['arm_x'].iloc[-1], df['arm_y'].iloc[-1], c='red', s=100, label='End', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Arm Trajectory (Top View)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)

    # Arm quaternion
    ax = axes[2, 0]
    ax.plot(df['timestamp'], df['arm_qx'], label='qx')
    ax.plot(df['timestamp'], df['arm_qy'], label='qy')
    ax.plot(df['timestamp'], df['arm_qz'], label='qz')
    ax.plot(df['timestamp'], df['arm_qw'], label='qw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion')
    ax.set_title('Arm Orientation Over Time')
    ax.legend()
    ax.grid(True)

    # Gripper position
    ax = axes[2, 1]
    ax.plot(df['timestamp'], df['gripper_pos'], 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gripper Position')
    ax.set_title('Gripper Over Time')
    ax.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = str(log_path).replace('.csv', '_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to: {plot_path}')

    plt.show()


def run_logging(duration=None, freq=30):
    """Run state logging from the robot."""
    from real_env import RealEnv

    print('Connecting to robot...')
    env = RealEnv(use_cameras=False)  # No cameras needed for state logging

    logger = StateLogger()

    print(f'Logging at {freq} Hz. Press Ctrl+C to stop.')
    if duration:
        print(f'Will stop after {duration} seconds.')

    period = 1.0 / freq
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            obs = env.get_obs()
            logger.log(obs)

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f'\nDuration of {duration}s reached.')
                break

            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        logger.close()
        env.close()

    return logger.log_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log and plot robot state over time')
    parser.add_argument('--plot', type=str, help='Plot from an existing log file')
    parser.add_argument('--duration', type=float, help='Duration to log in seconds (default: until Ctrl+C)')
    parser.add_argument('--freq', type=float, default=30, help='Logging frequency in Hz (default: 30)')
    parser.add_argument('--list', action='store_true', help='List existing log files')
    args = parser.parse_args()

    if args.list:
        logs = sorted(LOG_DIR.glob('*.csv'))
        if logs:
            print('Existing log files:')
            for log in logs:
                print(f'  {log}')
        else:
            print('No log files found.')
    elif args.plot:
        plot_log(args.plot)
    else:
        log_path = run_logging(duration=args.duration, freq=args.freq)
        print(f'\nTo plot this log, run:')
        print(f'  python state_logger.py --plot {log_path}')
