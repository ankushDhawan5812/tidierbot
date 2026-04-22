#!/usr/bin/env python3
"""
RPC server for RealSense depth cameras. Runs on the NUC (tidybot-arm).

Serves depth frames and intrinsics to remote clients (e.g., cuRobo Docker
container) for nvblox obstacle avoidance.

Usage:
    python camera_server.py

Runs alongside arm_server.py on the same NUC.
"""

import time
import numpy as np
import pyrealsense2 as rs
from multiprocessing.managers import BaseManager as MPBaseManager

# Camera serial numbers (from homer/envs/cfgs/real_base_arm.yaml)
CAMERA_SERIALS = {
    'base1': '023522070875',
    'base2': '027422070360',
}

CAMERA_RPC_HOST = '0.0.0.0'
CAMERA_RPC_PORT = 50002
RPC_AUTHKEY = b'secret password'

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30


class Cameras:
    def __init__(self):
        self.pipelines = {}
        self.aligns = {}
        self.depth_filters = {}
        self.profiles = {}

        for name, serial in CAMERA_SERIALS.items():
            print(f'Starting {name} (serial={serial})...')

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)

            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)

            # Warmup frames
            for _ in range(30):
                pipeline.wait_for_frames()

            filters = [
                rs.spatial_filter(),
                rs.hole_filling_filter(),
                rs.temporal_filter(),
            ]

            self.pipelines[name] = pipeline
            self.aligns[name] = align
            self.depth_filters[name] = filters
            self.profiles[name] = profile

            print(f'  {name} ready.')

        print('All cameras initialized.')

    def capture_depth(self, cam_name):
        """Capture a depth frame, returned as float32 in meters.

        Returns dict with:
            'depth': np.ndarray (H, W) float32 meters
        """
        pipeline = self.pipelines[cam_name]
        align = self.aligns[cam_name]
        filters = self.depth_filters[cam_name]

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()

        for f in filters:
            depth_frame = f.process(depth_frame)

        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        depth_scale = self.profiles[cam_name].get_device().first_depth_sensor().get_depth_scale()
        depth_meters = depth_raw.astype(np.float32) * depth_scale

        return {
            'depth': depth_meters,
        }

    def get_intrinsics(self, cam_name):
        """Return camera intrinsics dict.

        Returns dict with:
            'matrix': np.ndarray (3, 3) camera K matrix
            'width': int
            'height': int
            'depth_scale': float
        """
        profile = self.profiles[cam_name]
        cprofile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        cintrinsics = cprofile.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        return {
            'matrix': np.array([
                [cintrinsics.fx, 0, cintrinsics.ppx],
                [0, cintrinsics.fy, cintrinsics.ppy],
                [0, 0, 1.0],
            ]),
            'width': cintrinsics.width,
            'height': cintrinsics.height,
            'depth_scale': depth_scale,
        }

    def list_cameras(self):
        """Return list of available camera names."""
        return list(self.pipelines.keys())

    def close(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()


class CameraManager(MPBaseManager):
    pass

CameraManager.register('Cameras', Cameras)

if __name__ == '__main__':
    manager = CameraManager(address=(CAMERA_RPC_HOST, CAMERA_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Camera server started at {CAMERA_RPC_HOST}:{CAMERA_RPC_PORT}')
    server.serve_forever()
