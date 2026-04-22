# Rectify fisheye video frames using saved calibration
# Usage: python rectify_video.py path/to/video.mp4 [--calib fisheye_calibration.npz] [--scale 0.5]

import argparse
import os
import cv2 as cv
import numpy as np


def load_calibration(calib_file='fisheye_calibration.npz'):
    """Load saved fisheye calibration."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f'Calibration file not found: {calib_file}\n'
            f'Run calibration first: python calibrate_fisheye.py --calibrate'
        )
    data = np.load(calib_file)
    return data['K'], data['D'], data['img_shape']


def rectify_video(video_path, calib_file='fisheye_calibration.npz', scale=0.5):
    """Rectify all frames of a video and save to parent directory."""
    # Load calibration
    print(f'Loading calibration from: {calib_file}')
    K, D, calib_img_shape = load_calibration(calib_file)

    print(f'Camera matrix K:\n{K}')
    print(f'Distortion D: {D.flatten()}')
    print(f'Calibrated image shape: {calib_img_shape}')

    # Open video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f'\nVideo: {video_path}')
    print(f'Resolution: {width}x{height}')
    print(f'Frames: {total_frames}, FPS: {fps:.1f}')

    # Create output directory in parent of video file
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(video_dir, f'{video_name}_rectified')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # Compute undistortion maps (only once for efficiency)
    # Scale controls how much of the original FOV to keep
    # Lower scale = wider view but more black borders
    new_K = K.copy()
    new_K[0, 0] *= scale  # fx
    new_K[1, 1] *= scale  # fy

    map1, map2 = cv.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (width, height), cv.CV_32FC1
    )

    print(f'\nRectifying frames (scale={scale})...')
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rectify frame
        rectified = cv.remap(frame, map1, map2,
                             interpolation=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_CONSTANT)

        # Save frame
        output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.png')
        cv.imwrite(output_path, rectified)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f'  Processed {frame_idx}/{total_frames} frames...')

    cap.release()
    print(f'\nDone! Saved {frame_idx} rectified frames to: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rectify fisheye video frames')
    parser.add_argument('video', type=str, help='Path to input MP4 video')
    parser.add_argument('--calib', type=str, default='fisheye_calibration.npz',
                        help='Path to calibration file (default: fisheye_calibration.npz)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Focal length scale for output (0.3-1.0, lower=wider view, default: 0.5)')
    args = parser.parse_args()

    rectify_video(args.video, args.calib, args.scale)
