# Fisheye camera calibration script for Kinova wrist camera
# Usage:
#   1a. Capture images: python calibrate_fisheye.py --capture
#   1b. Or extract from video: python calibrate_fisheye.py --video path/to/video.mp4
#   2. Run calibration: python calibrate_fisheye.py --calibrate

import argparse
import os
import cv2 as cv
import numpy as np
import glob

CALIBRATION_DIR = 'calibration_images'
CHECKERBOARD = (9, 6)  # Inner corners (columns, rows) - adjust to match your pattern
SQUARE_SIZE = 0.025    # Size of checkerboard square in meters (25mm default)
MIN_FRAME_INTERVAL = 15  # Minimum frames between captures when extracting from video


def capture_images():
    """Capture calibration images from the wrist camera."""
    from cameras import KinovaCamera
    import time

    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    print('Starting wrist camera...')
    camera = KinovaCamera()
    time.sleep(1)  # Wait for camera to warm up

    print('\n=== Fisheye Calibration Image Capture ===')
    print(f'Checkerboard pattern: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners')
    print(f'Images will be saved to: {CALIBRATION_DIR}/')
    print('\nInstructions:')
    print('  - Hold a checkerboard pattern in front of the camera')
    print('  - Move it to different positions and angles')
    print('  - Try to cover the entire field of view, especially corners')
    print('  - Capture 15-20 images for good calibration')
    print('\nControls:')
    print('  SPACE - Capture image')
    print('  Q     - Quit and finish capture')
    print()

    img_count = len(glob.glob(f'{CALIBRATION_DIR}/*.png'))
    print(f'Existing images: {img_count}')

    while True:
        image = camera.get_image()
        if image is None:
            time.sleep(0.01)
            continue

        # Convert to BGR for OpenCV display
        display = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Try to find checkerboard
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

        if ret:
            # Draw corners
            cv.drawChessboardCorners(display, CHECKERBOARD, corners, ret)
            status = 'Checkerboard FOUND - Press SPACE to capture'
            color = (0, 255, 0)
        else:
            status = 'Checkerboard not detected'
            color = (0, 0, 255)

        # Draw status
        cv.putText(display, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(display, f'Images captured: {img_count}', (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('Fisheye Calibration', display)
        key = cv.waitKey(1) & 0xFF

        if key == ord(' ') and ret:
            # Save image
            img_path = f'{CALIBRATION_DIR}/calib_{img_count:03d}.png'
            cv.imwrite(img_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
            print(f'Saved: {img_path}')
            img_count += 1

        elif key == ord('q'):
            break

    cv.destroyAllWindows()
    camera.close()
    print(f'\nCapture complete. Total images: {img_count}')
    print(f'Run calibration with: python calibrate_fisheye.py --calibrate')


def run_calibration():
    """Run fisheye calibration on captured images."""
    images = sorted(glob.glob(f'{CALIBRATION_DIR}/*.png'))

    if len(images) < 5:
        print(f'Error: Need at least 5 images for calibration. Found {len(images)}.')
        print(f'Capture more images with: python calibrate_fisheye.py --capture')
        return

    print(f'\n=== Fisheye Calibration ===')
    print(f'Found {len(images)} calibration images')
    print(f'Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners')
    print(f'Square size: {SQUARE_SIZE * 1000:.1f} mm')

    # Prepare object points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in world space
    imgpoints = []  # 2D points in image plane
    img_shape = None

    print('\nProcessing images...')
    for fname in images:
        img = cv.imread(fname)
        if img_shape is None:
            img_shape = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            # Refine corner positions
            corners = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            objpoints.append(objp)
            imgpoints.append(corners.reshape(1, -1, 2))
            print(f'  [OK] {os.path.basename(fname)}')
        else:
            print(f'  [SKIP] {os.path.basename(fname)} - checkerboard not found')

    if len(objpoints) < 5:
        print(f'\nError: Only found checkerboard in {len(objpoints)} images. Need at least 5.')
        return

    print(f'\nRunning fisheye calibration with {len(objpoints)} images...')

    # Initialize calibration matrices
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    # Calibration flags
    calibration_flags = (
        cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv.fisheye.CALIB_CHECK_COND +
        cv.fisheye.CALIB_FIX_SKEW
    )

    try:
        ret, K, D, rvecs, tvecs = cv.fisheye.calibrate(
            objpoints,
            imgpoints,
            img_shape[::-1],
            K, D,
            None, None,
            calibration_flags,
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    except cv.error as e:
        print(f'\nCalibration failed: {e}')
        print('Try capturing more images with varied angles and positions.')
        return

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    print('\n' + '=' * 50)
    print('CALIBRATION RESULTS')
    print('=' * 50)
    print(f'\nReprojection error: {mean_error:.4f} pixels')
    print('(Lower is better, < 1.0 is good)')

    print(f'\nImage size: {img_shape[1]} x {img_shape[0]}')

    print(f'\nCamera matrix K:')
    print(f'  fx = {K[0, 0]:.4f}')
    print(f'  fy = {K[1, 1]:.4f}')
    print(f'  cx = {K[0, 2]:.4f}')
    print(f'  cy = {K[1, 2]:.4f}')

    print(f'\nDistortion coefficients D:')
    print(f'  k1 = {D[0, 0]:.6f}')
    print(f'  k2 = {D[1, 0]:.6f}')
    print(f'  k3 = {D[2, 0]:.6f}')
    print(f'  k4 = {D[3, 0]:.6f}')

    # Save calibration
    calib_file = 'fisheye_calibration.npz'
    np.savez(calib_file, K=K, D=D, img_shape=img_shape)
    print(f'\nCalibration saved to: {calib_file}')

    # Print as Python code for easy copy-paste
    print('\n' + '=' * 50)
    print('PYTHON CODE (copy-paste ready):')
    print('=' * 50)
    print(f'''
# Fisheye camera intrinsics
K = np.array([
    [{K[0, 0]:.6f}, 0.0, {K[0, 2]:.6f}],
    [0.0, {K[1, 1]:.6f}, {K[1, 2]:.6f}],
    [0.0, 0.0, 1.0]
])
D = np.array([{D[0, 0]:.6f}, {D[1, 0]:.6f}, {D[2, 0]:.6f}, {D[3, 0]:.6f}])
''')

    # Save undistorted sample images
    print('\nSaving sample undistortion images...')
    for i, img_path in enumerate(images[:3]):  # Process first 3 images
        sample_img = cv.imread(img_path)
        h, w = sample_img.shape[:2]

        # Use the original K as the new camera matrix, scaled to show more of the image
        # The balance parameter (0-1) controls how much of the original image is visible
        # 0 = all pixels valid but loses FOV, 1 = keeps FOV but has black borders
        new_K = K.copy()
        new_K[0, 0] *= 0.5  # Reduce focal length to see more of the fisheye image
        new_K[1, 1] *= 0.5

        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv.CV_32FC1)
        undistorted = cv.remap(sample_img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        # Save side by side comparison
        combined = np.hstack([sample_img, undistorted])
        cv.putText(combined, 'Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(combined, 'Undistorted', (w + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_path = f'{CALIBRATION_DIR}/undistorted_comparison_{i}.png'
        cv.imwrite(output_path, combined)
        print(f'  Saved: {output_path}')

    print(f'\nDone! Check {CALIBRATION_DIR}/ for comparison images.')


def extract_from_video(video_path):
    """Extract calibration images from a video file."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    print(f'\n=== Extracting Calibration Images from Video ===')
    print(f'Video: {video_path}')
    print(f'Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners')

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Could not open video file: {video_path}')
        return

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f'Total frames: {total_frames}, FPS: {fps:.1f}')

    img_count = len(glob.glob(f'{CALIBRATION_DIR}/*.png'))
    print(f'Existing images: {img_count}')

    frame_idx = 0
    last_capture_frame = -MIN_FRAME_INTERVAL
    last_corners = None

    print('\nScanning video for checkerboard...')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames too close to last capture
        if frame_idx - last_capture_frame < MIN_FRAME_INTERVAL:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Try to find checkerboard
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

        if ret:
            # Check if corners moved significantly from last capture
            if last_corners is not None:
                movement = np.mean(np.abs(corners - last_corners))
                if movement < 5.0:  # Skip if checkerboard hasn't moved much
                    continue

            # Save image
            img_path = f'{CALIBRATION_DIR}/calib_{img_count:03d}.png'
            cv.imwrite(img_path, frame)
            print(f'  Frame {frame_idx}/{total_frames}: Saved {img_path}')

            last_capture_frame = frame_idx
            last_corners = corners.copy()
            img_count += 1

        # Progress update
        if frame_idx % 100 == 0:
            print(f'  Processed {frame_idx}/{total_frames} frames...')

    cap.release()
    print(f'\nExtraction complete. Total images: {img_count}')
    print(f'Run calibration with: python calibrate_fisheye.py --calibrate')


def load_calibration(calib_file='fisheye_calibration.npz'):
    """Load saved calibration."""
    data = np.load(calib_file)
    return data['K'], data['D'], data['img_shape']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fisheye camera calibration')
    parser.add_argument('--capture', action='store_true', help='Capture calibration images from live camera')
    parser.add_argument('--video', type=str, help='Extract calibration images from a video file')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration on captured images')
    parser.add_argument('--checkerboard', type=str, default='10x7',
                        help='Checkerboard inner corners as WxH (default: 10x7)')
    parser.add_argument('--square-size', type=float, default=16.5,
                        help='Checkerboard square size in mm (default: 16.5)')
    args = parser.parse_args()

    # Parse checkerboard size
    if 'x' in args.checkerboard:
        w, h = map(int, args.checkerboard.split('x'))
        CHECKERBOARD = (w, h)

    SQUARE_SIZE = args.square_size / 1000.0  # Convert mm to meters

    if args.capture:
        capture_images()
    elif args.video:
        extract_from_video(args.video)
    elif args.calibrate:
        run_calibration()
    else:
        print('Usage:')
        print('  1a. Capture images:     python calibrate_fisheye.py --capture')
        print('  1b. Extract from video: python calibrate_fisheye.py --video path/to/video.mp4')
        print('  2.  Run calibration:    python calibrate_fisheye.py --calibrate')
        print('\nOptions:')
        print('  --checkerboard WxH    Checkerboard inner corners (default: 10x7)')
        print('  --square-size MM      Square size in millimeters (default: 16.5)')