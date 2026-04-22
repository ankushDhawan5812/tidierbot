# Fisheye alignment debug script
# Saves an image with the detected fisheye boundary and centers overlaid

import cv2 as cv
import numpy as np
from cameras import KinovaCamera, find_fisheye_center

def create_fisheye_debug_image(image):
    """Create debug image showing fisheye boundary and centers."""
    # Convert to BGR for drawing (image is in RGB)
    debug_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    height, width = debug_image.shape[:2]

    # Image center
    image_center = (width // 2, height // 2)

    # Find fisheye center and radius
    center, radius = find_fisheye_center(debug_image)

    # Draw image center (green cross)
    cross_size = 20
    cv.line(debug_image, (image_center[0] - cross_size, image_center[1]),
            (image_center[0] + cross_size, image_center[1]), (0, 255, 0), 2)
    cv.line(debug_image, (image_center[0], image_center[1] - cross_size),
            (image_center[0], image_center[1] + cross_size), (0, 255, 0), 2)
    cv.putText(debug_image, 'Image Center', (image_center[0] + 10, image_center[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if center is not None and radius is not None:
        detected_center = (int(center[0]), int(center[1]))
        detected_radius = int(radius)

        # Draw detected fisheye boundary (blue circle)
        cv.circle(debug_image, detected_center, detected_radius, (255, 0, 0), 2)

        # Draw detected center (red cross)
        cv.line(debug_image, (detected_center[0] - cross_size, detected_center[1]),
                (detected_center[0] + cross_size, detected_center[1]), (0, 0, 255), 2)
        cv.line(debug_image, (detected_center[0], detected_center[1] - cross_size),
                (detected_center[0], detected_center[1] + cross_size), (0, 0, 255), 2)
        cv.putText(debug_image, 'Fisheye Center', (detected_center[0] + 10, detected_center[1] + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw line between centers
        cv.line(debug_image, image_center, detected_center, (255, 255, 0), 1)

        # Calculate and display offset
        offset_x = detected_center[0] - image_center[0]
        offset_y = detected_center[1] - image_center[1]
        offset_pct_x = offset_x / width * 100
        offset_pct_y = offset_y / height * 100

        # Check if centered (within 5% threshold from check_fisheye_centered)
        is_centered = abs(offset_pct_x) < 5 and abs(offset_pct_y) < 5
        status_color = (0, 255, 0) if is_centered else (0, 0, 255)
        status_text = "CENTERED" if is_centered else "OFF-CENTER"

        # Display info
        cv.putText(debug_image, f'Offset: ({offset_x}, {offset_y}) px', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(debug_image, f'Offset: ({offset_pct_x:.1f}%, {offset_pct_y:.1f}%)', (10, 55),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(debug_image, f'Radius: {detected_radius} px', (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(debug_image, status_text, (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    else:
        cv.putText(debug_image, 'No fisheye boundary detected', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Legend
    cv.putText(debug_image, 'Green: Image Center', (10, height - 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.putText(debug_image, 'Red: Detected Fisheye Center', (10, height - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return debug_image


if __name__ == '__main__':
    import os
    import time

    print('Initializing Kinova camera...')
    camera = KinovaCamera()

    try:
        time.sleep(1)  # Wait for camera to stabilize
        image = camera.get_image()

        if image is None:
            print('Failed to capture image')
            exit(1)

        debug_image = create_fisheye_debug_image(image)

        # Save to camera_captures directory
        save_dir = os.path.join(os.path.dirname(__file__), 'camera_captures')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'fisheye_debug.png')

        cv.imwrite(save_path, debug_image)
        print(f'Saved fisheye debug image to {save_path}')

    finally:
        camera.close()
