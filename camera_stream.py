# Camera streaming server for remote viewing
# Run alongside main.py to stream camera feeds over HTTP

import threading
import time
import cv2 as cv
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Global references to cameras (set by start_stream_server)
_base_camera = None
_wrist_camera = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>TidyBot Camera Stream</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: white; margin: 20px; }
        h1 { text-align: center; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        .camera { text-align: center; }
        .camera img { max-width: 640px; border: 2px solid #444; }
        .camera h3 { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>TidyBot Camera Stream</h1>
    <div class="container">
        <div class="camera">
            <h3>Base Camera (RealSense)</h3>
            <img src="/stream/base" alt="Base Camera">
        </div>
        <div class="camera">
            <h3>Base Depth</h3>
            <img src="/stream/depth" alt="Base Depth">
        </div>
        <div class="camera">
            <h3>Wrist Camera (Kinova)</h3>
            <img src="/stream/wrist" alt="Wrist Camera">
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames(camera_type):
    """Generator function for MJPEG streaming."""
    while True:
        frame = None

        if camera_type == 'base' and _base_camera is not None:
            image = _base_camera.get_image()
            if image is not None:
                frame = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        elif camera_type == 'depth' and _base_camera is not None:
            depth = _base_camera.get_depth()
            if depth is not None:
                frame = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=0.03), cv.COLORMAP_JET)

        elif camera_type == 'wrist' and _wrist_camera is not None:
            image = _wrist_camera.get_image()
            if image is not None:
                frame = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if frame is not None:
            _, jpeg = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        time.sleep(0.033)  # ~30 fps

@app.route('/stream/<camera_type>')
def stream(camera_type):
    if camera_type not in ['base', 'depth', 'wrist']:
        return "Invalid camera type", 404
    return Response(generate_frames(camera_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_stream_server(base_camera, wrist_camera, host='0.0.0.0', port=5001):
    """Start the streaming server in a background thread."""
    global _base_camera, _wrist_camera
    _base_camera = base_camera
    _wrist_camera = wrist_camera

    def run_server():
        # Suppress Flask's default logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        print(f'Camera stream server running at http://{host}:{port}')
        app.run(host=host, port=port, threaded=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread


if __name__ == '__main__':
    # Standalone test mode
    from cameras import RealSenseCamera, KinovaCamera
    from constants import REALSENSE_SERIAL

    print('Starting cameras...')
    base_camera = RealSenseCamera(REALSENSE_SERIAL)
    wrist_camera = KinovaCamera()

    print('Starting stream server...')
    start_stream_server(base_camera, wrist_camera)

    print('Press Ctrl+C to stop')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        base_camera.close()
        wrist_camera.close()
