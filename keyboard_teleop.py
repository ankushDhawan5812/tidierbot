#!/usr/bin/env python3
import curses
import time
import numpy as np
import signal
import sys

from base_controller import Vehicle

# ---------------- Tunables ----------------
CONTROL_HZ = 100.0                       # match your gamepad loop
VEL_MAX = np.array([1.0, 1.0, 3.14])     # (vx, vy, yaw) max
ACC_MAX = np.array([0.5, 0.5, 2.36])     # accel limits passed to Vehicle
SCALE_DEFAULT = 0.5
SCALE_STEP = 0.1
POLL_WINDOW_S = 0.08                     # longer window so held keys are seen
# ------------------------------------------

HELP_TEXT = [
    "Keyboard Teleop for TidyBot (hold-to-move)",
    "-------------------------------------------",
    "Arrows or W/A/S/D : translate (UP/W fwd, DOWN/S back, LEFT/A strafe L, RIGHT/D strafe R)",
    "q / r             : rotate CCW / CW",
    "g                 : toggle global/local frame",
    "+ / -             : speed scale up / down",
    "z                 : stop (send zero once)",
    "x                 : quit",
]

def apply_deadzone(v, eps=0.0):
    v[np.abs(v) < eps] = 0.0
    return v

def draw_help(stdscr):
    stdscr.clear()
    for i, line in enumerate(HELP_TEXT):
        stdscr.addstr(i, 0, line)
    stdscr.refresh()

def read_key_with_arrows(stdscr, timeout_s=0.002):
    """
    Read a key; if it's an ESC-led arrow sequence, parse it and return a pseudo-key:
      'UP','DOWN','LEFT','RIGHT'  (strings)
    Otherwise return the integer code (ord).
    Non-blocking; may return -1 if no key.
    """
    ch = stdscr.getch()
    if ch != 27:  # not ESC, return as-is
        return ch

    # Possible escape sequence; read the next few bytes quickly
    # Typical arrow: ESC [ A/B/C/D  -> Up/Down/Right/Left
    stdscr.nodelay(True)
    start = time.time()
    seq = []
    while time.time() - start < timeout_s:
        nxt = stdscr.getch()
        if nxt == -1:
            time.sleep(0.0005)
            continue
        seq.append(nxt)
        if len(seq) >= 2:  # enough to decide
            break

    if len(seq) >= 2 and seq[0] == ord('['):
        code = seq[1]
        if code == ord('A'):
            return 'UP'
        if code == ord('B'):
            return 'DOWN'
        if code == ord('C'):
            return 'RIGHT'
        if code == ord('D'):
            return 'LEFT'

    # Unknown ESC sequence; ignore
    return -1

def keyboard_teleop(stdscr):
    # Curses setup
    curses.cbreak()
    curses.noecho()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    draw_help(stdscr)

    # Vehicle setup
    vehicle = Vehicle(max_vel=tuple(VEL_MAX), max_accel=tuple(ACC_MAX))
    vehicle.start_control()

    frame = 'local'
    scale = SCALE_DEFAULT

    dt = 1.0 / CONTROL_HZ
    next_t = time.time() + dt

    ended = False
    def cleanup():
        nonlocal ended
        if ended:
            return
        ended = True
        try:
            # send one zero just in case
            vehicle.set_target_velocity(np.zeros(3), frame=frame)
            vehicle.stop_control()
        except Exception:
            pass
        # Guard endwin: only call if we are still in curses mode
        try:
            if not curses.isendwin():
                curses.nocbreak()
                stdscr.keypad(False)
                curses.echo()
                curses.endwin()
        except Exception:
            pass

    def handle_sig(signum, _frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while True:
            # Aggregate keys seen during a small window this frame.
            t_end = time.time() + POLL_WINDOW_S
            want_left = want_right = want_up = want_down = False
            want_ccw = want_cw = False
            have_motion_key = False

            while True:
                key = read_key_with_arrows(stdscr)
                if key == -1:
                    if time.time() >= t_end:
                        break
                    time.sleep(0.001)
                    continue

                # Non-motion controls
                if key in (ord('+'), ord('=')):
                    scale = min(1.0, scale + SCALE_STEP)
                elif key == ord('-'):
                    scale = max(0.1, scale - SCALE_STEP)
                elif key == ord('g'):
                    frame = 'global' if frame == 'local' else 'local'
                elif key == ord('z'):
                    vehicle.set_target_velocity(np.zeros(3), frame=frame)
                elif key == ord('x'):
                    cleanup()
                    return
                # Motion flags
                elif key in ('LEFT', curses.KEY_LEFT, ord('a'), ord('A')):
                    want_left = True;  have_motion_key = True
                elif key in ('RIGHT', curses.KEY_RIGHT, ord('d'), ord('D')):
                    want_right = True; have_motion_key = True
                elif key in ('UP', curses.KEY_UP, ord('w'), ord('W')):
                    want_up = True;    have_motion_key = True
                elif key in ('DOWN', curses.KEY_DOWN, ord('s'), ord('S')):
                    want_down = True;  have_motion_key = True
                elif key == ord('q'):
                    want_ccw = True;   have_motion_key = True
                elif key == ord('r'):
                    want_cw = True;    have_motion_key = True
                # else: ignore

            # Build command for THIS frame only
            vx = 0.0
            vy = 0.0
            wz = 0.0
            if want_left:  vy += 1.0
            if want_right: vy -= 1.0
            if want_up:    vx += 1.0
            if want_down:  vx -= 1.0
            if want_ccw:   wz += 1.0
            if want_cw:    wz -= 1.0

            v = np.array([vx, vy, wz], dtype=float)
            v = apply_deadzone(v, 0.0)
            v = np.clip(v, -1.0, 1.0) * (VEL_MAX * scale)

            if have_motion_key:
                vehicle.set_target_velocity(v, frame=frame)
            else:
                vehicle.set_target_velocity(np.zeros(3), frame=frame)

            # HUD
            hud_row = len(HELP_TEXT) + 2
            stdscr.addstr(hud_row + 0, 0, f"Frame: {frame:<6}  Scale: {scale:0.2f}          ")
            stdscr.addstr(hud_row + 1, 0, f"v_cmd: [{v[0]: .2f}, {v[1]: .2f}, {v[2]: .2f}]  active:{have_motion_key}   ")
            stdscr.refresh()

            # Rate control
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

    finally:
        cleanup()

if __name__ == "__main__":
    curses.wrapper(keyboard_teleop)
