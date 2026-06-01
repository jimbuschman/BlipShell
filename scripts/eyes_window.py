"""Standalone Cozmo-style EYES device — BlipShell's face on a (simulated) OLED.

Run this and it connects to BlipShell's cube server like a real eye module. It's
a SMART device: BlipShell only sends the current mood (set_mood valence/arousal);
this window renders the procedural eyes AND runs its own keep-alive — random
blinks and saccade glances — so the eyes stay alive on their own, independent of
BlipShell/LLM latency (exactly how Cozmo/Vector eyes behave).

    # in BlipShell's config.yaml: robotics.enabled: true
    python -m scripts.eyes_window

Eyes are fully procedural (no pre-rendered frames): a rounded-rect base cut by
lids, parameterized by valence/arousal via robotics.eyes.eye_geometry, tweened
smoothly toward the commanded mood and animated with blink/saccade locally.
"""

import argparse
import asyncio
import json
import queue
import random
import socket
import threading
import tkinter as tk

from blipshell.robotics.cubes import VirtualEyes
from blipshell.robotics.emotion import AffectState, mood_label
from blipshell.robotics.eyes import eye_geometry

# Preview presets so you can SEE each mood (cycled by --demo or the 'd' key).
DEMO_PRESETS = [
    ("neutral", 0.0, -0.2),
    ("content", 0.5, -0.3),
    ("happy", 0.8, 0.3),
    ("excited", 0.8, 0.8),
    ("alert", 0.0, 0.9),
    ("sad", -0.7, -0.4),
    ("agitated", -0.7, 0.8),
    ("sleepy", 0.0, -0.9),
]

# 0.96" OLED is 128x64; render at 2x for visibility.
CANVAS_W, CANVAS_H = 256, 128
EYE_W, EYE_MAX_H, CORNER_R = 52, 76, 18
EYE_CY = 64
EYE_CX_L, EYE_CX_R = 90, 166
GAZE_X_RANGE, GAZE_Y_RANGE = 18, 14
ON_COLOR, BG_COLOR = "#46c8ff", "#000000"
FPS_MS = 33                 # ~30 fps render
TWEEN = 0.15                # mood/gaze easing per frame
MIN_H = 3                   # closed-eye sliver so a blink reads as a line
BLINK_FRAMES = 6


class EyesWindow:
    def __init__(self, host: str, port: int, cube_id: str):
        self.cube = VirtualEyes(cube_id=cube_id)
        self.loop = asyncio.new_event_loop()
        self.in_queue: queue.Queue = queue.Queue()
        self.send_lock = threading.Lock()
        self.sock = None
        self.sockfile = None

        # Displayed (tweened) mood — eases toward the cube's commanded target.
        self.disp_v = self.cube.target_valence
        self.disp_a = self.cube.target_arousal
        # Keep-alive state.
        self.frame = 0
        self.blink_phase = -1            # -1 = not blinking; else 0..BLINK_FRAMES
        self.next_blink = random.randint(60, 180)
        self.gaze = [0.0, 0.0]
        self.gaze_target = [0.0, 0.0]
        self.next_saccade = random.randint(45, 120)

        # Mood source: "live" (BlipShell drives), "manual" (arrow keys), "demo".
        self.mode = "live"
        self.manual_v, self.manual_a = 0.0, -0.2
        self.demo_idx = 0
        self.next_demo = 0

        self._build_ui(host, port)

    # --- UI -----------------------------------------------------------------

    def _build_ui(self, host: str, port: int) -> None:
        self.root = tk.Tk()
        self.root.title(f"Eyes — {self.cube.cube_id}")
        self.root.configure(bg=BG_COLOR)
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(padx=12, pady=12)
        self.status_var = tk.StringVar(value=f"connecting to {host}:{port}...")
        tk.Label(self.root, textvariable=self.status_var, fg="#888", bg=BG_COLOR,
                 font=("Consolas", 9)).pack()
        self.mood_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.mood_var, fg="#46c8ff", bg=BG_COLOR,
                 font=("Consolas", 11, "bold")).pack(pady=(2, 0))
        tk.Label(self.root,
                 text="arrows: set mood  ·  d: demo cycle  ·  b: back to BlipShell",
                 fg="#555", bg=BG_COLOR, font=("Consolas", 8)).pack(pady=(2, 8))

        self.root.bind("<Left>",  lambda e: self._nudge(-0.1, 0.0))
        self.root.bind("<Right>", lambda e: self._nudge(0.1, 0.0))
        self.root.bind("<Up>",    lambda e: self._nudge(0.0, 0.1))
        self.root.bind("<Down>",  lambda e: self._nudge(0.0, -0.1))
        self.root.bind("d", lambda e: self._set_mode("demo"))
        self.root.bind("b", lambda e: self._set_mode("live"))

    # --- mood source / preview controls -------------------------------------

    def _nudge(self, dv: float, da: float) -> None:
        self.mode = "manual"
        self.manual_v = max(-1.0, min(1.0, self.manual_v + dv))
        self.manual_a = max(-1.0, min(1.0, self.manual_a + da))

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "demo":
            self.demo_idx = 0
            self.next_demo = self.frame

    def _effective_target(self) -> tuple[float, float]:
        if self.mode == "manual":
            return self.manual_v, self.manual_a
        if self.mode == "demo":
            return DEMO_PRESETS[self.demo_idx][1], DEMO_PRESETS[self.demo_idx][2]
        return self.cube.target_valence, self.cube.target_arousal

    # --- drawing ------------------------------------------------------------

    @staticmethod
    def _round_rect_points(x0, y0, x1, y1, r):
        r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        return [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]

    def _draw_eye(self, cx, cy, shape, inner_is_right: bool) -> None:
        h = max(MIN_H, EYE_MAX_H * shape.openness)
        w = EYE_W * shape.width
        ox = cx + shape.gaze_x * GAZE_X_RANGE
        oy = cy + shape.gaze_y * GAZE_Y_RANGE
        x0, y0, x1, y1 = ox - w / 2, oy - h / 2, ox + w / 2, oy + h / 2

        # Lit eye body (rounded rect).
        self.canvas.create_polygon(
            self._round_rect_points(x0, y0, x1, y1, min(CORNER_R, h / 2)),
            smooth=True, fill=ON_COLOR, outline="")

        # Upper lids (cut from the top), mirrored per eye so the inner (nose-side)
        # corner is correct. cover_inner/outer are fractions of eye height.
        ci, co = shape.upper_lid_inner * h, shape.upper_lid_outer * h
        if ci > 0 or co > 0:
            if inner_is_right:   # left eye: inner corner is on the right
                pts = [x0, y0, x1, y0, x1, y0 + ci, x0, y0 + co]
            else:                # right eye: inner corner is on the left
                pts = [x0, y0, x1, y0, x1, y0 + co, x0, y0 + ci]
            self.canvas.create_polygon(pts, fill=BG_COLOR, outline="")

        # Lower lid (raise the bottom for a happy squint).
        ll = shape.lower_lid * h
        if ll > 0:
            self.canvas.create_rectangle(x0, y1 - ll, x1, y1, fill=BG_COLOR, outline="")

    def _render(self) -> None:
        self.frame += 1

        # Advance the demo cycle (hold each preset ~2.5s).
        if self.mode == "demo" and self.frame >= self.next_demo:
            self.demo_idx = (self.demo_idx + 1) % len(DEMO_PRESETS)
            self.next_demo = self.frame + 75

        # Tween displayed mood toward the effective target (live/manual/demo).
        tv, ta = self._effective_target()
        self.disp_v += (tv - self.disp_v) * TWEEN
        self.disp_a += (ta - self.disp_a) * TWEEN

        label = mood_label(AffectState(valence=self.disp_v, arousal=self.disp_a))
        tag = {"demo": DEMO_PRESETS[self.demo_idx][0] + " (demo)",
               "manual": "manual"}.get(self.mode, "live")
        self.mood_var.set(f"{label}   v={self.disp_v:+.2f} a={self.disp_a:+.2f}   [{tag}]")

        # Keep-alive: blink.
        blink = 0.0
        if self.blink_phase >= 0:
            half = BLINK_FRAMES / 2
            blink = 1.0 - abs(self.blink_phase - half) / half  # 0->1->0 triangle
            self.blink_phase += 1
            if self.blink_phase > BLINK_FRAMES:
                self.blink_phase = -1
                self.next_blink = self.frame + random.randint(60, 180)
        elif self.frame >= self.next_blink:
            self.blink_phase = 0

        # Keep-alive: saccade glances.
        if self.frame >= self.next_saccade:
            if random.random() < 0.5:
                self.gaze_target = [0.0, 0.0]            # often re-center
            else:
                self.gaze_target = [random.uniform(-1, 1) * 0.7,
                                    random.uniform(-1, 1) * 0.5]
            self.next_saccade = self.frame + random.randint(45, 150)
        self.gaze[0] += (self.gaze_target[0] - self.gaze[0]) * 0.25
        self.gaze[1] += (self.gaze_target[1] - self.gaze[1]) * 0.25

        shape = eye_geometry(self.disp_v, self.disp_a, blink=blink,
                             gaze=(self.gaze[0], self.gaze[1]))
        self.canvas.delete("all")
        self._draw_eye(EYE_CX_L, EYE_CY, shape, inner_is_right=True)
        self._draw_eye(EYE_CX_R, EYE_CY, shape, inner_is_right=False)
        self.root.after(FPS_MS, self._render)

    # --- networking ---------------------------------------------------------

    def connect(self, host: str, port: int) -> bool:
        try:
            self.sock = socket.create_connection((host, port), timeout=5)
        except OSError as e:
            self.status_var.set(f"could not connect to {host}:{port} — {e}. "
                                "Is BlipShell running with robotics.enabled?")
            return False
        self.sock.settimeout(None)
        self.sockfile = self.sock.makefile("rb")
        self._send({"type": "hello", "metadata": self.cube.describe().model_dump()})
        self.status_var.set(f"connected as {self.cube.cube_id}")
        threading.Thread(target=self._recv_loop, daemon=True).start()
        return True

    def _send(self, msg: dict) -> None:
        if self.sock is None:
            return
        try:
            with self.send_lock:
                self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        except OSError:
            self.in_queue.put({"type": "_closed"})

    def _recv_loop(self) -> None:
        try:
            for raw in self.sockfile:
                line = raw.strip()
                if not line:
                    continue
                try:
                    self.in_queue.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        self.in_queue.put({"type": "_closed"})

    def _poll(self) -> None:
        while True:
            try:
                msg = self.in_queue.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("type")
            if kind == "invoke":
                result = self.loop.run_until_complete(
                    self.cube.invoke(msg.get("action", ""), msg.get("args") or {}))
                self._send({"type": "result", "id": msg.get("id"), "result": result})
            elif kind == "_closed":
                self.status_var.set("disconnected from BlipShell")
        self.root.after(30, self._poll)

    def run(self, host: str, port: int) -> None:
        self.connect(host, port)
        self.root.focus_force()  # so arrow keys register immediately
        self.root.after(30, self._poll)
        self.root.after(FPS_MS, self._render)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Standalone Cozmo-style eyes device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cube-id", default="eyes_01")
    parser.add_argument("--demo", action="store_true",
                        help="cycle through the named moods so you can see each one")
    args = parser.parse_args()
    win = EyesWindow(args.host, args.port, args.cube_id)
    if args.demo:
        win._set_mode("demo")
    win.run(args.host, args.port)


if __name__ == "__main__":
    main()
