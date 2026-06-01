"""Standalone Cozmo-style EYES device — BlipShell's face on a (simulated) OLED.

Renders the faithful esp32-eyes model (blipshell.robotics.eye_config), so what
you see previews the real 0.96" OLED hardware. Connects to BlipShell's cube
server like a real eye module; it's a SMART device — BlipShell sends only the
mood (set_mood valence/arousal) and this window renders the procedural eyes,
tweens to the commanded mood, and runs its own keep-alive (random blinks +
saccade glances) locally, independent of BlipShell/LLM latency.

    # in BlipShell's config.yaml: robotics.enabled: true
    python -m scripts.eyes_window            # connect + show live mood
    python -m scripts.eyes_window --demo     # cycle the 18 named expressions

Keys: arrows set mood (valence/arousal) · d = demo cycle · b = back to BlipShell.
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
from blipshell.robotics.eye_config import (
    PRESETS,
    eye_outline,
    lerp_config,
    mirror_config,
    mood_to_config,
    with_blink,
)
from blipshell.robotics.emotion import AffectState, mood_label

# Logical display is 128x64 (the OLED); render at SCALE for visibility.
DISPLAY_W, DISPLAY_H, SCALE = 128, 64, 4
EYE_CX_L, EYE_CX_R, EYE_CY = 40, 88, 32     # logical eye centers
GAZE_X, GAZE_Y = 9, 6                        # logical gaze travel
ON_COLOR, BG_COLOR = "#46c8ff", "#000000"
FPS_MS, TWEEN = 33, 0.18
BLINK_FRAMES = 6
DEMO_ORDER = [
    "normal", "happy", "glee", "surprised", "awe", "skeptic", "suspicious",
    "focused", "worried", "sad", "sleepy", "annoyed", "unimpressed",
    "frustrated", "squint", "angry", "furious", "scared",
]


class EyesWindow:
    def __init__(self, host: str, port: int, cube_id: str):
        self.cube = VirtualEyes(cube_id=cube_id)
        self.loop = asyncio.new_event_loop()
        self.in_queue: queue.Queue = queue.Queue()
        self.send_lock = threading.Lock()
        self.sock = None
        self.sockfile = None

        self.disp_cfg = PRESETS["normal"]            # current (tweened) eye config
        self.frame = 0
        self.blink_phase = -1
        self.next_blink = random.randint(60, 180)
        self.gaze = [0.0, 0.0]
        self.gaze_target = [0.0, 0.0]
        self.next_saccade = random.randint(45, 120)

        self.mode = "live"                            # live | manual | demo
        self.manual_v, self.manual_a = 0.0, -0.2
        self.demo_idx = 0
        self.next_demo = 0

        self._build_ui(host, port)

    # --- UI -----------------------------------------------------------------

    def _build_ui(self, host: str, port: int) -> None:
        self.root = tk.Tk()
        self.root.title(f"Eyes — {self.cube.cube_id}")
        self.root.configure(bg=BG_COLOR)
        self.canvas = tk.Canvas(self.root, width=DISPLAY_W * SCALE, height=DISPLAY_H * SCALE,
                                bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(padx=12, pady=12)
        self.status_var = tk.StringVar(value=f"connecting to {host}:{port}...")
        tk.Label(self.root, textvariable=self.status_var, fg="#888", bg=BG_COLOR,
                 font=("Consolas", 9)).pack()
        self.mood_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.mood_var, fg="#46c8ff", bg=BG_COLOR,
                 font=("Consolas", 11, "bold")).pack(pady=(2, 0))
        tk.Label(self.root, text="arrows: set mood · d: demo cycle · b: back to BlipShell",
                 fg="#555", bg=BG_COLOR, font=("Consolas", 8)).pack(pady=(2, 8))
        self.root.bind("<Left>",  lambda e: self._nudge(-0.1, 0.0))
        self.root.bind("<Right>", lambda e: self._nudge(0.1, 0.0))
        self.root.bind("<Up>",    lambda e: self._nudge(0.0, 0.1))
        self.root.bind("<Down>",  lambda e: self._nudge(0.0, -0.1))
        self.root.bind("d", lambda e: self._set_mode("demo"))
        self.root.bind("b", lambda e: self._set_mode("live"))

    def _nudge(self, dv: float, da: float) -> None:
        self.mode = "manual"
        self.manual_v = max(-1.0, min(1.0, self.manual_v + dv))
        self.manual_a = max(-1.0, min(1.0, self.manual_a + da))

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "demo":
            self.demo_idx = 0
            self.next_demo = self.frame

    def _target_config(self):
        if self.mode == "demo":
            return PRESETS[DEMO_ORDER[self.demo_idx]], DEMO_ORDER[self.demo_idx]
        # A live reaction briefly overrides the mood baseline.
        if self.mode == "live":
            reaction = self.cube.active_reaction()
            if reaction is not None:
                return PRESETS[reaction], f"{reaction}!"
        v, a = ((self.manual_v, self.manual_a) if self.mode == "manual"
                else (self.cube.target_valence, self.cube.target_arousal))
        label = mood_label(AffectState(valence=v, arousal=a))
        return mood_to_config(v, a), f"{label}  v={v:+.2f} a={a:+.2f}"

    # --- drawing ------------------------------------------------------------

    def _draw_eye(self, cfg, lcx, lcy) -> None:
        cx = lcx + self.gaze[0] * GAZE_X
        cy = lcy + self.gaze[1] * GAZE_Y
        pts = eye_outline(cfg, cx, cy)
        flat = [coord * SCALE for xy in pts for coord in xy]
        if len(flat) >= 6:
            self.canvas.create_polygon(flat, fill=ON_COLOR, outline="")

    def _render(self) -> None:
        self.frame += 1
        if self.mode == "demo" and self.frame >= self.next_demo:
            self.demo_idx = (self.demo_idx + 1) % len(DEMO_ORDER)
            self.next_demo = self.frame + 75

        target_cfg, label = self._target_config()
        self.disp_cfg = lerp_config(self.disp_cfg, target_cfg, TWEEN)
        tag = {"demo": "demo", "manual": "manual"}.get(self.mode, "live")
        self.mood_var.set(f"{label}   [{tag}]")

        # Keep-alive blink.
        blink = 0.0
        if self.blink_phase >= 0:
            half = BLINK_FRAMES / 2
            blink = 1.0 - abs(self.blink_phase - half) / half
            self.blink_phase += 1
            if self.blink_phase > BLINK_FRAMES:
                self.blink_phase = -1
                self.next_blink = self.frame + random.randint(60, 180)
        elif self.frame >= self.next_blink:
            self.blink_phase = 0

        # Keep-alive saccades.
        if self.frame >= self.next_saccade:
            self.gaze_target = ([0.0, 0.0] if random.random() < 0.5
                                else [random.uniform(-1, 1) * 0.7, random.uniform(-1, 1) * 0.5])
            self.next_saccade = self.frame + random.randint(45, 150)
        self.gaze[0] += (self.gaze_target[0] - self.gaze[0]) * 0.25
        self.gaze[1] += (self.gaze_target[1] - self.gaze[1]) * 0.25

        cfg = with_blink(self.disp_cfg, blink)
        self.canvas.delete("all")
        self._draw_eye(cfg, EYE_CX_L, EYE_CY)
        self._draw_eye(mirror_config(cfg), EYE_CX_R, EYE_CY)
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
        self.root.focus_force()
        self.root.after(30, self._poll)
        self.root.after(FPS_MS, self._render)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Standalone Cozmo-style eyes device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cube-id", default="eyes_01")
    parser.add_argument("--demo", action="store_true",
                        help="cycle the 18 named expressions so you can see each one")
    args = parser.parse_args()
    win = EyesWindow(args.host, args.port, args.cube_id)
    if args.demo:
        win._set_mode("demo")
    win.run(args.host, args.port)


if __name__ == "__main__":
    main()
