"""Standalone digital LED-matrix cube — a separate process with its own window.

Run this and it connects to BlipShell's cube server on its own, exactly like a
real ESP32 cube powering on: it announces itself, then renders whatever actions
BlipShell sends and pushes back any events you trigger. BlipShell auto-registers
it the moment the socket arrives — no /cube connect needed.

    # in BlipShell's config.yaml: robotics.enabled: true
    python -m scripts.cube_window                 # connects to 127.0.0.1:8765
    python -m scripts.cube_window --port 8765 --cube-id led_matrix_01

The window shows the 8x8 grid (lit pixels + any text cue) and a row of buttons
that fire sensor/system events so you can watch the LLM-authored behaviors react
in real time. This is the full cycle, in software, before any hardware exists.
"""

import argparse
import asyncio
import json
import queue
import socket
import threading
import tkinter as tk

from blipshell.robotics.cubes import VirtualLEDMatrix

# Events you can fire from the window (mirrors CORE_EVENTS + a couple sensors).
EVENT_BUTTONS = [
    "user_present", "user_absent",
    "speech_detected", "speech_ended",
    "thinking_started", "thinking_ended",
    "notification", "system_idle",
]

CELL = 36   # pixel size of each LED
PAD = 3


class CubeWindow:
    def __init__(self, host: str, port: int, cube_id: str):
        self.cube = VirtualLEDMatrix(cube_id=cube_id)  # state + metadata source
        self.loop = asyncio.new_event_loop()           # drives the async invoke()
        self.in_queue: queue.Queue = queue.Queue()
        self.send_lock = threading.Lock()
        self.sock: socket.socket | None = None
        self.sockfile = None
        self._build_ui(host, port)

    # --- UI -----------------------------------------------------------------

    def _build_ui(self, host: str, port: int) -> None:
        self.root = tk.Tk()
        self.root.title(f"Cube — {self.cube.cube_id} ({self.cube.width}x{self.cube.height})")
        self.root.configure(bg="#111")

        w = self.cube.width * (CELL + PAD) + PAD
        h = self.cube.height * (CELL + PAD) + PAD
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="#111",
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.cells = []
        for r in range(self.cube.height):
            row = []
            for c in range(self.cube.width):
                x0 = PAD + c * (CELL + PAD)
                y0 = PAD + r * (CELL + PAD)
                rect = self.canvas.create_rectangle(
                    x0, y0, x0 + CELL, y0 + CELL, fill="#1c1c1c", outline="#000")
                row.append(rect)
            self.cells.append(row)

        self.text_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.text_var, fg="#0f0", bg="#111",
                 font=("Consolas", 16, "bold")).pack()

        self.status_var = tk.StringVar(value=f"connecting to {host}:{port}...")
        tk.Label(self.root, textvariable=self.status_var, fg="#888", bg="#111",
                 font=("Consolas", 9)).pack(pady=(2, 6))

        btns = tk.Frame(self.root, bg="#111")
        btns.pack(padx=10, pady=(0, 10))
        for i, name in enumerate(EVENT_BUTTONS):
            b = tk.Button(btns, text=name, font=("Consolas", 8),
                          command=lambda n=name: self._fire(n))
            b.grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="ew")

    def _render(self) -> None:
        for r in range(self.cube.height):
            for c in range(self.cube.width):
                on = self.cube.frame[r][c]
                self.canvas.itemconfig(self.cells[r][c], fill="#33ff55" if on else "#1c1c1c")
        self.text_var.set(self.cube.last_text or "")

    # --- networking ---------------------------------------------------------

    def connect(self, host: str, port: int) -> bool:
        try:
            self.sock = socket.create_connection((host, port), timeout=5)
        except OSError as e:
            self.status_var.set(f"could not connect to {host}:{port} — {e}. "
                                "Is BlipShell running with robotics.enabled?")
            return False
        # The 5s timeout was only for establishing the connection. Clear it so
        # reads block indefinitely — a passive cube may sit idle for a long time
        # between actions, and a read timeout would look like a disconnect.
        self.sock.settimeout(None)
        self.sockfile = self.sock.makefile("rb")
        self._send({"type": "hello", "metadata": self.cube.describe().model_dump()})
        self.status_var.set(f"connected to {host}:{port} as {self.cube.cube_id}")
        threading.Thread(target=self._recv_loop, daemon=True).start()
        return True

    def _send(self, msg: dict) -> None:
        if self.sock is None:
            return
        data = (json.dumps(msg) + "\n").encode("utf-8")
        try:
            with self.send_lock:
                self.sock.sendall(data)
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

    def _fire(self, event_name: str) -> None:
        self._send({"type": "event", "name": event_name, "payload": {}})

    # --- main-thread event pump --------------------------------------------

    def _poll(self) -> None:
        while True:
            try:
                msg = self.in_queue.get_nowait()
            except queue.Empty:
                break
            self._handle(msg)
        self.root.after(30, self._poll)

    def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "invoke":
            result = self.loop.run_until_complete(
                self.cube.invoke(msg.get("action", ""), msg.get("args") or {}))
            self._render()
            self._send({"type": "result", "id": msg.get("id"), "result": result})
        elif kind == "_closed":
            self.status_var.set("disconnected from BlipShell")

    def run(self, host: str, port: int) -> None:
        self.connect(host, port)
        self._render()
        self.root.after(30, self._poll)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Standalone digital LED-matrix cube")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cube-id", default="led_matrix_01")
    args = parser.parse_args()
    CubeWindow(args.host, args.port, args.cube_id).run(args.host, args.port)


if __name__ == "__main__":
    main()
