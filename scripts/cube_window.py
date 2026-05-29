"""Standalone digital LED-matrix cube — a faithful 8x8 pixel simulator.

Run this and it connects to BlipShell's cube server on its own, exactly like a
real ESP32 cube powering on: it announces itself, then renders whatever actions
BlipShell sends. BlipShell auto-registers it the moment the socket arrives — no
/cube connect needed.

    # in BlipShell's config.yaml: robotics.enabled: true
    python -m scripts.cube_window                 # connects to 127.0.0.1:8765
    python -m scripts.cube_window --port 8765 --cube-id led_matrix_01

This is a real LED grid: everything is pixels. display_frame sets the 64 pixels,
clear blanks them, and display_text scrolls the string across the grid as a
pixel marquee using a built-in font (just like a real MAX7219-style module) —
there is no text "label", because real hardware has none. The buttons fire
sensor/system events for when we build the reactive (sensor) side later.
"""

import argparse
import asyncio
import json
import queue
import socket
import threading
import tkinter as tk

from blipshell.robotics.cubes import VirtualLEDMatrix

EVENT_BUTTONS = [
    "user_present", "user_absent",
    "speech_detected", "speech_ended",
    "thinking_started", "thinking_ended",
    "notification", "system_idle",
]

CELL = 36
PAD = 3
ON = "#33ff55"
OFF = "#1c1c1c"
SCROLL_MS = 90  # ms per column shift

# 5x7 pixel font (7 rows, 5 cols, '#' = lit). Real LED matrices carry a font
# like this in firmware; text is drawn by scrolling these glyphs across.
FONT = {
    " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "],
    "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
    "J": ["#####", "   # ", "   # ", "   # ", "#  # ", "#  # ", " ##  "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "# # #", "#  ##", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
    "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "S": [" ### ", "#   #", "#    ", " ### ", "    #", "#   #", " ### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    "3": ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
    ".": ["     ", "     ", "     ", "     ", "     ", "  ## ", "  ## "],
    ",": ["     ", "     ", "     ", "     ", "  ## ", "  ## ", " #   "],
    "!": ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  "],
    "?": [" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "],
    ":": ["     ", "  ## ", "  ## ", "     ", "  ## ", "  ## ", "     "],
    "-": ["     ", "     ", "     ", "#####", "     ", "     ", "     "],
    "'": ["  #  ", "  #  ", " #   ", "     ", "     ", "     ", "     "],
}
UNKNOWN = FONT["?"]


class CubeWindow:
    def __init__(self, host: str, port: int, cube_id: str):
        self.cube = VirtualLEDMatrix(cube_id=cube_id)   # state + metadata source
        self.loop = asyncio.new_event_loop()
        self.in_queue: queue.Queue = queue.Queue()
        self.send_lock = threading.Lock()
        self.sock: socket.socket | None = None
        self.sockfile = None
        # Scrolling-text state (the firmware's marquee).
        self.scrolling = False
        self.scroll_cols: list[int] = []
        self.scroll_pos = 0
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
                    x0, y0, x0 + CELL, y0 + CELL, fill=OFF, outline="#000")
                row.append(rect)
            self.cells.append(row)

        self.status_var = tk.StringVar(value=f"connecting to {host}:{port}...")
        tk.Label(self.root, textvariable=self.status_var, fg="#888", bg="#111",
                 font=("Consolas", 9)).pack(pady=(2, 6))

        btns = tk.Frame(self.root, bg="#111")
        btns.pack(padx=10, pady=(0, 10))
        for i, name in enumerate(EVENT_BUTTONS):
            b = tk.Button(btns, text=name, font=("Consolas", 8),
                          command=lambda n=name: self._fire(n))
            b.grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="ew")

    # --- rendering (everything is pixels) -----------------------------------

    def _render_frame(self, frame) -> None:
        for r in range(self.cube.height):
            for c in range(self.cube.width):
                on = frame[r][c] if (r < len(frame) and c < len(frame[r])) else 0
                self.canvas.itemconfig(self.cells[r][c], fill=ON if on else OFF)

    def _text_to_columns(self, text: str) -> list[int]:
        """Turn a string into a column buffer (each column = 8-bit pixel mask)."""
        cols = [0] * self.cube.width  # lead-in blanks so it scrolls in
        for ch in text.upper():
            glyph = FONT.get(ch, UNKNOWN)
            for x in range(5):
                mask = 0
                for r in range(7):
                    if glyph[r][x] == "#":
                        mask |= (1 << r)
                cols.append(mask)
            cols.append(0)  # 1-column gap between characters
        cols += [0] * self.cube.width  # tail blanks
        return cols

    def _start_scroll(self, text: str) -> None:
        self.scroll_cols = self._text_to_columns(text)
        self.scroll_pos = 0
        self.scrolling = bool(text)

    def _stop_scroll(self) -> None:
        self.scrolling = False

    def _scroll_tick(self) -> None:
        if self.scrolling and self.scroll_cols:
            for c in range(self.cube.width):
                idx = self.scroll_pos + c
                mask = self.scroll_cols[idx] if 0 <= idx < len(self.scroll_cols) else 0
                for r in range(self.cube.height):
                    on = (mask >> r) & 1
                    self.canvas.itemconfig(self.cells[r][c], fill=ON if on else OFF)
            self.scroll_pos += 1
            if self.scroll_pos > len(self.scroll_cols) - self.cube.width:
                self.scroll_pos = 0  # loop the marquee
        self.root.after(SCROLL_MS, self._scroll_tick)

    # --- networking ---------------------------------------------------------

    def connect(self, host: str, port: int) -> bool:
        try:
            self.sock = socket.create_connection((host, port), timeout=5)
        except OSError as e:
            self.status_var.set(f"could not connect to {host}:{port} — {e}. "
                                "Is BlipShell running with robotics.enabled?")
            return False
        # The 5s timeout was only for establishing the connection. Clear it so
        # reads block indefinitely — a passive cube may sit idle a long time
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
            action = msg.get("action", "")
            args = msg.get("args") or {}
            result = self.loop.run_until_complete(self.cube.invoke(action, args))
            if action == "display_text":
                self._start_scroll(str(args.get("text", "")))
            else:
                # display_frame / clear — both reflected in the cube's frame.
                self._stop_scroll()
                self._render_frame(self.cube.frame)
            self._send({"type": "result", "id": msg.get("id"), "result": result})
        elif kind == "_closed":
            self.status_var.set("disconnected from BlipShell")

    def run(self, host: str, port: int) -> None:
        self.connect(host, port)
        self._render_frame(self.cube.frame)
        self.root.after(30, self._poll)
        self.root.after(SCROLL_MS, self._scroll_tick)
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
