"""Simulate how the mood-awareness line reads as a mood persists over time.

Answers the "does the trajectory carry weight at scale, or flatten out?" question:
prints the actual [Your state] line BlipShell would see, for a sustained mood, at
escalating durations (a minute → a month). No LLM, no GUI — pure string output
from the real _mood_awareness_text path.

    python -m scripts.sim_mood_trajectory
"""

import asyncio
import time

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import EmotionEngine, RoboticsCore
from blipshell.robotics.cubes import VirtualEyes

DURATIONS = [
    ("1 min", 60),
    ("15 min", 15 * 60),
    ("1 hour", 60 * 60),
    ("6 hours", 6 * 60 * 60),
    ("1 day", 24 * 60 * 60),
    ("3 days", 3 * 24 * 60 * 60),
    ("1 week", 7 * 24 * 60 * 60),
    ("1 month", 30 * 24 * 60 * 60),
]


class _Mini(ChatMixin):
    def __init__(self, robotics, emotion):
        self.robotics = robotics
        self.emotion = emotion
        self._mood_trend_clause = ""


async def _scenario(core, valence, arousal, trend, title):
    em = EmotionEngine()
    em.state.valence = valence
    em.state.arousal = arousal
    agent = _Mini(core, em)
    agent._mood_trend_clause = trend
    print(f"\n== {title} ==")
    for label, secs in DURATIONS:
        agent._mood_label_since = time.time() - secs
        print(f"  held {label:>8}: {agent._mood_awareness_text()}")


async def main():
    core = RoboticsCore(ToolRegistry())
    await core.connect(VirtualEyes())  # a connected "face" so the mood surfaces
    await _scenario(core, 0.0, -0.8, "", "Sustained low energy (sleepy)")
    await _scenario(core, 0.5, -0.3, "", "Sustained contentment (baseline-ish)")
    await _scenario(core, -0.6, 0.6, ", and it's sinking", "Sustained agitation, sinking")
    print("\nNote: in practice the mood DECAYS toward baseline, so a non-baseline")
    print("state only persists this long with sustained events or a shifted baseline.")
    print("The long-duration language mostly attaches to the resting (baseline) mood.")


if __name__ == "__main__":
    asyncio.run(main())
