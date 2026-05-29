"""Core event vocabulary the robot emits regardless of which cubes are present.

Input cubes advertise their own events (a motion sensor emits "motion_detected"),
but the core itself raises a baseline set so that even an output-only setup
(just an LED matrix) has meaningful triggers to bind behaviors to. These are the
trigger names the LLM may use in a behavior in addition to any connected cube's
advertised events.
"""

# name -> what it means, shown to the LLM when it authors behaviors.
CORE_EVENTS: dict[str, str] = {
    "user_present": "a user has just been detected nearby / arrived",
    "user_absent": "the user appears to have left",
    "speech_detected": "the user started speaking (good moment to show a listening cue)",
    "speech_ended": "the user stopped speaking",
    "thinking_started": "the assistant began processing a request",
    "thinking_ended": "the assistant finished processing",
    "notification": "a notification should be surfaced to the user",
    "system_idle": "no activity for a while",
}
