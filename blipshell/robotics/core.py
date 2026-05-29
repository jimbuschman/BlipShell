"""RoboticsCore — the assembled robot brain the Agent holds.

Bundles the four pieces into one object and wires the reactive-provisioning
loop: when a cube connects, its actions become LLM tools (bridge) AND the LLM
authors a CapabilityProfile whose behaviors compile into the rules engine; when
it disconnects, both unwind. With no cubes connected this is inert — zero cost
at rest — so the Agent can always hold one.

Profile generation is optional: pass a generate_fn (wired to the LLM router) to
enable it. Without one, cubes still connect and their tools register; they just
get no auto-authored behaviors.
"""

import logging

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics.capability import CubeMetadata
from blipshell.robotics.cube import Cube
from blipshell.robotics.profile import CapabilityProfile, GenerateFn, ProfileGenerator
from blipshell.robotics.registry import CapabilityRegistry
from blipshell.robotics.rules import RulesEngine
from blipshell.robotics.tool_bridge import CubeToolBridge

logger = logging.getLogger(__name__)


class RoboticsCore:
    """Capability registry + tool bridge + rules engine + profile generation."""

    def __init__(self, tool_registry: ToolRegistry, generate_fn: GenerateFn | None = None):
        self.registry = CapabilityRegistry()
        self.bridge = CubeToolBridge(self.registry, tool_registry)
        self.bridge.attach()
        self.rules = RulesEngine(self.registry)
        self.profiles = ProfileGenerator(generate_fn) if generate_fn else None
        # cube_id -> the profile the LLM authored for it.
        self._profiles: dict[str, CapabilityProfile] = {}

        if self.profiles is not None:
            self.registry.add_connect_listener(self._author_profile)
            self.registry.add_disconnect_listener(self._drop_profile)

    async def connect(self, cube: Cube) -> CubeMetadata:
        """Connect a cube: registers tools, and (if enabled) authors behaviors."""
        return await self.registry.connect(cube)

    async def disconnect(self, cube_id: str) -> bool:
        """Disconnect a cube: unregisters tools and removes its behaviors."""
        return await self.registry.disconnect(cube_id)

    def get_profile(self, cube_id: str) -> CapabilityProfile | None:
        return self._profiles.get(cube_id)

    # --- reactive provisioning (LLM in the configuration loop) --------------

    async def _author_profile(self, meta: CubeMetadata) -> None:
        """Connect listener: ask the LLM to author behaviors for the new cube."""
        if self.profiles is None:
            return
        try:
            profile = await self.profiles.generate(meta, self.registry)
        except Exception as e:
            # An LLM hiccup must not break cube connection — tools are already
            # registered; the cube is usable, it just has no auto behaviors.
            logger.warning("Profile generation failed for '%s': %s", meta.cube_id, e)
            return
        self._profiles[meta.cube_id] = profile
        logger.info("Authored profile for '%s': %d behavior(s), role=%r",
                    meta.cube_id, len(profile.behaviors), profile.semantic_role)
        self._reload_behaviors()

    async def _drop_profile(self, meta: CubeMetadata) -> None:
        """Disconnect listener: forget the cube's profile and recompile rules."""
        if self._profiles.pop(meta.cube_id, None) is not None:
            self._reload_behaviors()

    def _reload_behaviors(self) -> None:
        """Recompile the union of all profiles' behaviors into the engine."""
        all_behaviors = [
            b for profile in self._profiles.values() for b in profile.behaviors
        ]
        self.rules.load(all_behaviors)
