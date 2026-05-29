"""RoboticsCore — the assembled robot brain the Agent holds.

Bundles the pieces and wires the reactive-provisioning loop: when a cube
connects, its actions become LLM tools (bridge) AND it gets a CapabilityProfile
whose behaviors compile into the rules engine; when it disconnects, both unwind.
With no cubes connected this is inert — zero cost at rest.

Profiles persist across connects (keyed by cube *type*) so a cube has a
consistent body language instead of being re-authored from scratch every time.
On connect: reuse the stored profile if one exists (no LLM call), else author a
fresh one and save it. Reuse re-validates behaviors against the actual connected
cube, so a stored profile can't drive an action the cube no longer exposes.
"""

import json
import logging
from typing import Awaitable, Callable

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics.capability import CubeMetadata
from blipshell.robotics.cube import Cube
from blipshell.robotics.profile import CapabilityProfile, GenerateFn, ProfileGenerator
from blipshell.robotics.registry import CapabilityRegistry
from blipshell.robotics.rules import RulesEngine
from blipshell.robotics.tool_bridge import CubeToolBridge

logger = logging.getLogger(__name__)

# Persistence callbacks, keyed by a string (the cube type). load returns the
# stored JSON (or None); save stores it.
LoadProfileFn = Callable[[str], Awaitable[str | None]]
SaveProfileFn = Callable[[str, str], Awaitable[None]]


class RoboticsCore:
    """Capability registry + tool bridge + rules engine + persistent profiles."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        generate_fn: GenerateFn | None = None,
        load_profile_fn: LoadProfileFn | None = None,
        save_profile_fn: SaveProfileFn | None = None,
    ):
        self.registry = CapabilityRegistry()
        self.bridge = CubeToolBridge(self.registry, tool_registry)
        self.bridge.attach()
        self.rules = RulesEngine(self.registry)
        self.profiles = ProfileGenerator(generate_fn) if generate_fn else None
        self._load_profile = load_profile_fn
        self._save_profile = save_profile_fn
        # cube_id -> the active profile for it.
        self._profiles: dict[str, CapabilityProfile] = {}

        if self.profiles is not None or self._load_profile is not None:
            self.registry.add_connect_listener(self._author_profile)
            self.registry.add_disconnect_listener(self._drop_profile)

    async def connect(self, cube: Cube) -> CubeMetadata:
        """Connect a cube: registers tools, and loads/authors its behaviors."""
        return await self.registry.connect(cube)

    async def disconnect(self, cube_id: str) -> bool:
        """Disconnect a cube: unregisters tools and removes its behaviors."""
        return await self.registry.disconnect(cube_id)

    def get_profile(self, cube_id: str) -> CapabilityProfile | None:
        return self._profiles.get(cube_id)

    # --- provisioning: reuse stored profile, else author fresh --------------

    async def _author_profile(self, meta: CubeMetadata) -> None:
        """Connect listener: load the cube type's stored profile, or author one."""
        try:
            profile = await self._load_or_author(meta)
        except Exception as e:
            # Never break cube connection — tools are already registered.
            logger.warning("Profile load/author failed for '%s': %s", meta.cube_id, e)
            return
        if profile is None:
            return
        self._profiles[meta.cube_id] = profile
        self._reload_behaviors()

    async def _load_or_author(self, meta: CubeMetadata) -> CapabilityProfile | None:
        """Reuse the stored profile for this cube type, or author + save a new one."""
        key = meta.module_type

        # 1) Try the stored profile.
        if self._load_profile is not None:
            stored = await self._load_profile(key)
            if stored:
                profile = self._profile_from_stored(stored, meta)
                if profile is not None:
                    logger.info("Reused stored profile for '%s' (type %s): %d behavior(s)",
                                meta.cube_id, key, len(profile.behaviors))
                    return profile

        # 2) No usable stored profile — author fresh (needs the LLM).
        if self.profiles is None:
            return None
        profile = await self.profiles.generate(meta, self.registry)
        if self._save_profile is not None:
            try:
                await self._save_profile(key, profile.model_dump_json())
            except Exception as e:
                logger.warning("Failed to save profile for type %s: %s", key, e)
        logger.info("Authored new profile for '%s' (type %s): %d behavior(s)",
                    meta.cube_id, key, len(profile.behaviors))
        return profile

    def _profile_from_stored(self, stored: str, meta: CubeMetadata) -> CapabilityProfile | None:
        """Rebuild a profile from stored JSON, re-validating against this cube.

        Behaviors that reference an action the connected cube no longer exposes
        are dropped — a stored profile can't drive a stale capability.
        """
        try:
            data = json.loads(stored)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Stored profile for type %s is unreadable: %s", meta.module_type, e)
            return None
        raw_behaviors = data.get("behaviors", [])
        # The profile is stored per type; its behaviors targeted whatever
        # instance first authored it. Remap targets to this connecting cube so
        # the same learned behaviors drive a different instance of the type.
        # (Safe for single-cube profiles; multi-cube coordination would need
        # richer remapping.)
        for b in raw_behaviors:
            if isinstance(b, dict):
                for a in b.get("actions", []):
                    if isinstance(a, dict):
                        a["target"] = meta.cube_id
        behaviors = ProfileGenerator._validate_behaviors(raw_behaviors, self.registry)
        return CapabilityProfile(
            cube_id=meta.cube_id,
            semantic_role=str(data.get("semantic_role", "")),
            intended_uses=[str(u) for u in data.get("intended_uses", []) if u],
            usage_guidance=str(data.get("usage_guidance", "")),
            behaviors=behaviors,
        )

    async def reauthor(self, cube_id: str) -> CapabilityProfile | None:
        """Force a fresh authoring for a connected cube, replacing the stored one.

        Useful while tuning (e.g. after changing the cube's description). Ignores
        any stored profile, re-runs the LLM, saves the result, and reloads.
        """
        meta = self.registry.get_metadata(cube_id)
        if meta is None or self.profiles is None:
            return None
        profile = await self.profiles.generate(meta, self.registry)
        if self._save_profile is not None:
            try:
                await self._save_profile(meta.module_type, profile.model_dump_json())
            except Exception as e:
                logger.warning("Failed to save re-authored profile: %s", e)
        self._profiles[cube_id] = profile
        self._reload_behaviors()
        logger.info("Re-authored profile for '%s': %d behavior(s)",
                    cube_id, len(profile.behaviors))
        return profile

    async def _drop_profile(self, meta: CubeMetadata) -> None:
        """Disconnect listener: forget the cube's active profile and recompile."""
        if self._profiles.pop(meta.cube_id, None) is not None:
            self._reload_behaviors()

    def _reload_behaviors(self) -> None:
        """Recompile the union of all active profiles' behaviors into the engine."""
        all_behaviors = [
            b for profile in self._profiles.values() for b in profile.behaviors
        ]
        self.rules.load(all_behaviors)
