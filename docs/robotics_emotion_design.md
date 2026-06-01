# Emotion & Expression Design — research-grounded reference

Design reference for BlipShell's affective/embodiment layer, grounded in research
on how Anki's Cozmo/Vector and academic affective robots work. Captured 2026-06.

> **Source-strength caveat.** Anki never published Cozmo/Vector's emotion-engine
> internals (dimensions, decay constants, mood-update rules). Those are from trade
> press + an Anki-animator interview + fan reverse-engineering (randym32), which
> itself flags uncertainty. The *procedural eye* details are high-confidence (real
> code: `pycozmo`, `esp32-eyes`, `ggldnl/Procedural-Expression-Library`). The one
> rigorous numeric mood/emotion formalism is from the **academic Mini robot**
> (peer-reviewed), NOT Cozmo — treat it as a transferable pattern.

## The architecture: three layers

```
keep-alive   (always on: blinks, saccades, micro-moves)   ← independent of the LLM
   over
mood         (slow, persistent valence/arousal baseline)  ← BlipShell's EmotionEngine
   with
reactions    (fast, event-triggered, decay ~100s, variants) ← brief overrides
```

- **Mood vs reactions is confirmed** as the right model. Vector explicitly separates
  short-term *emotions* (stimulus-driven, brief) from a stable *mood* (slow). Reactions
  layer **over** the mood baseline, then settle back.
- **Keep-alive is the biggest aliveness trick** and runs **continuously, independent of
  mood, reactions, and the LLM**. Vector's eyes scan/blink on their own, decoupled from
  perception. Crucial for us: eye liveliness must not stall on LLM latency.

## Mood + reaction formalism (from the Mini robot — transferable pattern)

- Mood and emotion are kept in **two separate valence/arousal spaces** (each axis
  −100..100), then **blended** — *not* one shared space. (Our current EmotionEngine is
  one space = the mood; add a second fast layer for reactions.)
- Transient emotions **rise sharply** on an event then **decay exponentially**:
  `e(t) = 100·e^(−0.05·t)` → ~100 s to fully decay. (We already have exp decay; reactions
  just use a much faster tau than mood.)
- A **dominant emotion** = the single emotion above a ~20-unit threshold (else "none").
- Final affective state blends **dominant emotion + active mood** (`AE = {de, cm}`),
  mapped to outputs via **"modulation profiles."**
- **Mood timescale is the least certain quantity** — the one published ~2h figure was
  *refuted*. Tune mood decay by feel; don't treat any half-life as established.

## Reactions = named families with variants (Cozmo "Animation Triggers")

Cozmo's control hierarchy: Actions → Animations → **Animation Triggers** (named families,
~544) → Behaviors. Playing a trigger **picks one variant at random (or mood-weighted)** —
that randomness *is* the aliveness. So a reaction (e.g. `delight`) should be a *family of
variants*, not one fixed expression.

## Procedural eyes (HIGH confidence — real code)

- **Never pre-rendered frames.** Fully parametric, animated by **linear interpolation
  (tweening)** between keyframe parameter arrays: `y = from[i] + x·(to[i] − from[i])`, x:0→1.
- Cozmo's face = **43 floats**: 5 face-level (center_x, center_y, scale_x, scale_y, angle)
  + 19 per eye. Per eye = corner-curvature radii (lower/upper × inner/outer, x&y) + upper/
  lower lids (y, angle, bend). Renders on 128×64; default eye 28×40 px.
- **Authoring is simplified to a few continuous axes** (mad↔happy, worry↔curiosity, blink
  rate, gaze) that interpolate between endpoint presets → maps directly onto our
  valence/arousal.
- **Blink** = collapse vertical scale to ~0 while widening horizontal, over ~5–6 frames
  (`BLINK_SCALE_Y≈0.01`, `BLINK_SCALE_X≈15`, `BLINK_STEPS≈FRAME_RATE/6`).
- **Keep-alive** = random blinks + saccade micro-moves, each on its **own independent
  timer** (`esp32-eyes`: RandomBlink + RandomLook, default on).
- **Simpler alternative** (`ggldnl`): each eye = an **8-point polygon in normalized [0,1]
  space**; transition by interpolating each point toward a target polygon; blink =
  interpolate to a thin rectangle and back. Easier first cut than corner-radii.

## Decisions for BlipShell

| Decision | Choice |
|---|---|
| Architecture | 3 layers: keep-alive (always on) / mood (slow) / reactions (transient) |
| Mood | existing EmotionEngine (valence/arousal, exp decay) — slow tau |
| Reactions | second fast affective layer; decay ~100 s; named families with variants |
| Blend | dominant reaction (above threshold) overrides/blends with mood for the face |
| Eyes | procedural, tweened; **start with the 8-point polygon model**, valence/arousal → preset endpoints |
| Keep-alive | always-on blink + saccade loop, independent of LLM/mood |
| LLM → display | LLM sets continuous mood axes + may fire a named reaction; never draws pixels |
| Display-only? | mood currently does NOT touch responses; "LLM knows its mood" (tone) is a separate opt-in |

## Must decide ourselves (unpublished by Anki)
- Our own emotion dimensions/decay constants (valence/arousal is fine).
- Mood drift timescale (tune by feel — no established figure).
- Arbitration when keep-alive / reaction / mood conflict (reaction overrides mood; keep-alive
  runs underneath — blinks happen during any expression).

## Not reused: cozmo-explorer
Separate project (mapping/surveying robot; real Cozmo SDK animations; mood-string + journal,
no valence/arousal engine, no procedural eyes). Confirms the Animation-Trigger pattern; no
direct code reuse for this layer.

## Key sources
- Mini robot affective formalism (peer-reviewed): https://link.springer.com/article/10.1007/s12369-022-00915-9
- pycozmo procedural face (reverse-engineered ref): https://github.com/zayfod/pycozmo
- esp32-eyes (procedural, Cozmo-inspired): https://github.com/playfultechnology/esp32-eyes
- ggldnl Procedural-Expression-Library (8-point polygon): https://github.com/ggldnl/Procedural-Expression-Library
- Vector character study (mood vs emotion): https://randym32.github.io/Anki.Vector.Documentation/guides/Vector%20character%20study.html
- CMU 15-494 (Cozmo control hierarchy): https://www.cs.cmu.edu/afs/cs/academic/class/15494-s18/lectures/arch+python/arch+python.pdf
