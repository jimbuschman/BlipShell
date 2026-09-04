#!/usr/bin/env bash
# Deep agentic sweep over Pi-realistic LOCAL models, run on the Ollama PC via
# Tailscale. Capability is hardware-independent, so this answers "can a model
# this size hold the agent loop" without needing the Pi itself. Only speed has
# to be measured on real hardware later.
#
# Sizes are the on-disk footprint; a Pi 5 8GB realistically wants <=4GB of
# weights to leave room for the KV cache.
# Machine-specific values come from the environment (repo is public — never
# hardcode the Tailscale endpoint or a local working directory here).
cd "${DEEPTEST_DIR:?set DEEPTEST_DIR to the deeptest working directory}" || exit 1

export DEEPTEST_URL="${DEEPTEST_URL:?set DEEPTEST_URL to your Ollama endpoint, e.g. http://localhost:11434}"
export DEEPTEST_TIMEOUT=240
export PYTHONIOENCODING=utf-8

# smallest first, so the cheapest signal lands soonest
MODELS="gemma3:1b qwen2.5:1.5b qwen3:1.7b qwen2.5:3b phi4-mini:latest qwen3:4b qwen3.5:4b gemma3:4b lfm2.5:latest"

for m in $MODELS; do
  echo "############ START $m $(date -u +%H:%M:%S) ############"
  python run.py "$m" 3 > "run_pi_$(echo "$m" | tr ':/' '__').log" 2>&1
  echo "############ DONE  $m rc=$? $(date -u +%H:%M:%S) ############"
  tail -12 "run_pi_$(echo "$m" | tr ':/' '__').log"
done
echo "PI SWEEP COMPLETE"
