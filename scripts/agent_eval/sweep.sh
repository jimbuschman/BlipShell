#!/usr/bin/env bash
# Deep agentic sweep for the BlipShell MAIN (tool-calling) model.
#
# Phase 1 = free tier. These are the only models that still work after the paid
# plan lapses on the 27th, so this is the answer that has to hold.
# Phase 2 = paid-only models, to price what upgrading would actually buy.
cd "C:/Windows/TEMP/claude/C--Users-[user]-source-repos-jimbuschman-BlipShell/96b97df7-8444-4a6d-9d97-26ee6bb0f1e0/scratchpad/deeptest" || exit 1

FREE="minimax-m3:cloud nemotron-3-ultra:cloud nemotron-3-super:cloud nemotron-3-nano:30b-cloud gpt-oss:120b-cloud gpt-oss:20b-cloud gemma4:31b-cloud gemma4:cloud"
PAID="glm-5.2:cloud kimi-k2.7-code:cloud minimax-m2.7:cloud deepseek-v4-flash:cloud qwen3.5:397b-cloud mistral-large-3:675b-cloud"

run_group () {
  local label="$1"; shift
  for m in $@; do
    echo "############ START [$label] $m $(date -u +%H:%M:%S) ############"
    python run.py "$m" 3 2>&1 | grep -v "^WARNING\|^INFO"
    echo "############ DONE  [$label] $m $(date -u +%H:%M:%S) ############"
  done
}

run_group FREE $FREE
echo "PHASE1 COMPLETE"
run_group PAID $PAID
echo "SWEEP COMPLETE"
