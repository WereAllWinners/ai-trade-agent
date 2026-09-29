#!/bin/bash
# sprint04 F2.3 one-shot: run the discovery funnel replay near tomorrow's
# market open, to diff against today's 2026-07-08 midday run and test the
# partial-bar hypothesis (Jerome, 2026-07-08). Self-removes its own crontab
# line after running so it doesn't recur next year on the same date.
cd /home/zgx/personal-projects/ai-trade-agent || exit 1
OUT="logs/f2_3_morning_replay_$(date +%Y%m%d_%H%M%S).txt"
./venv/bin/python3 scripts/tools/replay_discovery_funnel.py > "$OUT" 2>&1
crontab -l 2>/dev/null | grep -v 'run_f2_3_morning_replay.sh' | crontab -
