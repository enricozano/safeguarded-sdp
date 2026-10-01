#!/usr/bin/env bash
# Stop benchmark/hpc/run_all.sh and its workers. Finished searches are kept; running
# it again continues with the remaining ones.
cd "$(dirname "$0")/.."
PID="$(cat logs/run_all.pid 2>/dev/null)" || { echo "nothing to stop"; exit 0; }
kill -TERM -- "-$PID" 2>/dev/null && echo "stopped (process group $PID)" || echo "not running"
rm -f logs/run_all.pid
