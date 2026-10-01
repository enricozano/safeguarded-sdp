#!/usr/bin/env bash
# Optional: all experiments on every available core, detached from the terminal.
# Log: benchmark/logs/run_all.log; progress: python benchmark/hpc/status.py;
# stop: bash benchmark/hpc/stop.sh
#
#   bash benchmark/hpc/run_all.sh                 # uses $PYTHON (default: python3)
#   NCPU=16 bash benchmark/hpc/run_all.sh         # override the detected number of cores
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
NCPU="${NCPU:-$("$PY" -c 'import sys; sys.path.insert(0, "hpc"); import runner; print(runner.available_cpus())')}"
mkdir -p logs

# one single-threaded search per core; the two large OVRP instances instead share the
# cores between 6 concurrent searches for their multi-threaded dense linear algebra
nohup setsid bash -c "
  set -x
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  '$PY' hpc/runner.py logistics --problem asp --workers $NCPU --quiet
  '$PY' hpc/runner.py portfolio --workers $NCPU --quiet
  OMP_NUM_THREADS=$(( NCPU / 6 > 0 ? NCPU / 6 : 1 )) OPENBLAS_NUM_THREADS=$(( NCPU / 6 > 0 ? NCPU / 6 : 1 )) \
    '$PY' hpc/runner.py logistics --problem vrp --workers 6 --quiet
  '$PY' hpc/runner.py precision_scaling --workers $NCPU --budget-hours 12 --quiet
" > logs/run_all.log 2>&1 &
echo $! > logs/run_all.pid
echo "started on $NCPU cores (pid $!). progress: $PY benchmark/hpc/status.py  log: benchmark/logs/run_all.log"
