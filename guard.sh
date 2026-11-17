#!/bin/bash
# guard.sh PID LOG CAP_SECONDS: SIGSTOP while Song Forge has jobs_running>0, stop if system free memory <15%,
# stop at the run-time cap (paused time not counted).
P=$1; LOG=$2; CAP=${3:-600}; ran=0; paused=0
while kill -0 $P 2>/dev/null; do
  jr=$(curl -s -m 3 127.0.0.1:8767/api/status | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin).get('jobs_running',0))" 2>/dev/null || echo 0)
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2);print $2}')
  if [ "${free:-100}" -lt 15 ]; then echo "GUARD $(date +%T) free ${free}% <15%, stopping" >> $LOG; kill -CONT $P; kill -INT $P; sleep 10; kill -9 $P 2>/dev/null; exit; fi
  if [ "${jr:-0}" -gt 0 ] && [ $paused = 0 ]; then kill -STOP $P; paused=1; echo "GUARD $(date +%T) PAUSE songforge jobs_running=$jr" >> $LOG; fi
  if [ "${jr:-0}" = 0 ] && [ $paused = 1 ]; then kill -CONT $P; paused=0; echo "GUARD $(date +%T) RESUME" >> $LOG; fi
  [ $paused = 0 ] && ran=$((ran+2))
  if [ $ran -ge $CAP ]; then echo "GUARD $(date +%T) cap ${CAP}s reached, stopping" >> $LOG; kill -INT $P; sleep 10; kill -9 $P 2>/dev/null; exit; fi
  sleep 2
done
