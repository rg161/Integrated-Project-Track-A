#!/usr/bin/env python3
"""
Run script for PartD nodes.
Starts ekf.py, Controller.py and trajectory.py in background for local testing.
"""
import subprocess
import time

procs = []

try:
    scripts = ['ekf.py', 'Controller.py', 'trajectory.py']
    for s in scripts:
        print('Starting', s)
        p = subprocess.Popen(['python3', s])
        procs.append(p)
        time.sleep(0.5)
    print('All nodes started. Press Ctrl-C to stop.')
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Shutting down...')
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(1)
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
