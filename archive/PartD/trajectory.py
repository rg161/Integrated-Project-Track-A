#!/usr/bin/env python3
"""Convenience launcher for the Part D ROS nodes.

This script starts the estimation, controller, and trajectory publisher nodes in
parallel so they can be tested together locally without launching each node by
hand.
"""
import subprocess
import time

procs = []

try:
    """Start the required helper nodes for local testing."""
    scripts = ['PartC/Estimation.py', 'Controller.py', 'trajectory.py']
    for s in scripts:
        print('Starting', s)
        p = subprocess.Popen(['python3', s])
        procs.append(p)
        time.sleep(0.5)

    print('All nodes started. Press Ctrl-C to stop.')

    while True:
        time.sleep(1)
except KeyboardInterrupt:
    """Shut down launched processes on Ctrl-C."""
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
