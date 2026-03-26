#!/usr/bin/env python3
"""
Diagnostic script — run this first if anything breaks.
  python3 check.py
"""

import sys
import os
import socket

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ~"

issues = []

print("\n======================================")
print("  FinaceBro Diagnostic Check")
print("======================================\n")

# ── Python version ────────────────────────────────────────────────────────────
major, minor = sys.version_info[:2]
ver_str = f"{major}.{minor}.{sys.version_info.micro}"
if major == 3 and minor >= 9:
    print(f"{PASS} Python {ver_str}")
else:
    print(f"{FAIL} Python {ver_str}  (need 3.9+)")
    issues.append("Python version too old — install python3.9+ via brew")

# ── Required packages ─────────────────────────────────────────────────────────
print()
PKGS = [
    ("numpy",      "import numpy as np; print(np.__version__)"),
    ("pandas",     "import pandas as pd; print(pd.__version__)"),
    ("sklearn",    "import sklearn; print(sklearn.__version__)"),
    ("matplotlib", "import matplotlib; print(matplotlib.__version__)"),
    ("plotly",     "import plotly; print(plotly.__version__)"),
    ("dash",       "import dash; print(dash.__version__)"),
    ("pyarrow",    "import pyarrow; print(pyarrow.__version__)"),
    ("yfinance",   "import yfinance; print(yfinance.__version__)"),
]
for name, stmt in PKGS:
    try:
        exec(stmt)
        ver = eval(stmt.split(";")[1].strip().replace("print(", "").rstrip(")"))
        print(f"{PASS} {name:<12} {ver}")
    except Exception as e:
        if name == "yfinance":
            print(f"{WARN} {name:<12} not installed — demo mode only")
        else:
            print(f"{FAIL} {name:<12} {e}")
            issues.append(f"Missing package: {name}  →  pip3 install {name}")

# ── Local module imports ──────────────────────────────────────────────────────
print()
LOCAL = ["config", "demo_data", "simulation", "predictor", "report", "app"]
for mod in LOCAL:
    try:
        __import__(mod)
        print(f"{PASS} {mod}.py")
    except Exception as e:
        print(f"{FAIL} {mod}.py  — {e}")
        issues.append(f"Import error in {mod}.py: {e}")

# ── Port 8050 available? ──────────────────────────────────────────────────────
print()
try:
    s = socket.socket()
    s.settimeout(1)
    result = s.connect_ex(("127.0.0.1", 8050))
    s.close()
    if result == 0:
        print(f"{WARN} Port 8050 already in use — stop the other process or use a different port")
        issues.append("Port 8050 in use — kill existing process: lsof -ti:8050 | xargs kill")
    else:
        print(f"{PASS} Port 8050 is free")
except Exception as e:
    print(f"{WARN} Could not check port: {e}")

# ── Internet connectivity (for live data) ─────────────────────────────────────
try:
    s = socket.create_connection(("8.8.8.8", 53), timeout=2)
    s.close()
    print(f"{PASS} Internet reachable (live data available)")
except OSError:
    print(f"{WARN} No internet — use Demo mode in the dashboard")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if issues:
    print("Issues found:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()
    print("Fix the above, then run:  python3 app.py")
else:
    print("All checks passed — run:  python3 app.py")
    print("Then open:               http://localhost:8050")
print()
