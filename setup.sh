#!/bin/bash
# FinaceBro — One-shot setup script for macOS
# Run: bash setup.sh

set -e

PYTHON=""
VENV_DIR=".venv"

echo ""
echo "========================================"
echo "  FinaceBro Setup"
echo "========================================"

# ── 1. Find Python 3.9+ ──────────────────────────────────────────────────────
for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON="$cmd"
            echo ""
            echo "  ✓ Found Python: $cmd ($VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "  ✗ Python 3.9+ not found."
    echo ""
    echo "  Install it with:"
    echo "    brew install python"
    echo ""
    echo "  (If brew is missing: https://brew.sh)"
    exit 1
fi

# ── 2. Create virtual environment ────────────────────────────────────────────
echo ""
echo "  Creating virtual environment in $VENV_DIR/ ..."
"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "  ✓ Virtual environment ready"

# ── 3. Upgrade pip silently ───────────────────────────────────────────────────
echo ""
echo "  Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel

# ── 4. Install packages with Python-version-aware pinning ────────────────────
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo ""
echo "  Installing packages for Python $PYVER..."
echo ""

install_pkg() {
    local PKG="$1"
    echo -n "    $PKG ... "
    if pip install --quiet "$PKG"; then
        echo "✓"
    else
        echo "FAILED"
        echo "    Retrying with --no-build-isolation ..."
        pip install --quiet --no-build-isolation "$PKG" || echo "    ✗ Could not install $PKG"
    fi
}

# Core numerical / ML
install_pkg "numpy"
install_pkg "pandas"
install_pkg "pyarrow"
install_pkg "scikit-learn"
install_pkg "scipy"

# Visualisation / web
install_pkg "matplotlib"
install_pkg "plotly"
install_pkg "dash"

# Data fetching — yfinance has a tricky dependency (multitasking)
echo -n "    yfinance ... "
pip install --quiet yfinance 2>/dev/null && echo "✓" || {
    echo "retrying..."
    pip install --quiet "setuptools>=70" 2>/dev/null
    pip install --quiet yfinance 2>/dev/null && echo "    ✓ yfinance" || echo "    ✗ yfinance failed — demo mode will still work"
}

# ── 5. Verify imports ─────────────────────────────────────────────────────────
echo ""
echo "  Verifying imports..."
"$VENV_DIR/bin/python" - <<'EOF'
issues = []
for mod in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "plotly", "dash"]:
    try:
        __import__(mod)
        print(f"    ✓ {mod}")
    except ImportError as e:
        print(f"    ✗ {mod}: {e}")
        issues.append(mod)
try:
    import yfinance
    print(f"    ✓ yfinance")
except ImportError:
    print(f"    ~ yfinance unavailable (demo mode only)")

if issues:
    print(f"\n  WARNING: some imports failed: {issues}")
else:
    print("\n  All core imports OK")
EOF

# ── 6. Write a launch script ──────────────────────────────────────────────────
cat > run.sh <<'RUNEOF'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/.venv/bin/activate"
echo ""
echo "  Starting FinaceBro..."
echo "  Open: http://localhost:8050"
echo "  Press Ctrl+C to stop."
echo ""
python "$DIR/app.py"
RUNEOF
chmod +x run.sh

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  To start the dashboard:"
echo "    bash run.sh"
echo ""
echo "  Then open: http://localhost:8050"
echo "========================================"
echo ""
