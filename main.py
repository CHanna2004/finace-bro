#!/usr/bin/env python3
"""
Commodity Futures Simulation & Prediction CLI

Usage examples:
  python main.py run --commodity gold
  python main.py run --commodity crude_oil --model jump_diffusion --days 60
  python main.py run --commodity corn --simulations 50000 --years 10
  python main.py run --commodity gold --ml-model ensemble --save-csv
  python main.py run --commodity gold --demo          # offline / no API key needed
  python main.py run --commodity gold --demo --plot   # show charts
  python main.py run --commodity gold --save-chart    # save chart to file
  python main.py list
  python main.py quote --commodity silver
  python main.py quote --demo
"""

import argparse
import logging
import sys

from config import COMMODITIES, DEFAULT_SIMULATIONS, DEFAULT_FORECAST_DAYS, DEFAULT_HISTORY_YEARS
from scraper import get_commodity_history
from current_data import fetch_current_commodity, fetch_multiple_current
from simulation import run_simulation
from predictor import predict_futures, ensemble_predict, FORECAST_HORIZONS
from report import (
    print_header,
    print_simulation_summary,
    print_ml_predictions,
    print_combined_outlook,
    save_csv,
)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ── Subcommand: list ──────────────────────────────────────────────────────────

def cmd_list(args) -> None:
    print("\nSupported Commodities:\n")
    print(f"  {'Key':<15} {'Name':<30} {'Ticker':<10} {'Unit'}")
    print(f"  {'-'*15} {'-'*30} {'-'*10} {'-'*15}")
    for key, meta in COMMODITIES.items():
        print(f"  {key:<15} {meta['name']:<30} {meta['ticker']:<10} {meta['unit']}")
    print()


# ── Subcommand: quote ─────────────────────────────────────────────────────────

def cmd_quote(args) -> None:
    commodities = args.commodity if args.commodity else list(COMMODITIES.keys())
    if getattr(args, "demo", False):
        from demo_data import generate_history, generate_current
        import pandas as pd
        rows = []
        for c in commodities:
            try:
                hist = generate_history(c)
                rows.append(generate_current(c, hist))
            except Exception as exc:
                logging.warning("Demo quote failed for %s: %s", c, exc)
        df = pd.DataFrame(rows).set_index("commodity") if rows else pd.DataFrame()
    else:
        df = fetch_multiple_current(commodities, force_refresh=args.refresh)
    if df.empty:
        print("No data retrieved.")
        return
    display_cols = ["name", "price", "unit", "change", "change_pct", "day_high", "day_low"]
    display_cols = [c for c in display_cols if c in df.columns]
    print("\nCurrent Prices:\n")
    print(df[display_cols].to_string())
    print()


# ── Subcommand: run ───────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    commodity = args.commodity

    # 1. Fetch historical data
    if args.demo:
        from demo_data import generate_history, generate_current
        print(f"\n[DEMO MODE] Generating {args.years} years of synthetic data for {commodity}...")
        hist = generate_history(commodity, years=args.years)
        print(f"  Generated {len(hist)} trading days of synthetic history.")
        print("Generating synthetic current price...")
        live = generate_current(commodity, hist)
    else:
        print(f"\nFetching {args.years} years of historical data for {commodity}...")
        hist = get_commodity_history(commodity, years=args.years, force_refresh=args.refresh)
        print(f"  Loaded {len(hist)} trading days of history.")
        print("Fetching current market price...")
        live = fetch_current_commodity(commodity, force_refresh=args.refresh)
    current_price = live["price"]

    # 3. Print header
    print_header(commodity, COMMODITIES[commodity], live)

    # 4. Monte Carlo simulation
    print(f"\nRunning {args.simulations:,} Monte Carlo simulations ({args.model})...")
    sim = run_simulation(
        historical_df=hist,
        current_price=current_price,
        forecast_days=args.days,
        n_simulations=args.simulations,
        model=args.model,
        seed=args.seed,
    )
    print_simulation_summary(sim)

    # 5. ML predictions
    pred_df = None
    if not args.no_ml:
        print(f"\nTraining ML models (this may take a moment)...")
        # Include the MC forecast horizon in the horizons list if not already there
        horizons = sorted(set(FORECAST_HORIZONS + [args.days]))
        if args.ml_model == "ensemble":
            pred_df = ensemble_predict(hist, current_price, horizons=horizons)
        else:
            pred_df = predict_futures(
                hist, current_price, horizons=horizons, model_type=args.ml_model
            )
        print_ml_predictions(pred_df, current_price)
    else:
        import pandas as pd
        pred_df = pd.DataFrame()

    # 6. Combined outlook
    print_combined_outlook(sim, pred_df)

    # 7. Optionally save CSV
    if args.save_csv:
        save_csv(sim, pred_df, commodity)

    # 8. Visualize
    if args.plot or args.save_chart:
        from plot import plot_simulation
        import os
        from datetime import datetime
        chart_path = None
        if args.save_chart:
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = f"output/{commodity}_chart_{ts}.png"
        plot_simulation(sim, pred_df, commodity, live, save_path=chart_path)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finace-bro",
        description="Commodity Futures Simulation & Prediction Tool",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all supported commodities")

    # quote
    q = sub.add_parser("quote", help="Show current prices for commodities")
    q.add_argument(
        "commodity", nargs="*",
        help="Commodity key(s) (default: all). E.g. gold crude_oil"
    )
    q.add_argument("--refresh", action="store_true", help="Force cache refresh")
    q.add_argument("--demo", action="store_true", help="Use synthetic data (no internet required)")

    # run
    r = sub.add_parser("run", help="Run simulation and prediction for a commodity")
    r.add_argument(
        "--commodity", "-c", required=True,
        choices=list(COMMODITIES.keys()),
        help="Commodity to analyse"
    )
    r.add_argument(
        "--model", "-m", default="gbm",
        choices=["gbm", "jump_diffusion"],
        help="Monte Carlo model (default: gbm)"
    )
    r.add_argument(
        "--ml-model", default="ensemble",
        choices=["rf", "gb", "ensemble"],
        help="ML model: rf=Random Forest, gb=Gradient Boosting, ensemble=both (default: ensemble)"
    )
    r.add_argument(
        "--days", "-d", type=int, default=DEFAULT_FORECAST_DAYS,
        help=f"Forecast horizon in trading days (default: {DEFAULT_FORECAST_DAYS})"
    )
    r.add_argument(
        "--simulations", "-n", type=int, default=DEFAULT_SIMULATIONS,
        help=f"Number of Monte Carlo paths (default: {DEFAULT_SIMULATIONS:,})"
    )
    r.add_argument(
        "--years", "-y", type=int, default=DEFAULT_HISTORY_YEARS,
        help=f"Years of historical data (default: {DEFAULT_HISTORY_YEARS})"
    )
    r.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    r.add_argument("--no-ml", action="store_true", help="Skip ML predictions (faster)")
    r.add_argument("--save-csv", action="store_true", help="Save results to CSV files")
    r.add_argument("--refresh", action="store_true", help="Force cache refresh")
    r.add_argument("--demo", action="store_true", help="Use synthetic data (no internet required)")
    r.add_argument("--plot", action="store_true", help="Show interactive chart window")
    r.add_argument("--save-chart", action="store_true", help="Save chart as PNG to output/")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    setup_logging(args.verbose)

    dispatch = {
        "list":  cmd_list,
        "quote": cmd_quote,
        "run":   cmd_run,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        if hasattr(args, "verbose") and args.verbose:
            raise
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
