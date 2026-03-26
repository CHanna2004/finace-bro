"""
Commodity Futures Simulation — Localhost Web Dashboard
Run with: python3 app.py
Then open: http://localhost:8050
"""

import math
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COMMODITIES, DEFAULT_SIMULATIONS, DEFAULT_FORECAST_DAYS, DEFAULT_HISTORY_YEARS
from demo_data import generate_history, generate_current
from simulation import run_simulation
from predictor import ensemble_predict, predict_futures, FORECAST_HORIZONS

# ── App setup ─────────────────────────────────────────────────────────────────

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

app = dash.Dash(
    __name__,
    title="FinaceBro — Commodity Futures",
    suppress_callback_exceptions=True,
    assets_folder=_os.path.join(_HERE, "assets"),
)

# Inject CSS directly into the HTML — works on every Dash version / OS
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  * { box-sizing: border-box; }
  body, #react-entry-point {
    background: #0d0d1a !important;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    color: #e8e8f8 !important;
    margin: 0;
  }
  /* Dropdown */
  .Select-control { background-color: #1a1a35 !important; border-color: #2a2a4a !important; }
  .Select-value-label, .Select-placeholder { color: #ffffff !important; font-size: 14px !important; }
  .Select-menu-outer { background-color: #1a1a35 !important; border-color: #2a2a4a !important; z-index: 999 !important; }
  .Select-option { background-color: #1a1a35 !important; color: #e8e8f8 !important; }
  .Select-option:hover, .Select-option.is-focused { background-color: #2a2a55 !important; color: #fff !important; }
  .Select-option.is-selected { background-color: #4488ff !important; color: #fff !important; }
  .Select-arrow { border-top-color: #9999cc !important; }
  /* Radio labels */
  label { color: #e8e8f8 !important; font-family: system-ui, -apple-system, sans-serif !important; }
  /* Slider marks */
  .rc-slider-mark-text { color: #e8e8f8 !important; font-size: 12px !important; }
  .rc-slider-track { background-color: #4488ff !important; }
  .rc-slider-handle { border-color: #4488ff !important; background-color: #4488ff !important; }
  .rc-slider-rail { background-color: #2a2a4a !important; }
  .rc-slider-tooltip-inner { background-color: #4488ff !important; color: #fff !important; font-size: 12px !important; font-weight: 600 !important; }
  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0d0d1a; }
  ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #4488ff; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

DARK_BG    = "#0d0d1a"
PANEL_BG   = "#13132a"
BORDER     = "#2a2a4a"
TEXT       = "#e8e8f8"
MUTED      = "#9999cc"
LABEL      = "#ffffff"
ACCENT     = "#4488ff"
GREEN      = "#33cc88"
RED        = "#ff4466"
YELLOW     = "#ffcc44"

SIDEBAR_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LABEL_STYLE  = {"fontSize": "13px", "fontWeight": "600", "color": LABEL,
                "letterSpacing": "0.5px", "fontFamily": SIDEBAR_FONT,
                "marginBottom": "6px", "display": "block"}
RADIO_LABEL  = {"display": "block", "marginTop": "8px", "fontSize": "14px",
                "color": TEXT, "fontFamily": SIDEBAR_FONT, "cursor": "pointer"}

COMMODITY_OPTIONS = [
    {"label": f"{meta['name']} ({key})", "value": key}
    for key, meta in COMMODITIES.items()
]


def card(children, style=None):
    base = {
        "background": PANEL_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "20px",
        "marginBottom": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


# ── Layout ────────────────────────────────────────────────────────────────────

SLIDER_MARK = {"color": TEXT, "fontSize": "12px", "fontFamily": SIDEBAR_FONT}

app.layout = html.Div(style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": SIDEBAR_FONT, "color": TEXT}, children=[

    # Header
    html.Div(style={"background": "#0a0a18", "borderBottom": f"1px solid {BORDER}", "padding": "16px 32px", "display": "flex", "alignItems": "center", "gap": "16px"}, children=[
        html.Span("📈", style={"fontSize": "28px"}),
        html.Div([
            html.H1("FinaceBro", style={"margin": 0, "fontSize": "22px", "color": "#ffffff", "letterSpacing": "2px", "fontFamily": SIDEBAR_FONT}),
            html.P("Commodity Futures Simulation & Prediction", style={"margin": 0, "fontSize": "12px", "color": MUTED, "fontFamily": SIDEBAR_FONT}),
        ]),
    ]),

    # Body
    html.Div(style={"display": "flex", "gap": "20px", "padding": "20px 32px"}, children=[

        # ── Left sidebar: controls ─────────────────────────────────────────
        html.Div(style={"width": "270px", "flexShrink": 0}, children=[
            card([
                html.Label("Commodity", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="commodity",
                    options=COMMODITY_OPTIONS,
                    value="gold",
                    clearable=False,
                    style={"background": "#1a1a35", "color": "#ffffff", "border": f"1px solid {BORDER}", "marginTop": "4px", "fontSize": "14px"},
                ),
                html.Div(style={"height": "20px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px"}),

                html.Label("MC Model", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="mc-model",
                    options=[
                        {"label": "  GBM (standard)", "value": "gbm"},
                        {"label": "  Jump-Diffusion",  "value": "jump_diffusion"},
                    ],
                    value="gbm",
                    labelStyle=RADIO_LABEL,
                ),
                html.Div(style={"height": "20px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px"}),

                html.Label("ML Model", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="ml-model",
                    options=[
                        {"label": "  Ensemble (RF + GB)", "value": "ensemble"},
                        {"label": "  Random Forest",       "value": "rf"},
                        {"label": "  Gradient Boost",      "value": "gb"},
                    ],
                    value="ensemble",
                    labelStyle=RADIO_LABEL,
                ),
                html.Div(style={"height": "20px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px"}),

                # Forecast horizon — max 165 trading days ≈ mid-November from March 26
                html.Label("Forecast Horizon", style=LABEL_STYLE),
                html.Div(id="horizon-display", style={"fontSize": "13px", "color": MUTED, "marginBottom": "8px", "fontFamily": SIDEBAR_FONT}),
                dcc.Slider(id="forecast-days", min=5, max=165, step=5, value=30,
                           marks={5: {"label": "5d", "style": SLIDER_MARK},
                                  30: {"label": "30d", "style": SLIDER_MARK},
                                  60: {"label": "60d", "style": SLIDER_MARK},
                                  90: {"label": "90d", "style": SLIDER_MARK},
                                  120: {"label": "120d", "style": SLIDER_MARK},
                                  165: {"label": "~Mid-Nov", "style": {**SLIDER_MARK, "color": YELLOW}}},
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Div(style={"height": "20px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px"}),

                html.Label("Simulations", style=LABEL_STYLE),
                dcc.Slider(id="n-sims", min=1000, max=20000, step=1000, value=5000,
                           marks={1000:  {"label": "1k",  "style": SLIDER_MARK},
                                  5000:  {"label": "5k",  "style": SLIDER_MARK},
                                  10000: {"label": "10k", "style": SLIDER_MARK},
                                  20000: {"label": "20k", "style": SLIDER_MARK}},
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Div(style={"height": "20px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px"}),

                html.Label("History (years)", style=LABEL_STYLE),
                dcc.Slider(id="history-years", min=1, max=10, step=1, value=5,
                           marks={i: {"label": str(i), "style": SLIDER_MARK} for i in [1, 2, 3, 5, 7, 10]},
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Div(style={"height": "20px", "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px"}),

                html.Label("Data Source", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="data-source",
                    options=[
                        {"label": "  Live (internet)", "value": "live"},
                        {"label": "  Demo (synthetic)", "value": "demo"},
                    ],
                    value="demo",
                    labelStyle=RADIO_LABEL,
                ),
                html.Div(style={"height": "24px"}),

                html.Button(
                    "▶  Run Simulation",
                    id="run-btn",
                    n_clicks=0,
                    style={
                        "width": "100%", "padding": "14px",
                        "background": ACCENT, "color": "#fff",
                        "border": "none", "borderRadius": "8px",
                        "fontSize": "15px", "fontFamily": SIDEBAR_FONT,
                        "fontWeight": "700", "cursor": "pointer",
                        "letterSpacing": "0.5px",
                    },
                ),
                html.Div(id="status-msg", style={"marginTop": "10px", "fontSize": "13px", "color": MUTED, "textAlign": "center", "fontFamily": SIDEBAR_FONT}),
            ]),
        ]),

        # ── Right: output panels ───────────────────────────────────────────
        html.Div(style={"flex": 1, "minWidth": 0}, children=[

            # Quote bar
            html.Div(id="quote-bar", style={"marginBottom": "16px"}),

            # Main charts
            dcc.Loading(
                id="loading",
                type="circle",
                color=ACCENT,
                children=html.Div(id="charts-area"),
            ),

            # Stats table
            html.Div(id="stats-table"),
        ]),
    ]),
])


# ── Horizon label callback ────────────────────────────────────────────────────

@callback(Output("horizon-display", "children"), Input("forecast-days", "value"))
def update_horizon_label(days):
    from datetime import date, timedelta
    if not days:
        return ""
    # Approximate calendar days from trading days (×1.4 accounts for weekends)
    target = date.today() + timedelta(days=int(days * 1.4))
    return f"{days} trading days  →  ~{target.strftime('%b %d, %Y')}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plotly_template():
    return {
        "layout": {
            "paper_bgcolor": PANEL_BG,
            "plot_bgcolor":  PANEL_BG,
            "font":          {"color": TEXT, "family": "monospace"},
            "xaxis":         {"gridcolor": BORDER, "linecolor": BORDER},
            "yaxis":         {"gridcolor": BORDER, "linecolor": BORDER},
        }
    }


def _stat_pill(label, value, color=TEXT):
    return html.Div(style={
        "display": "inline-block", "background": DARK_BG,
        "border": f"1px solid {BORDER}", "borderRadius": "6px",
        "padding": "8px 16px", "margin": "4px",
    }, children=[
        html.Div(label, style={"fontSize": "10px", "color": MUTED, "letterSpacing": "1px"}),
        html.Div(value, style={"fontSize": "18px", "color": color, "fontWeight": "bold"}),
    ])


# ── Callback ──────────────────────────────────────────────────────────────────

@callback(
    Output("quote-bar",   "children"),
    Output("charts-area", "children"),
    Output("stats-table", "children"),
    Output("status-msg",  "children"),
    Input("run-btn", "n_clicks"),
    State("commodity",     "value"),
    State("mc-model",      "value"),
    State("ml-model",      "value"),
    State("forecast-days", "value"),
    State("n-sims",        "value"),
    State("history-years", "value"),
    State("data-source",   "value"),
    prevent_initial_call=True,
)
def run(n_clicks, commodity, mc_model, ml_model, forecast_days, n_sims, years, data_source):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, ""

    # ── 1. Data ──────────────────────────────────────────────────────────
    try:
        if data_source == "demo":
            hist = generate_history(commodity, years=years)
            live = generate_current(commodity, hist)
        else:
            from scraper import get_commodity_history
            from current_data import fetch_current_commodity
            hist = get_commodity_history(commodity, years=years)
            live = fetch_current_commodity(commodity)
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, f"Data error: {e}"

    cur  = live["price"]
    meta = COMMODITIES[commodity]
    unit = meta["unit"]

    # ── 2. Simulation ─────────────────────────────────────────────────────
    try:
        sim = run_simulation(
            historical_df=hist,
            current_price=cur,
            forecast_days=forecast_days,
            n_simulations=n_sims,
            model=mc_model,
            seed=42,
        )
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, f"Simulation error: {e}"

    # ── 3. ML ─────────────────────────────────────────────────────────────
    try:
        horizons = sorted(set(FORECAST_HORIZONS + [forecast_days]))
        if ml_model == "ensemble":
            pred_df = ensemble_predict(hist, cur, horizons=horizons)
        else:
            pred_df = predict_futures(hist, cur, horizons=horizons, model_type=ml_model)
    except Exception as e:
        pred_df = pd.DataFrame()

    paths  = sim["paths"]
    stats  = sim["stats"]
    params = sim["params"]
    days   = np.arange(paths.shape[1])

    # ── Quote bar ─────────────────────────────────────────────────────────
    chg_color = GREEN if live["change"] >= 0 else RED
    sign = "+" if live["change"] >= 0 else ""
    quote_bar = card([
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}, children=[
            _stat_pill(meta["name"], f"{cur:,.2f} {unit}",          color="#eeeeff"),
            _stat_pill("Change",     f"{sign}{live['change']:,.2f}", color=chg_color),
            _stat_pill("Change %",   f"{sign}{live['change_pct']:.2f}%", color=chg_color),
            _stat_pill("Day High",   f"{live['day_high']:,.2f}"),
            _stat_pill("Day Low",    f"{live['day_low']:,.2f}"),
            _stat_pill("Source",     "DEMO" if data_source == "demo" else "LIVE",
                       color=YELLOW if data_source == "demo" else GREEN),
        ]),
    ], style={"padding": "12px 16px"})

    # ── Charts ────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Monte Carlo Price Paths",
            f"Terminal Price Distribution (Day {forecast_days})",
            "ML Forecast by Horizon",
            "Price Fan Chart + ML Overlay",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # Sample paths
    idx = np.random.default_rng(0).choice(len(paths), min(150, len(paths)), replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=days, y=paths[i], mode="lines",
            line=dict(color="rgba(80,120,220,0.07)", width=0.8),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # Bands + mean
    if "p5" in stats.columns:
        fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([stats["p95"], stats["p5"][::-1]]),
            fill="toself", fillcolor="rgba(60,100,200,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="5–95%", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([stats["p75"], stats["p25"][::-1]]),
            fill="toself", fillcolor="rgba(60,100,200,0.30)",
            line=dict(color="rgba(0,0,0,0)"), name="25–75%", showlegend=True,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=stats["mean"], mode="lines",
        line=dict(color=ACCENT, width=2), name="Mean"), row=1, col=1)
    fig.add_hline(y=cur, line=dict(color=YELLOW, dash="dash", width=1.5),
        annotation_text="Current", row=1, col=1)

    # Terminal histogram
    terminal = paths[:, -1]
    fig.add_trace(go.Histogram(
        x=terminal, nbinsx=60, marker_color=ACCENT,
        opacity=0.75, name="Terminal price", showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=cur, line=dict(color=YELLOW, dash="dash"), row=1, col=2)
    fig.add_vline(x=float(stats["mean"].iloc[-1]), line=dict(color=ACCENT, dash="dot"), row=1, col=2)
    if "p5" in stats.columns:
        fig.add_vline(x=float(stats["p5"].iloc[-1]),  line=dict(color=RED,   dash="dot", width=1), row=1, col=2)
        fig.add_vline(x=float(stats["p95"].iloc[-1]), line=dict(color=GREEN, dash="dot", width=1), row=1, col=2)

    # ML bar chart
    if not pred_df.empty:
        colors_ml = [GREEN if p >= cur else RED for p in pred_df["predicted_price"]]
        fig.add_trace(go.Bar(
            x=[f"{h}d" for h in pred_df.index],
            y=pred_df["predicted_price"],
            marker_color=colors_ml, opacity=0.8,
            name="ML Forecast", showlegend=False,
            text=[f"{c:+.1f}%" for c in pred_df["change_pct"]],
            textposition="outside",
        ), row=2, col=1)
        fig.add_hline(y=cur, line=dict(color=YELLOW, dash="dash", width=1.5), row=2, col=1)

    # Fan chart panel (duplicate of panel 1 without paths, cleaner)
    if "p5" in stats.columns:
        fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([stats["p95"], stats["p5"][::-1]]),
            fill="toself", fillcolor="rgba(60,100,200,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([stats["p75"], stats["p25"][::-1]]),
            fill="toself", fillcolor="rgba(60,100,200,0.30)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ), row=2, col=2)
    fig.add_trace(go.Scatter(x=days, y=stats["mean"], mode="lines",
        line=dict(color=ACCENT, width=2), showlegend=False), row=2, col=2)
    if "p50" in stats.columns:
        fig.add_trace(go.Scatter(x=days, y=stats["p50"], mode="lines",
            line=dict(color="#aaccff", width=1, dash="dash"), showlegend=False), row=2, col=2)
    fig.add_hline(y=cur, line=dict(color=YELLOW, dash="dash", width=1.5), row=2, col=2)

    # ML dots on fan chart
    if not pred_df.empty:
        for h, row_ml in pred_df.iterrows():
            if h <= forecast_days:
                c = GREEN if row_ml["predicted_price"] >= cur else RED
                fig.add_trace(go.Scatter(
                    x=[h], y=[row_ml["predicted_price"]],
                    mode="markers+text",
                    marker=dict(color=c, size=10, symbol="circle"),
                    text=[f"{row_ml['change_pct']:+.1f}%"],
                    textposition="top center",
                    textfont=dict(size=9),
                    showlegend=False,
                ), row=2, col=2)

    fig.update_layout(
        height=660,
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, family="monospace", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)

    charts = card([dcc.Graph(figure=fig, config={"displayModeBar": True})], style={"padding": "8px"})

    # ── Stats table ───────────────────────────────────────────────────────
    last = stats.iloc[-1]
    mean_p = last["mean"]
    p5_v   = last.get("p5",  np.nan)
    p95_v  = last.get("p95", np.nan)

    rows_data = [
        ("Annual Drift (μ)",   f"{params['annual_mu']*100:.2f}%"),
        ("Annual Volatility (σ)", f"{params['annual_sigma']*100:.2f}%"),
        ("MC Model",           params["model"].upper()),
        (f"Mean Price (Day {forecast_days})", f"{mean_p:,.2f}"),
        ("Mean Change",        f"{(mean_p/cur-1)*100:+.2f}%"),
        ("5th pct (bear)",     f"{p5_v:,.2f}  ({(p5_v/cur-1)*100:+.1f}%)"),
        ("95th pct (bull)",    f"{p95_v:,.2f}  ({(p95_v/cur-1)*100:+.1f}%)"),
        ("95% VaR",            f"{sim['var_95']*100:.2f}%"),
        ("95% CVaR",           f"{sim['cvar_95']*100:.2f}%"),
        ("Simulations",        f"{n_sims:,}"),
    ]
    if not pred_df.empty and forecast_days in pred_df.index:
        ml_row = pred_df.loc[forecast_days]
        rows_data += [
            ("ML Forecast",     f"{ml_row['predicted_price']:,.2f}  ({ml_row['change_pct']:+.2f}%)"),
            ("ML Direction",    ml_row["direction"]),
            ("ML Confidence",   f"{ml_row['confidence']:.1f}%"),
            ("ML CV R²",        f"{ml_row['cv_r2']:.4f}"),
            ("Blended Estimate", f"{(mean_p + ml_row['predicted_price'])/2:,.2f}"),
        ]

    table = card([
        html.H4("Simulation Summary", style={"marginTop": 0, "color": "#eeeeff", "fontSize": "13px", "letterSpacing": "1px"}),
        html.Table(style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"}, children=[
            html.Tbody([
                html.Tr(style={"borderBottom": f"1px solid {BORDER}"}, children=[
                    html.Td(k, style={"padding": "6px 8px", "color": MUTED}),
                    html.Td(v, style={"padding": "6px 8px", "color": TEXT, "textAlign": "right"}),
                ]) for k, v in rows_data
            ])
        ]),
    ])

    return quote_bar, charts, table, f"Done — {n_sims:,} paths in {forecast_days}d horizon"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args, _ = parser.parse_known_args()
    port = args.port

    # Check if port is already in use and find a free one
    def port_free(p):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return True
        except OSError:
            return False

    if not port_free(port):
        print(f"\n  Port {port} is already in use.")
        for p in range(port + 1, port + 20):
            if port_free(p):
                port = p
                print(f"  Using port {port} instead.")
                break
        else:
            print("  ERROR: No free port found in range. Kill existing processes:")
            print(f"    lsof -ti:{args.port} | xargs kill -9")
            raise SystemExit(1)

    print(f"\n  FinaceBro starting...")
    print(f"  Open: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(debug=False, host="127.0.0.1", port=port)
