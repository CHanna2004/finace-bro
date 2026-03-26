"""
Configuration for commodity futures simulation.
"""

# Supported commodities with their Yahoo Finance ticker symbols
COMMODITIES = {
    "gold":         {"ticker": "GC=F",  "name": "Gold Futures",          "unit": "USD/oz"},
    "silver":       {"ticker": "SI=F",  "name": "Silver Futures",        "unit": "USD/oz"},
    "crude_oil":    {"ticker": "CL=F",  "name": "WTI Crude Oil Futures", "unit": "USD/bbl"},
    "brent_oil":    {"ticker": "BZ=F",  "name": "Brent Crude Oil",       "unit": "USD/bbl"},
    "natural_gas":  {"ticker": "NG=F",  "name": "Natural Gas Futures",   "unit": "USD/MMBtu"},
    "corn":         {"ticker": "ZC=F",  "name": "Corn Futures",          "unit": "USX/bu"},
    "wheat":        {"ticker": "ZW=F",  "name": "Wheat Futures",         "unit": "USX/bu"},
    "soybeans":     {"ticker": "ZS=F",  "name": "Soybean Futures",       "unit": "USX/bu"},
    "copper":       {"ticker": "HG=F",  "name": "Copper Futures",        "unit": "USD/lb"},
    "platinum":     {"ticker": "PL=F",  "name": "Platinum Futures",      "unit": "USD/oz"},
    "palladium":    {"ticker": "PA=F",  "name": "Palladium Futures",     "unit": "USD/oz"},
    "cotton":       {"ticker": "CT=F",  "name": "Cotton Futures",        "unit": "USX/lb"},
    "sugar":        {"ticker": "SB=F",  "name": "Sugar Futures",         "unit": "USX/lb"},
    "coffee":       {"ticker": "KC=F",  "name": "Coffee Futures",        "unit": "USX/lb"},
    "cocoa":        {"ticker": "CC=F",  "name": "Cocoa Futures",         "unit": "USD/MT"},
}

# Simulation defaults
DEFAULT_SIMULATIONS = 10_000
DEFAULT_FORECAST_DAYS = 30
DEFAULT_HISTORY_YEARS = 5
CONFIDENCE_INTERVALS = [0.05, 0.25, 0.50, 0.75, 0.95]  # percentiles

# Data cache directory
CACHE_DIR = ".cache"
CACHE_TTL_MINUTES = 15  # re-fetch live data after this many minutes
