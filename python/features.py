"""Feature engineering for next-week stock-out prediction.

Every feature describes the situation at the END of week t, using only information a planner
has on Sunday night: this week's closing stock, recent sales and service history, and the
known plan for next week (delivery schedule, promotions, holidays). The label is whether
the outlet-SKU stocks out in week t+1. No information from week t+1 itself leaks in.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

NUMERIC = ["closing_stock", "avg_sales_4w", "weeks_of_cover", "sales_trend", "stockouts_8w",
           "late_deliveries_8w", "short_deliveries_8w", "weeks_since_delivery", "delivery_next_week",
           "promo_next_week", "holiday_next_week", "depot_shortage_this_week", "merch_visits_per_week",
           "distance_km", "shelf_life_days", "price_kes"]
CATEGORICAL = ["channel", "region", "category"]


def build(panel: pd.DataFrame, outlets: pd.DataFrame) -> pd.DataFrame:
    d = panel.merge(outlets, on="outlet_id").sort_values(["outlet_id", "sku", "week"]).reset_index(drop=True)
    g = d.groupby(["outlet_id", "sku"], sort=False)
    roll = lambda col, w, f="mean": g[col].transform(lambda s: getattr(s.rolling(w, min_periods=1), f)())
    d["avg_sales_4w"] = roll("units_sold", 4)
    d["weeks_of_cover"] = d.closing_stock / d.avg_sales_4w.clip(lower=.5)
    d["sales_trend"] = roll("units_sold", 2) / d.avg_sales_4w.clip(lower=.5)
    d["stockouts_8w"] = roll("stockout", 8, "sum")
    d["late_deliveries_8w"] = roll("delivery_late", 8, "sum")
    d["short_deliveries_8w"] = roll("delivery_short", 8, "sum")
    # weeks since the last delivery that actually arrived
    arrived = (d.delivered_units > 0).astype(int)
    grp = arrived.groupby([d.outlet_id, d.sku]).cumsum()
    d["weeks_since_delivery"] = d.groupby([d.outlet_id, d.sku, grp]).cumcount()
    d["depot_shortage_this_week"] = d.depot_shortage
    # known plan for next week (delivery calendar, promo calendar, holidays)
    for src, dst in [("delivery_week", "delivery_next_week"), ("promo", "promo_next_week"),
                     ("holiday", "holiday_next_week"), ("stockout", "target")]:
        d[dst] = g[src].shift(-1)
    d["week_index"] = d.groupby(["outlet_id", "sku"]).cumcount()
    d = d[(d.week_index >= 8) & d.target.notna()].copy()            # 8-week warm-up for rolling features
    d["target"] = d.target.astype(int)
    for c in ["delivery_next_week", "promo_next_week", "holiday_next_week"]:
        d[c] = d[c].astype(int)
    return d


def load():
    panel = pd.read_csv(ROOT / "data" / "weekly_inventory.csv.gz", parse_dates=["week"])
    outlets = pd.read_csv(ROOT / "data" / "outlets.csv")
    return build(panel, outlets)


def time_split(d: pd.DataFrame):
    """Out-of-time split: train on older weeks, tune on the next 8, test on the final 10."""
    w = np.sort(d.week.unique())
    train_end, valid_end = w[-19], w[-11]
    return (d[d.week <= train_end], d[(d.week > train_end) & (d.week <= valid_end)], d[d.week > valid_end])
