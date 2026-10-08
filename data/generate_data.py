"""Simulate weekly shelf inventory for retail outlets x SKUs to create a stock-out prediction dataset.

Each outlet-SKU pair runs a simple inventory process, week by week:
  * Demand        ~ Poisson(base rate x seasonality x promo uplift x holiday uplift)
  * Replenishment : deliveries on a fixed cycle (weekly or fortnightly by channel). The order
                    quantity follows an "order-up-to" rule with human error, and the depot
                    sometimes ships short (fill rate < 100%) or late (arrives a week later)
  * Merchandiser  : outlets with frequent visits get extra top-up orders when stock looks low
  * Stock-out     : weekly demand exceeds stock on hand -> shelf empties, sales are lost

The label we want to predict: will this outlet-SKU stock out NEXT week?
Features are recorded at the END of each week (what a planner would know on Sunday night).

All data is synthetic. Usage: python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
OUT = Path(__file__).parent
N_OUTLETS, N_WEEKS = 220, 60
weeks = pd.date_range("2025-07-07", periods=N_WEEKS, freq="W-MON")

SKUS = {  # sku: (category, base weekly units in a mid-size outlet, shelf life days, price KES)
    "Yoghurt 500ml Strawberry": ("Yoghurt", 14, 21, 180), "Yoghurt 500ml Vanilla": ("Yoghurt", 11, 21, 180),
    "Yoghurt 250ml Kids": ("Yoghurt", 18, 21, 90), "Greek Yoghurt 450g": ("Yoghurt", 5, 28, 420),
    "Fresh Milk 500ml": ("Milk", 40, 7, 65), "Long-life Milk 1L": ("Milk", 16, 180, 140),
    "Cheddar 200g": ("Cheese", 6, 90, 520), "Mozzarella 200g": ("Cheese", 4, 60, 560),
    "Butter 250g": ("Butter", 7, 120, 390), "Ice Cream 1L Vanilla": ("Ice Cream", 5, 365, 650),
    "Ice Cream 1L Chocolate": ("Ice Cream", 4, 365, 650), "Cream 250ml": ("Cream", 3, 14, 260),
    "Sausages 500g": ("Deli", 8, 30, 420), "Bacon 250g": ("Deli", 3, 30, 480), "Burger Buns 6pk": ("Bakery", 9, 5, 180),
}
CHANNELS = {"Supermarket": (.25, 3.0, 1), "Duka": (.45, .6, 2), "Petrol Station": (.15, .8, 2), "HoReCa": (.15, 1.3, 1)}
REGIONS = ["Nairobi", "Central", "Rift Valley", "Western", "Nyanza", "Coast"]

outlets = pd.DataFrame({"outlet_id": [f"OUT{i:04d}" for i in range(1, N_OUTLETS + 1)]})
outlets["channel"] = RNG.choice(list(CHANNELS), N_OUTLETS, p=[c[0] for c in CHANNELS.values()])
outlets["region"] = RNG.choice(REGIONS, N_OUTLETS, p=[.3, .15, .17, .13, .12, .13])
outlets["merch_visits_per_week"] = np.where(outlets.channel == "Supermarket", RNG.choice([2, 3], N_OUTLETS),
                                            RNG.choice([0, 0, 1], N_OUTLETS))
outlets["size_factor"] = RNG.lognormal(0, .35, N_OUTLETS)
outlets["distance_km"] = np.round(RNG.gamma(2, 20, N_OUTLETS), 1)       # from depot: drives late deliveries

season = 1 + .12 * np.sin(2 * np.pi * (np.arange(N_WEEKS) + 10) / 52)
holiday = np.isin(weeks.strftime("%m-%d"), []) | ((weeks.month == 12) & (weeks.day >= 15))
depot_short = RNG.random(N_WEEKS) < .10                                   # network-wide depot shortage weeks

rows = []
for o in outlets.itertuples():
    _, scale, cycle = CHANNELS[o.channel]
    late_p = .04 + .004 * o.distance_km
    for sku, (cat, base, shelf, price) in SKUS.items():
        if o.channel in ("Duka", "Petrol Station") and cat in ("Cheese", "Cream") and RNG.random() < .6:
            continue                                                       # small outlets don't range every SKU
        mu = base * scale * o.size_factor
        promo = RNG.random(N_WEEKS) < .07
        target_cover = cycle + RNG.uniform(.25, .9)                        # order-up-to level: cycle + safety cover (weeks)
        on_hand = int(mu * target_cover)
        pipeline = 0                                                       # late delivery arriving next week
        hist_sales, hist_so, hist_late = [], [], []
        offset = int(RNG.integers(0, cycle))
        for t in range(N_WEEKS):
            delivery_week = (t + offset) % cycle == 0
            # --- deliveries arrive at the start of the week
            arrived = pipeline; pipeline = 0
            late = short = False
            if delivery_week:
                recent = np.mean(hist_sales[-4:]) if hist_sales else mu
                forecast = (.5 * recent + .5 * mu * season[t]) * RNG.uniform(.85, 1.15)   # sales history + planner judgement
                qty = max(0, int(forecast * target_cover + 2 * np.sqrt(forecast) - on_hand))
                if depot_short[t] or RNG.random() < .05:
                    qty = int(qty * RNG.uniform(.3, .8)); short = True
                if RNG.random() < late_p:
                    pipeline = qty; late = True
                else:
                    arrived += qty
            if (o.merch_visits_per_week and on_hand < .7 * mu and (not delivery_week or late)
                    and RNG.random() < min(1, .35 * o.merch_visits_per_week)):
                arrived += int(mu * .9) + 1                                # merchandiser spots low stock -> top-up order
            stock = on_hand + arrived
            demand = RNG.poisson(mu * season[t] * (1.35 if promo[t] else 1) * (1.25 if holiday[t] else 1))
            sold = min(demand, stock)
            stockout = int(demand > stock)
            rows.append(dict(week=weeks[t].date(), outlet_id=o.outlet_id, sku=sku, category=cat,
                             shelf_life_days=shelf, price_kes=price, opening_stock=stock, units_sold=sold,
                             closing_stock=stock - sold, stockout=stockout, delivery_week=int(delivery_week),
                             delivered_units=arrived, delivery_late=int(late), delivery_short=int(short),
                             promo=int(promo[t]), holiday=int(holiday[t]), depot_shortage=int(depot_short[t])))
            on_hand = stock - sold
            # perishables: a share of old stock expires
            if shelf <= 7:
                on_hand = int(on_hand * .8)
            elif shelf <= 21:
                on_hand = int(on_hand * .95)
            hist_sales.append(sold)

panel = pd.DataFrame(rows)
panel.to_csv(OUT / "weekly_inventory.csv.gz", index=False, compression="gzip")
outlets.drop(columns="size_factor").to_csv(OUT / "outlets.csv", index=False)
print(f"{panel.outlet_id.nunique()} outlets x {panel.sku.nunique()} SKUs x {N_WEEKS} weeks = {len(panel):,} rows | "
      f"stock-out rate {panel.stockout.mean():.1%}")
