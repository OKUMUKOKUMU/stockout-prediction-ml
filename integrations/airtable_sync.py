"""Push this week's stock-out alerts to Airtable (the data layer behind the Softr field portal).

Modes
  * LIVE    - if AIRTABLE_TOKEN and AIRTABLE_BASE_ID are set, records are upserted into the
              "Stock-out Alerts" table using the official pyairtable client. Upsert key =
              Alert ID (week + outlet + SKU), so re-running never creates duplicates.
  * DRY-RUN - otherwise, the exact payload that would be sent is written to
              outputs/airtable_payload.json. This keeps the pipeline runnable anywhere.

Expected Airtable table "Stock-out Alerts" fields:
  Alert ID (text, primary) · Week (date) · Outlet ID (text) · Channel (single select) ·
  Region (single select) · SKU (text) · Closing Stock (number) · Weeks of Cover (number) ·
  Risk (percent) · Expected Value KES (currency) · Reasons (long text) · Suggested Action (single select) ·
  Status (single select: New / Actioned / Not needed) - set by merchandisers in Softr

Usage:  python integrations/airtable_sync.py
Never commit tokens: set them as environment variables (or a .env file excluded by .gitignore).
"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = "Stock-out Alerts"


def to_records(alerts: pd.DataFrame) -> list[dict]:
    recs = []
    for r in alerts.itertuples():
        recs.append({"fields": {
            "Alert ID": f"{r.week}-{r.outlet_id}-{r.sku}",
            "Week": str(r.week)[:10],
            "Outlet ID": r.outlet_id,
            "Channel": r.channel,
            "Region": r.region,
            "SKU": r.sku,
            "Closing Stock": int(r.closing_stock),
            "Weeks of Cover": round(float(r.weeks_of_cover), 2),
            "Risk": round(float(r.risk), 3),
            "Expected Value KES": round(float(r.ev), 0),
            "Reasons": r.reasons,
            "Suggested Action": r.suggested_action,
            "Status": "New",
        }})
    return recs


def main():
    alerts = pd.read_csv(ROOT / "outputs" / "alerts_latest_week.csv")
    records = to_records(alerts)
    token, base_id = os.getenv("AIRTABLE_TOKEN"), os.getenv("AIRTABLE_BASE_ID")
    if token and base_id:
        from pyairtable import Api
        table = Api(token).table(base_id, TABLE)
        result = table.batch_upsert(records, key_fields=["Alert ID"], typecast=True)
        print(f"LIVE: upserted {len(records)} alerts into Airtable table '{TABLE}' "
              f"({len(result['createdRecords'])} new, {len(result['updatedRecords'])} updated)")
    else:
        out = ROOT / "outputs" / "airtable_payload.json"
        out.write_text(json.dumps({"table": TABLE, "performUpsert": {"fieldsToMergeOn": ["Alert ID"]},
                                   "typecast": True, "records": records}, indent=2))
        print(f"DRY-RUN: {len(records)} alert records prepared for Airtable table '{TABLE}' -> {out.relative_to(ROOT)}")
        print("Set AIRTABLE_TOKEN and AIRTABLE_BASE_ID to sync for real.")


if __name__ == "__main__":
    main()
