"""Turn the alert table into a short, plain-English briefing per region using an LLM.

  * If ANTHROPIC_API_KEY is set, the alert summary is sent to Claude with a structured prompt
    (anthropic Python SDK) and the model writes the briefing.
  * Otherwise a deterministic template briefing is produced from the same data, so the
    pipeline always runs and the output format stays identical.

Only aggregated, non-personal operational data is sent: outlet IDs, SKUs, risk and reason codes.
Output: outputs/ai_briefing.md

Usage:  python integrations/ai_briefing.py
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

PROMPT = """You are a supply-chain analyst for an FMCG dairy distributor in Kenya.
Below are this week's stock-out alerts produced by a machine-learning model, one line per
outlet-SKU, with the model's risk score and the main reasons (from SHAP).

Write a briefing for each REGION's sales manager:
- 2-3 sentences on the overall picture (number of alerts, main causes)
- the 3 most urgent actions, naming outlet and SKU
- keep it under 120 words per region, plain English, no jargon

ALERTS (region | outlet | channel | SKU | risk | reasons | suggested action):
{rows}
"""


def summarise(alerts: pd.DataFrame) -> str:
    rows = "\n".join(f"{r.region} | {r.outlet_id} | {r.channel} | {r.sku} | {r.risk:.0%} | {r.reasons} | {r.suggested_action}"
                     for r in alerts.itertuples())
    return PROMPT.format(rows=rows)


def template_briefing(alerts: pd.DataFrame) -> str:
    out = [f"# Stock-out briefing — week of {str(alerts.week.iloc[0])[:10]}",
           f"*{len(alerts)} alerts across {alerts.region.nunique()} regions · generated from model scores and SHAP reason codes*", ""]
    for region, g in alerts.groupby("region"):
        top_reason = (g.reasons.str.split("; ").explode().value_counts().index[0])
        out.append(f"## {region} — {len(g)} alerts")
        out.append(f"{len(g)} outlet-SKUs are at risk of running out next week, across {g.outlet_id.nunique()} outlets. "
                   f"The most common cause is **{top_reason}**.")
        out.append("")
        out.append("**Most urgent:**")
        for r in g.sort_values("risk", ascending=False).head(3).itertuples():
            out.append(f"1. {r.outlet_id} ({r.channel}) — {r.sku}: {r.risk:.0%} risk ({r.reasons}). **{r.suggested_action}.**")
        out.append("")
    return "\n".join(out)


def main():
    alerts = pd.read_csv(ROOT / "outputs" / "alerts_latest_week.csv")
    out = ROOT / "outputs" / "ai_briefing.md"
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=1500,
                                                    messages=[{"role": "user", "content": summarise(alerts)}])
        out.write_text(msg.content[0].text)
        print(f"LLM: briefing written by {MODEL} -> {out.relative_to(ROOT)}")
    else:
        out.write_text(template_briefing(alerts))
        (ROOT / "outputs" / "ai_briefing_prompt.txt").write_text(summarise(alerts))
        print(f"TEMPLATE: briefing for {alerts.region.nunique()} regions -> {out.relative_to(ROOT)} "
              "(prompt saved to outputs/ai_briefing_prompt.txt; set ANTHROPIC_API_KEY to use the LLM)")


if __name__ == "__main__":
    main()
