# Softr field portal — design spec

A no-code **Softr** app on top of the Airtable base that `airtable_sync.py` fills each week. It turns model output into a to-do list for the field team and collects the feedback the model needs to keep improving.

## Data source

| Airtable table | Purpose |
|---|---|
| **Stock-out Alerts** | One record per alert (written weekly by the ML pipeline) |
| **Outlets** | Outlet master: ID, name, channel, region, assigned merchandiser (linked record) |
| **Team** | Merchandisers and regional managers: name, email, role, region |

`Stock-out Alerts.Outlet ID` links to `Outlets`, which links to `Team`, so every alert has an owner.

## User groups and permissions

| Group | Sees | Can do |
|---|---|---|
| **Merchandiser** | Alerts for *their own* outlets only (filter: `Outlet → Assigned merchandiser = Logged-in user`) | Change **Status** to *Actioned* / *Not needed*, add a note and a shelf photo |
| **Regional manager** | All alerts in *their region* | Reassign alerts, see completion rates, read the AI briefing |
| **Head of commercial / analytics** | Everything | Dashboards, export |

## Pages

1. **My alerts (home)**: a list block sorted by Risk (high → low). Each card shows outlet, SKU, closing stock, weeks of cover, a risk badge (red ≥ 50%, amber ≥ 25%), the **reasons** and the **suggested action**. Buttons: **Mark actioned** and **Not needed** (both update Status in Airtable), plus **Add photo**.
2. **Outlet detail**: alert history for one outlet, a map pin, and the last delivery date.
3. **Region dashboard** (managers): chart blocks for alerts by reason, % actioned within 24h, and stock-outs that happened despite an alert.
4. **Weekly briefing**: renders `ai_briefing.md` (stored in a long-text field of a `Briefings` table).

## Feedback loop into the model

The **Status** and **actioned date** fields are the most valuable data the portal creates:

- *Actioned* alerts where no stock-out followed → evidence that the intervention works (measures uplift)
- *Not needed* alerts → false positives, used to recalibrate the threshold
- Stock-outs that **weren't** alerted → missed cases, reviewed monthly

A monthly job pulls these fields back from Airtable (pyairtable `table.all()`), adds them to the training data, and refits the model.

## Why Airtable + Softr?

- **Fast to build:** a working field app in days, with no front-end code
- **Mobile-friendly** for merchandisers on Android phones
- **Role-based access** without building authentication
- The **ML pipeline stays in Python**, and Airtable is the simple contract between data science and operations
