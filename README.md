# Enviro Labs – Minimal LIMS Portal (Flask)

A Render-ready Flask app with **Admin** and **Client** portals, static credentials, CSV import, search, and a lightweight audit trail.

## Features
- 🔐 Static login for **Admin** and **Client** (change via env vars)
- 🔎 Search by **Lab ID** and **Resulted Date** range
- 📤 Admin CSV import (create/update reports)
- 🧾 Client can view/print only their own reports
- 📝 Audit trail: login/logout, imports, exports
- 📦 SQLite database (`app.db`) stored with the app
- 🖨️ Print-friendly report view
- 🚀 Render deployment (Procfile + Gunicorn)

## Quickstart (local)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; edit values if needed
python app.py
# visit http://localhost:5000
```

## Credentials (defaults)
- **Admin**: `admin` / `Enviro#123`
- **Client**: `client` / `Client#123` (scoped to `CLIENT_NAME` env var, default `Artemis`)

You can override in environment or Render dashboard:
```
SECRET_KEY=change-me
ADMIN_USERNAME=admin
ADMIN_PASSWORD=Enviro#123
CLIENT_USERNAME=client
CLIENT_PASSWORD=Client#123
CLIENT_NAME=Artemis
KEEP_UPLOADED_CSVS=true
```

## CSV Columns
Provide a `.csv` (or Excel) with headers (case-insensitive, common variants supported):
- **Lab ID** (required) – also accepts: accession, accession_id, id, labid…
- **Client** (required)
- Patient
- Test
- Result
- Collected Date
- Resulted Date
- PDF URL

See `sample_reports.csv` for an example.

## Deploy to Render
1. Create a new **GitHub repo** and upload all files.
2. In Render, create a **Web Service**:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
3. Add Environment variables (recommended to change passwords!).
4. Click **Deploy**.

## Should you keep CSVs on the server?
- **Pros:** reproducibility, re-imports, off-line audit of raw submissions.
- **Cons:** extra storage, potential PHI handling risks.
- Default behavior keeps them; uncheck “Keep original file” on upload _or_ set `KEEP_UPLOADED_CSVS=false` to delete them after import.
