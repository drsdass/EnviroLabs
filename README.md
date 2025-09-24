# Enviro Labs Portal

Two portals (Admin & Client) with CSV import and Excel report generation that references
Google Sheets **CSV** datasets.

## Deploy

- Python: `runtime.txt` -> `python-3.12.6`
- Procfile: `web: gunicorn app:app`

## Environment

Copy `.env.example` to `.env` (Render: add env vars in dashboard).

## Folders

- `uploads/`  (auto-created) — optional storage of uploaded report CSVs
- `data/`     (auto-created) — **place Google Sheets CSVs here**, or upload via Admin UI:
  - `TotalProducts.csv`
  - `Data_Consolidator.csv`
  - Any number of `Gen_LIMs_*.csv`

## Admin Flow

1. **Login** with admin credentials.
2. Upload **Reports CSV** to seed/update the database (Lab ID, Client required).
3. Upload datasets used by formulas:
   - `TotalProducts.csv`
   - `Data_Consolidator.csv`
   - `Gen_LIMs_*.csv` (one or many)
4. From Reports table, click **Excel** to download a report workbook:
   - Sheets: `Report`, `TotalProducts`, `Data_Consolidator`, `Gen_Combined`
   - `Report` contains your formulas. Excel 365/2021 required for `LET/FILTER/TEXTAFTER/SEQUENCE`.

## Client Flow

- Login as client; search/filter their own reports; download Excel report.

## Notes

- `E17` in the report is a dropdown sourced from `TotalProducts!I:I` (column I). Ensure your Google Sheet export keeps IDs in column I.
- We consolidate all `Gen_LIMs_*.csv` into `Gen_Combined` so the report can find Analyte/Result/ValueW without Google Sheets-specific functions.
- `%Recovery` calculation parses numbers from `H68` using Excel formulas; keep `H68` textual like "… 12.34 ng/mL" etc.
