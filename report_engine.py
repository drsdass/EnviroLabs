# report_engine.py
import io
import re
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# -------- Helpers --------

def _read_csv(file_storage_or_path):
    """
    Accepts a Flask FileStorage or a file path. Returns a pandas DataFrame.
    """
    if hasattr(file_storage_or_path, "stream"):  # FileStorage
        file_storage_or_path.stream.seek(0)
        return pd.read_csv(file_storage_or_path.stream)
    # path
    return pd.read_csv(file_storage_or_path)

def _sanitize_sheet_name(name: str) -> str:
    # Excel sheet name rules; also keep Google Sheets friendly
    bad = r'[\\/*?:\[\]]'
    name = re.sub(bad, "_", name)
    name = name.strip()
    if not name:
        name = "Sheet"
    return name[:31]  # Excel limit

def _write_df_to_sheet(wb: Workbook, sheet_name: str, df: pd.DataFrame):
    ws = wb.create_sheet(_sanitize_sheet_name(sheet_name))
    # Write header
    ws.append([str(c) for c in df.columns.tolist()])
    # Write rows
    for row in df.itertuples(index=False, name=None):
        ws.append(list(row))
    return ws

def _ensure_report_sheet(wb: Workbook):
    # Default WB has one sheet named "Sheet"; replace with "Report"
    if wb.worksheets:
        wb.remove(wb.worksheets[0])
    return wb.create_sheet("Report")

def _put_formula(ws, cell_ref: str, formula_text: str):
    if not cell_ref:
        return
    cell = str(cell_ref).strip().upper()
    if not cell:
        return
    if not formula_text:
        return
    f = str(formula_text).strip()
    if f.lower() in ("dropdown", "select", "(dropdown)"):
        # We'll leave empty; route may fill E17 with the selected Product/ID.
        return
    if not f.startswith("="):
        f = "=" + f
    ws[cell] = f

# -------- Public API --------

def build_report_workbook(
    product_or_id_value: str,
    formulas_csv,                  # FileStorage or path; Column B = cell, Column C = formula
    total_products_csv,            # FileStorage or path -> sheet "TotalProducts"
    data_consolidator_csv,         # FileStorage or path -> sheet "Data_Consolidator"
    gen_lims_csv_list=None,        # list[FileStorage or path]; each becomes its own sheet (name from filename)
) -> io.BytesIO:
    """
    Returns an in-memory .xlsx with:
      - Sheet 'Report' that has formulas from formulas CSV (Col B cell, Col C formula).
      - Sheet 'TotalProducts' from CSV.
      - Sheet 'Data_Consolidator' from CSV.
      - One sheet per uploaded Gen_LIMs_* CSV.
      - Cell E17 in 'Report' set to product_or_id_value (so formulas depending on E17 have an input).

    NOTE: We *do not* evaluate formulas server-side. Excel/Google Sheets will evaluate on open.
    """
    gen_lims_csv_list = gen_lims_csv_list or []

    # Load dataframes
    tp_df  = _read_csv(total_products_csv)
    dc_df  = _read_csv(data_consolidator_csv)
    # Formulas CSV may have arbitrary headers; we will take the 2nd and 3rd columns.
    f_df   = _read_csv(formulas_csv)

    if f_df.shape[1] < 3:
        raise ValueError("Formulas.csv must have at least 3 columns (B = target cell, C = formula).")

    # Workbook build
    wb = Workbook()
    report_ws = _ensure_report_sheet(wb)

    # Populate data sheets first (so formulas referencing them are valid on open)
    _write_df_to_sheet(wb, "TotalProducts", tp_df)
    _write_df_to_sheet(wb, "Data_Consolidator", dc_df)

    for item in gen_lims_csv_list:
        # Use incoming filename (without extension) if available
        sheet_name = getattr(item, "filename", None)
        if sheet_name:
            sheet_name = sheet_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            sheet_name = sheet_name.rsplit(".", 1)[0]
        else:
            sheet_name = "Gen_LIMs_Data"
        df = _read_csv(item)
        _write_df_to_sheet(wb, sheet_name, df)

    # Apply formulas from Formulas.csv:
    #    Column B -> cell, Column C -> formula text (we add '=' if missing)
    for _, row in f_df.iterrows():
        cell_ref  = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        formula_t = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        if cell_ref:
            _put_formula(report_ws, cell_ref, formula_t)

    # Provide the selected product/id into E17 (if your mapping expects E17 as the "dropdown" value)
    if product_or_id_value:
        report_ws["E17"] = str(product_or_id_value)

    # Nice to have: freeze header area
    report_ws.freeze_panes = "A2"

    # Save to bytes
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out
