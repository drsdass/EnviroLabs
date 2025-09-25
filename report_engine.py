# report_engine.py
import io
import re
import datetime as _dt
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# ---------------- Helpers ----------------

def _read_csv(file_storage_or_path):
    """Accepts a Flask FileStorage or a file path. Returns a pandas DataFrame."""
    if hasattr(file_storage_or_path, "stream"):  # Flask FileStorage
        file_storage_or_path.stream.seek(0)
        return pd.read_csv(file_storage_or_path.stream)
    return pd.read_csv(file_storage_or_path)

def _sanitize_sheet_name(name: str) -> str:
    # Excel sheet name rules
    bad = r'[\\/*?:\[\]]'
    name = re.sub(bad, "_", str(name or "Sheet")).strip()
    return name[:31] or "Sheet"

def _write_df_to_sheet(wb: Workbook, sheet_name: str, df: pd.DataFrame):
    ws = wb.create_sheet(_sanitize_sheet_name(sheet_name))
    ws.append([str(c) for c in df.columns.tolist()])
    for row in df.itertuples(index=False, name=None):
        ws.append(list(row))
    return ws

def _ensure_report_sheet(wb: Workbook):
    # Replace default "Sheet" with "Report"
    if wb.worksheets:
        wb.remove(wb.worksheets[0])
    return wb.create_sheet("Report")

def _put_value(ws, cell_ref: str, value, number_format: str | None = None):
    cell = str(cell_ref).strip().upper()
    if not cell:
        return
    ws[cell] = value
    if number_format:
        ws[cell].number_format = number_format

def _put_formula(ws, cell_ref: str, formula_text: str):
    """Writes a formula (adds leading '=' if missing)."""
    cell = str(cell_ref).strip().upper()
    if not cell or not formula_text:
        return
    f = str(formula_text).strip()
    if not f.startswith("="):
        f = "=" + f
    ws[cell] = f

def _combine_gen_lims(gen_lims_csv_list):
    """
    Build two helper sheets from all Gen_LIMs_* CSVs:
      - Combined_CFM: columns [C, F, M] -> ["ID", "Analyte", "Result"]
      - Combined_CFW: columns [C, F, W] -> ["ID", "Analyte", "ValueW"]
    Returns (df_cfm, df_cfw).
    """
    frames_cfm, frames_cfw = [], []
    for item in gen_lims_csv_list or []:
        df = _read_csv(item)
        # We're using column positions, not headers (A=0, B=1, C=2 ...)
        def _safe_pick_cols(df_, idxs):
            cols = []
            for idx in idxs:
                if idx < df_.shape[1]:
                    cols.append(df_.iloc[:, idx])
                else:
                    cols.append(pd.Series([""] * len(df_)))
            return pd.DataFrame({i: c for i, c in zip(idxs, cols)})

        # C(2), F(5), M(12); C(2), F(5), W(22)
        if df.shape[1] > 2:  # only if at least C exists
            cfm = _safe_pick_cols(df, [2, 5, 12])
            cfm.columns = ["ID", "Analyte", "Result"]
            frames_cfm.append(cfm)

            cfw = _safe_pick_cols(df, [2, 5, 22])
            cfw.columns = ["ID", "Analyte", "ValueW"]
            frames_cfw.append(cfw)

    df_cfm = pd.concat(frames_cfm, ignore_index=True) if frames_cfm else pd.DataFrame(columns=["ID","Analyte","Result"])
    df_cfw = pd.concat(frames_cfw, ignore_index=True) if frames_cfw else pd.DataFrame(columns=["ID","Analyte","ValueW"])
    return df_cfm, df_cfw

def _is_today_formula(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in ("today()", "today", "=today()", "=today")

def _excel_equivalent(cell_ref: str, original_formula: str) -> str | None:
    """
    For a few known Google-only patterns, return an Excel/Sheets-compatible formula
    that uses our helper sheets (Combined_CFM / Combined_CFW).
    Extend this mapping as needed.
    """
    cell = (cell_ref or "").strip().upper()
    ftxt = (original_formula or "").strip()

    uses_google_only = ("GETSHEETNAMES" in ftxt) or ("QUERY(" in ftxt) or ("INDIRECT(" in ftxt)
    if not uses_google_only:
        return None

    # A28: First matching Analyte (BPS or PFAS) for current E17 from combined CFM
    if cell == "A28":
        return (
            "=IFERROR("
            "INDEX(FILTER(Combined_CFM!B:B,"
            "(LEFT(Combined_CFM!A:A,LEN($E$17))=$E$17)"
            "*((Combined_CFM!B:B=\"Bisphenol S\")+(Combined_CFM!B:B=\"PFAS\"))),1),"
            "\"Not Found\")"
        )

    # G28: First matching ValueW for current E17 from combined CFW
    if cell == "G28":
        return (
            "=IFERROR("
            "INDEX(FILTER(Combined_CFW!C:C,"
            "(LEFT(Combined_CFW!A:A,LEN($E$17))=_
