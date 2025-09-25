import os
import re
import glob
from datetime import date
from typing import Dict, List, Tuple, Optional

import pandas as pd

# ------------------------------
# Helpers
# ------------------------------

EXCEL_LETTERS = {chr(i): idx for idx, i in enumerate(range(ord('A'), ord('Z') + 1))}

def col_letter_to_index(letter: str) -> int:
    """
    Convert Excel column letter(s) to zero-based index.
    e.g. A->0, B->1, ..., Z->25, AA->26
    """
    letter = letter.strip().upper()
    val = 0
    for ch in letter:
        if ch < 'A' or ch > 'Z':
            raise ValueError(f"Invalid column letter: {letter}")
        val = val * 26 + (ord(ch) - ord('A') + 1)
    return val - 1

def try_read(path: str) -> pd.DataFrame:
    """
    Read CSV by default; if .xlsx and openpyxl is present, read Excel.
    """
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path)
    elif low.endswith(".xlsx") or low.endswith(".xls"):
        try:
            import openpyxl  # noqa: F401
            return pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"Cannot read Excel without openpyxl: {e}")
    else:
        # best effort CSV
        return pd.read_csv(path)

def latest_one(patterns: List[str], folder: str) -> Optional[str]:
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def all_gen_lims(folder: str) -> List[str]:
    pats = [
        "Gen_LIMs_*.csv", "Gen_LIMS_*.csv",  # common spellings
        "Gen_LIMs_*.xlsx", "Gen_LIMS_*.xlsx",
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files

def series_first_or_not_found(s: pd.Series) -> str:
    try:
        val = s.dropna().astype(str)
        return val.iloc[0] if len(val) else "Not Found"
    except Exception:
        return "Not Found"

def ensure_numeric(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def text_contains(haystack: pd.Series, needle: str) -> pd.Series:
    # case-insensitive "contains"
    return haystack.astype(str).str.contains(re.escape(needle), case=False, na=False)

def left_matches(s: pd.Series, prefix: str) -> pd.Series:
    return s.astype(str).str.startswith(str(prefix), na=False)

def col_by_letter(df: pd.DataFrame, letter: str) -> pd.Series:
    try:
        idx = col_letter_to_index(letter)
        return df.iloc[:, idx]
    except Exception:
        # fallback: empty series with same index
        return pd.Series([None] * len(df), index=df.index)

# ------------------------------
# File discovery
# ------------------------------

def load_latest_files(uploads_dir: str) -> Dict[str, object]:
    """
    Returns:
      {
        "total_products": <path or None>,
        "data_consolidator": <path or None>,
        "gen_lims": [<paths>]
      }
    """
    tp = latest_one(
        ["*TotalProducts*.csv", "*Total_Products*.csv", "*TotalProducts*.xlsx", "*Total_Products*.xlsx"],
        uploads_dir,
    )
    dc = latest_one(
        ["*Data_Consolidator*.csv", "*Data Consolidator*.csv", "*Data_Consolidator*.xlsx", "*Data Consolidator*.xlsx"],
        uploads_dir,
    )
    gl = all_gen_lims(uploads_dir)
    return {"total_products": tp, "data_consolidator": dc, "gen_lims": gl}

# ------------------------------
# ID list (E17 dropdown)
# ------------------------------

def list_sample_ids(uploads_dir: str, limit: int = 5000) -> List[str]:
    files = load_latest_files(uploads_dir)
    tp_path = files["total_products"]
    if not tp_path:
        return []
    df = try_read(tp_path)
    # In Google formulas, E17 is matched to TotalProducts column I
    # So default to "I" (9th col) for the ID column
    try:
        id_col = col_by_letter(df, "I")
    except Exception:
        id_col = df.iloc[:, min(8, df.shape[1]-1)]
    ids = id_col.dropna().astype(str).unique().tolist()
    return ids[:limit]

# ------------------------------
# Report computation (Google formulas -> pandas)
# ------------------------------

def compute_report(sample_id: str, uploads_dir: str) -> Dict[str, str]:
    files = load_latest_files(uploads_dir)
    tp_path = files["total_products"]
    dc_path = files["data_consolidator"]
    gen_paths = files["gen_lims"]

    result = {}  # "cell" -> "value" (strings)

    # 1) Load frames (best effort)
    tp = try_read(tp_path) if tp_path else pd.DataFrame()
    dc = try_read(dc_path) if dc_path else pd.DataFrame()
    gens: List[pd.DataFrame] = []
    for p in gen_paths:
        try:
            gens.append(try_read(p))
        except Exception:
            pass

    # ---------------- TotalProducts (letters) ----------------
    # We mimic the sheet letters referenced in your formulas:
    # A=1.., B=2, D=4, G=7, H=8, I=9, M=13
    if not tp.empty:
        tp_col_I = col_by_letter(tp, "I").astype(str)
        mask_tp = tp_col_I == str(sample_id)

        # A17 = INDEX(TotalProducts!B:B, MATCH(E17, TotalProducts!I:I, 0))
        result["A17"] = series_first_or_not_found(col_by_letter(tp, "B")[mask_tp])

        # D17 = INDEX(TotalProducts!H:H, MATCH(E17, TotalProducts!I:I, 0))
        result["D17"] = series_first_or_not_found(col_by_letter(tp, "H")[mask_tp])

        # G17 = INDEX(TotalProducts!D:D, MATCH(E17, TotalProducts!I:I, 0))
        result["G17"] = series_first_or_not_found(col_by_letter(tp, "D")[mask_tp])

        # H17 = INDEX(TotalProducts!G:G, MATCH(E17, TotalProducts!I:I, 0))
        result["H17"] = series_first_or_not_found(col_by_letter(tp, "G")[mask_tp])

        # D28 = VLOOKUP(E17, TotalProducts!I:M, 5, FALSE) -> column M from the matched row
        result["D28"] = series_first_or_not_found(col_by_letter(tp, "M")[mask_tp])
    else:
        result["A17"] = result["D17"] = result["G17"] = result["H17"] = result["D28"] = "Not Found"

    # E17 is the chosen Sample ID (dropdown)
    result["E17"] = str(sample_id)
    # h12 = e17; i3 = h12; f12 = h17; d12 = today()
    result["H12"] = result["E17"]
    result["I3"] = result["H12"]
    result["F12"] = result.get("H17", "Not Found")
    result["D12"] = date.today().isoformat()

    # ---------------- Gen_LIMs union (like GETSHEETNAMES + QUERY) ----------------
    # Build CombinedData with columns: C (ID), F (Analyte), M (Result), W (ValueW)
    if gens:
        parts_C = []
        parts_F = []
        parts_M = []
        parts_W = []
        for g in gens:
            try:
                parts_C.append(col_by_letter(g, "C"))
                parts_F.append(col_by_letter(g, "F"))
                parts_M.append(col_by_letter(g, "M"))
                parts_W.append(col_by_letter(g, "W"))
            except Exception:
                continue
        if parts_C:
            C = pd.concat(parts_C, ignore_index=True)
            F = pd.concat(parts_F, ignore_index=True)
            M = pd.concat(parts_M, ignore_index=True)
            W = pd.concat(parts_W, ignore_index=True)
            combined = pd.DataFrame({"C": C, "F": F, "M": M, "W": W})
            # Filter by sample_id prefix and Analyte in {Bisphenol S, PFAS}
            mask_id = combined["C"].astype(str).str.startswith(str(sample_id), na=False)
            mask_an = combined["F"].astype(str).str.fullmatch(r"(?i)(Bisphenol S|PFAS)")
            filtered = combined[mask_id & mask_an]

            # A28: first matching Analyte (F)
            result["A28"] = series_first_or_not_found(filtered["F"])

            # G28: first matching W
            result["G28"] = series_first_or_not_found(filtered["W"])

            # E28 = 1 * G28 (force numeric)
            g28_num = ensure_numeric(result["G28"])
            result["E28"] = ("" if g28_num is None else str(g28_num))
        else:
            result["A28"] = result["G28"] = result["E28"] = "Not Found"
    else:
        result["A28"] = result["G28"] = result["E28"] = "Not Found"

    # ---------------- Data_Consolidator lookups ----------------
    # Columns assumption by letters (as in your formulas):
    # A: Name/Row label, B: Analyte, C: Result, D: Units/Value?, E: Qualifier/Text, G: SheetName
    if not dc.empty:
        colA = col_by_letter(dc, "A").astype(str)
        colB = col_by_letter(dc, "B").astype(str)
        colC = col_by_letter(dc, "C")
        colD = col_by_letter(dc, "D")
        colE = col_by_letter(dc, "E").astype(str)
        colG = col_by_letter(dc, "G").astype(str)

        # SheetName = first row where LEFT(A)=sample_id and A not containing 'Method Blank'/'Calibration Blank'
        m_base = left_matches(colA, str(sample_id)) & ~text_contains(colA, "Method Blank") & ~text_contains(colA, "Calibration Blank")
        sheet_name = series_first_or_not_found(colG[m_base])
        if sheet_name == "Not Found":
            # fallback: any row for sample_id
            sheet_name = series_first_or_not_found(colG[left_matches(colA, str(sample_id))])

        # H28 = first E where G==sheet_name AND A contains sample_id AND analyte is BPS|PFAS excluding spike/blank/calibrant
        mask_h28 = (colG == sheet_name) & left_matches(colA, str(sample_id)) \
                   & ~text_contains(colA.str.lower(), "spike") \
                   & ~text_contains(colA.str.lower(), "blank") \
                   & ~text_contains(colA.str.lower(), "calibrant") \
                   & colB.str.contains(r"(?i)Bisphenol S|PFAS", regex=True)
        result["H28"] = series_first_or_not_found(colE[mask_h28])

        # 65-block: Method Blank (first row)
        mask_mb = (colG == sheet_name) & text_contains(colA, "Method Blank") & colB.str.contains(r"(?i)Bisphenol S|PFAS", regex=True)
        result["A65"] = series_first_or_not_found(colB[mask_mb])
        result["D65"] = series_first_or_not_found(colC[mask_mb])
        # E65 = 1 * G65 (we fill later after G65)
        # F65 = F28
        result["F65"] = result.get("F28", result.get("D28", ""))  # you referenced f28; we map to D28 if F28 is not explicitly provided
        result["H65"] = series_first_or_not_found(colE[mask_mb])

        # 68-block: Matrix Spike 1
        mask_ms1 = (colG == sheet_name) & text_contains(colA, "Matrix Spike 1") & colB.str.contains(r"(?i)Bisphenol S|PFAS", regex=True)
        result["A68"] = series_first_or_not_found(colB[mask_ms1])
        result["D68"] = series_first_or_not_found(colC[mask_ms1])
        result["G68"] = series_first_or_not_found(colD[mask_ms1])
        # E68 = 1 * G68
        g68_num = ensure_numeric(result["G68"])
        result["E68"] = ("" if g68_num is None else str(g68_num))
        # F68 = F65
        result["F68"] = result.get("F65", "")

        # 71-block: Matrix Spike Duplicate
        mask_msd = (colG == sheet_name) & text_contains(colA, "Matrix Spike Duplicate") & colB.str.contains(r"(?i)Bisphenol S|PFAS", regex=True)
        result["A71"] = series_first_or_not_found(colB[mask_msd])
        result["D71"] = series_first_or_not_found(colC[mask_msd])
        result["F71"] = series_first_or_not_found(colD[mask_msd])
        result["E71"] = result.get("F68", "")

        # G65 = first D where G==sheet_name AND row describes Method Blank for analyte (already computed above as D of MB)
        result["G65"] = result.get("F28", result.get("D28", ""))  # your formula references F28; we align with D28 when F28 is not used explicitly
        # Now E65 = 1 * G65
        g65_num = ensure_numeric(result["G65"])
        result["E65"] = ("" if g65_num is None else str(g65_num))

        # Advanced % recovery calculations:

        # Helper: extract parent ID from the MS1 row A text "Matrix Spike 1: <ParentID>"
        def parent_id_from_ms1() -> Optional[str]:
            a_text = series_first_or_not_found(colA[mask_ms1])
            if a_text and a_text != "Not Found":
                m = re.search(r":\s*(.+)$", a_text)
                if m:
                    return m.group(1).strip()
            return None

        parent_id = parent_id_from_ms1() or str(sample_id)

        # OriginalResult for parent (first C where LEFT(A)=ParentID and analyte matches)
        mask_parent = (colG == sheet_name) & left_matches(colA, parent_id) & colB.str.contains(r"(?i)Bisphenol S|PFAS", regex=True)
        original_result = ensure_numeric(series_first_or_not_found(colC[mask_parent]))

        # Value parsed from H68 (numbers inside text)
        def parse_number_from_text(s: str) -> Optional[float]:
            if not s or s == "Not Found":
                return None
            m = re.search(r"(\d+\.?\d*)", str(s))
            return float(m.group(1)) if m else None

        h68_num = parse_number_from_text(result.get("H68", ""))  # H68 might be text like "xx ug/L"
        d68_num = ensure_numeric(result.get("D68", None))

        # i68 = (D68*100)/(OriginalResult + value_from_H68)
        if d68_num is not None and original_result is not None:
            denom = original_result + (h68_num or 0.0)
            result["I68"] = str(round((d68_num * 100.0) / denom, 3)) if denom else "Calculation Error"
        else:
            result["I68"] = "Calculation Error"

        # g71 similar but using duplicate spike D71 and H68 (as in your sheet)
        d71_num = ensure_numeric(result.get("D71", None))
        if d71_num is not None and original_result is not None:
            denom2 = original_result + (h68_num or 0.0)
            result["G71"] = str(round((d71_num * 100.0) / denom2, 3)) if denom2 else "Calculation Error"
        else:
            result["G71"] = "Calculation Error"

    else:
        # Data consolidator missing
        for key in ["H28", "A65", "D65", "E65", "F65", "H65",
                    "A68", "D68", "E68", "F68", "G68",
                    "A71", "D71", "E71", "F71", "G71", "I68"]:
            result[key] = "Not Found"

    # Mirror values you listed:
    # i3=h12, i54=h12, a57=a10, f57=f10, d59=Today(), f59=f12, h59=e17
    result["I54"] = result["H12"]
    result["D59"] = date.today().isoformat()
    result["F59"] = result["F12"]
    result["H59"] = result["E17"]

    return result
