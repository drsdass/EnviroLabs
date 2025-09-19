# -*- coding: utf-8 -*-
"""Formula hooks for Enviro Labs LIMS.

Edit compute_fields(row) to return any derived fields you want to store
and display on the report. The function receives a dict per CSV row with
your original headers (case-insensitive helper `get()` included).

This code runs on your server during CSV import; only trusted admins
should edit it.
"""

import math
from datetime import datetime

def get(row, key, default=None):
    """Case-insensitive, trimmed lookup from the incoming row dict."""
    lk = str(key).strip().lower()
    for k, v in row.items():
        if str(k).strip().lower() == lk:
            return v
    return default

def to_float(x):
    try:
        if x is None or str(x).strip() in {"", "nan", "None"}:
            return None
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def contains(text, needle):
    return (str(text) if text is not None else "").lower().find(str(needle).lower()) >= 0

def compute_fields(row: dict) -> dict:
    """Return a dict of derived fields for this row.

    Replace the examples below with your real formulas.
    """
    fields = {}

    # Example: flag based on 'Result'
    result_text = get(row, "Result", "")
    if contains(result_text, "detected") or contains(result_text, "positive") or str(result_text).strip() == "1" or result_text is True:
        fields["Flag"] = "Detected"
    elif str(result_text).strip() == "" or contains(result_text, "pending"):
        fields["Flag"] = "Pending"
    else:
        fields["Flag"] = "Not Detected"

    # Example: threshold interpretation
    value = to_float(get(row, "Concentration"))
    action = to_float(get(row, "ActionLevel", 70))
    if value is not None and action is not None:
        fields["ExceedsActionLevel"] = value > action

    # Example: sum any columns named PFAS_*
    pfas_cols = [k for k in row.keys() if str(k).upper().startswith("PFAS_")]
    if pfas_cols:
        fields["PFAS_Sum"] = round(sum(to_float(row[k]) or 0.0 for k in pfas_cols), 3)

    # Example: pretty display
    fields["DisplayName"] = (str(get(row, "Patient")) or str(get(row, "Patient Name")) or "").strip()

    return fields
