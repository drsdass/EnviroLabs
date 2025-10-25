import os
import io
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import json # <--- NEW: For storing structured data

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text as sql_text
import pandas as pd

# ------------------- Config -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = str(os.getenv("KEEP_UPLOADED_CSVS", "true")).lower() == "true"

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- App -------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ------------------- DB -------------------
DB_PATH = os.path.join(BASE_DIR, "app.db")
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)

    # Core identity / simple fields
    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)

    # Sample Results (primary)
    test = Column(String, nullable=True)     # analyte (e.g., "Bisphenol S", "PFAS GROUP", ...)
    result = Column(String, nullable=True)   # numeric-as-text or textual

    collected_date = Column(Date, nullable=True) # "Received Date"
    resulted_date = Column(Date, nullable=True)  # "Reported Date"
    pdf_url = Column(String, nullable=True) # <-- Sample Analyte Accumulation (Renamed for clarity in comments)

    # ---- Optional metadata fields (strings to keep SQLite simple) ----
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    project_lead = Column(String, nullable=True)
    address = Column(String, nullable=True)

    sample_name = Column(String, nullable=True)
    prepared_by = Column(String, nullable=True)
    matrix = Column(String, nullable=True)
    prepared_date = Column(String, nullable=True)
    qualifiers = Column(String, nullable=True)
    asin = Column(String, nullable=True)
    product_weight_g = Column(String, nullable=True)

    # Sample results extras
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)

    # QC: Method Blank
    mb_analyte = Column(String, nullable=True) 
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)
    
    # QC Accumulation Repurposed Fields
    acq_datetime = Column(String, nullable=True) # <-- MB Accumulation
    sheet_name = Column(String, nullable=True)   # <-- MS1 Accumulation

    # QC: Matrix Spike 1
    ms1_analyte = Column(String, nullable=True) 
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True) # <-- MSD Accumulation

    # QC: Matrix Spike Duplicate
    msd_analyte = Column(String, nullable=True)
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)

    # --- NEW CRITICAL FIELD FOR STRUCTURED QC DATA ---
    qc_json_data = Column(Text, nullable=True) 
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False) # 'admin' or 'client'
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# --- one-time add columns if the DB was created earlier without the new fields ---
def _ensure_report_columns():
    needed = {
        # Existing Fields
        "phone", "email", "project_lead", "address", "sample_name", "prepared_by",
        "matrix", "prepared_date", "qualifiers", "asin", "product_weight_g",
        "sample_mrl", "sample_units", "sample_dilution", "sample_analyzed", "sample_qualifier",
        "mb_analyte", "mb_result", "mb_mrl", "mb_units", "mb_dilution",
        "ms1_analyte", "ms1_result", "ms1_mrl", "ms1_units", "ms1_dilution",
        "ms1_fortified_level", "ms1_pct_rec", "ms1_pct_rec_limits",
        "msd_analyte", "msd_result", "msd_units", "msd_dilution",
        "msd_pct_rec", "msd_pct_rec_limits", "msd_pct_rpd", "msd_pct_rpd_limit",
        "acq_datetime", "sheet_name",
        # NEW Field
        "qc_json_data", 
    }
    with engine.begin() as conn:
        cols = set()
        for row in conn.execute(sql_text("PRAGMA table_info(reports)")):
            cols.add(row[1]) # name
        missing = needed - cols
        for col in sorted(missing):
            conn.execute(sql_text(f"ALTER TABLE reports ADD COLUMN {col} TEXT"))

_ensure_report_columns()

# ------------------- Helpers -------------------
PFAS_LIST = [
    "PFOA","PFOS","PFNA","FOSAA","N-MeFOSAA","N-EtFOSAA",
    "SAmPAP","PFOSA","N-MeFOSA","N-MeFOSE","N-EtFOSA","N-EtFOSE","diSAmPAP", "BISPHENOL S" # Include BPS for static array
]
PFAS_SET_UPPER = {a.upper() for a in PFAS_LIST}

# Static structure to ensure all 14 analytes appear in QC tables
STATIC_ANALYTES = [
    "PFOA", "PFOS", "PFNA", "FOSAA", "N-MeFOSAA", "N-EtFOSAA", 
    "SAmPAP", "PFOSA", "N-MeFOSA", "N-MeFOSE", "N-EtFOSA", "N-EtFOSE", 
    "diSAmPAP", "Bisphenol S"
]

def current_user():
    return {
        "username": session.get("username"),
        "role": session.get("role"),
        "client_name": session.get("client_name"),
    }

def require_login(role=None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("home"))
            if role and session.get("role") != role:
                flash("Unauthorized", "error")
                return redirect(url_for("dashboard"))
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

def log_action(username, role, action, details=""):
    db = SessionLocal()
    try:
        db.add(AuditLog(username=username, role=role, action=action, details=details))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

def parse_date(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None

def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _find_token_col(cols: List[str], *needles: str) -> Optional[int]:
    """Find a column index whose normalized name contains all tokens (AND match)."""
    tokens = [t.lower() for t in needles]
    for i, c in enumerate(cols):
        name = _norm(c)
        if all(tok in name for tok in tokens):
            return i
    return None

def _find_exact(cols: List[str], name: str) -> Optional[int]:
    name_l = name.strip().lower()
    for i, c in enumerate(cols):
        if c.strip().lower() == name_l:
            return i
    return None

def _find_sequence(cols: List[str], seq: List[str]) -> Optional[int]:
    """Find starting index of a consecutive sequence of column captions (case-insensitive)."""
    n = len(cols)
    m = len(seq)
    seq_l = [s.lower() for s in seq]
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            if cols[i + j].strip().lower() != seq_l[j]:
                ok = False
                break
        if ok:
            return i
    return None

# --- CRITICAL HELPER: Lab ID Normalization ---
def _normalize_lab_id(lab_id: str) -> str:
    """
    Removes common Lab ID suffixes (like ' 2x_nitrogen blow down', ' 0.5ppb', etc.)
    using regular expressions for robust stripping.
    """
    s = (lab_id or "").strip()
    if not s:
        return s
        
    # Pattern looks for a space followed by numbers, units, or common suffixes.
    pattern = r'\s+[\-\+]?\d*\.?\d+(?:ppb|ppt|ng\/g|ug\/g|\s\d*|\s.*)?$'
    
    # Apply regex substitution to remove the matched suffix
    normalized = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
    
    # If stripping resulted in an empty string, return the original.
    if not normalized:
        return s
        
    return normalized

def _target_analyte_ok(analyte: str) -> bool:
    if analyte is None:
        return False
    a = analyte.strip().upper()
    return (a == "BISPHENOL S") or (a in PFAS_SET_UPPER)

def _is_pfas_analyte(analyte: str) -> bool:
    if analyte is None:
        return False
    return analyte.strip().upper() in PFAS_SET_UPPER

# --- NEW HELPER: Generates the HTML table body content ---
def _generate_report_table_html(reports: List[Report], app_instance) -> str:
    """
    Generates the raw <tbody> HTML string for the dashboard table.
    """
    html_rows = []
    
    def get_report_detail_url(report_id):
        with app_instance.app_context():
            return url_for('report_detail', report_id=report_id)

    for r in reports:
        # Use r.id for the database lookup in the URL
        detail_url = get_report_detail_url(r.id)
        
        # Prepare data safely
        lab_id = r.lab_id or ""
        client = r.client or ""
        sample_name = r.sample_name or ""
        test = r.test or ""
        result = r.result or ""
        sample_units = r.sample_units or ""
        reported_date = r.resulted_date.isoformat() if r.resulted_date else ""

        row = f"""
        <tr>
            <td><a class="link" href="{detail_url}">{lab_id}</a></td>
            <td>{client}</td>
            <td>{sample_name}</td> 
            <td>{test}</td>
            <td>{result}</td>
            <td>{sample_units}</td> 
            <td>{reported_date}</td>
        </tr>
        """
        html_rows.append(row)
        
    return "\n".join(html_rows)

# --- NEW HELPER: Retrieves and structures QC data for the template ---
def _get_structured_qc_data(r: Report) -> Dict[str, Dict[str, Any]]:
    """
    Parses the accumulation strings and reorganizes data for the template.
    Returns a dictionary keyed by analyte name (str) for easy lookup in Jinja.
    """
    qc_map = {name: {} for name in STATIC_ANALYTES}
    
    # 1. Main Sample Results Parsing (r.pdf_url: Analyte: ResultUnit | ...)
    if r.pdf_url:
        for item in r.pdf_url.split(' | '):
            if ': ' in item:
                analyte, result_unit = item.split(': ', 1)
                analyte = analyte.strip()
                if analyte in qc_map:
                    qc_map[analyte]['sample_result_units'] = result_unit.strip()

    # 2. Method Blank Parsing (r.acq_datetime: Analyte|Result|MRL|Units|Dilution | ...)
    if r.acq_datetime:
        for item in r.acq_datetime.split(' | '):
            parts = item.split('|')
            if len(parts) >= 5:
                analyte = parts[0].strip()
                if analyte in qc_map:
                    qc_map[analyte]['mb_result'] = parts[1].strip()
                    qc_map[analyte]['mb_mrl'] = parts[2].strip()
                    qc_map[analyte]['mb_units'] = parts[3].strip()
                    qc_map[analyte]['mb_dilution'] = parts[4].strip()

    # 3. Matrix Spike 1 Parsing (r.sheet_name: Analyte|Result|MRL|Units|Dilution|FortifiedLevel|%REC | ...)
    if r.sheet_name:
        for item in r.sheet_name.split(' | '):
            parts = item.split('|')
            if len(parts) >= 7:
                analyte = parts[0].strip()
                if analyte in qc_map:
                    qc_map[analyte]['ms1_result'] = parts[1].strip()
                    qc_map[analyte]['ms1_mrl'] = parts[2].strip()
                    qc_map[analyte]['ms1_units'] = parts[3].strip()
                    qc_map[analyte]['ms1_dilution'] = parts[4].strip()
                    qc_map[analyte]['ms1_fortified_level'] = parts[5].strip()
                    qc_map[analyte]['ms1_pct_rec'] = parts[6].strip()

    # 4. Matrix Spike Duplicate Parsing (r.ms1_pct_rec_limits: Analyte|Result|Units|Dilution|%REC|%REC Limits|%RPD | ...)
    if r.ms1_pct_rec_limits:
        for item in r.ms1_pct_rec_limits.split(' | '):
            parts = item.split('|')
            if len(parts) >= 7:
                analyte = parts[0].strip()
                if analyte in qc_map:
                    qc_map[analyte]['msd_result'] = parts[1].strip()
                    qc_map[analyte]['msd_units'] = parts[2].strip()
                    qc_map[analyte]['msd_dilution'] = parts[3].strip()
                    qc_map[analyte]['msd_pct_rec'] = parts[4].strip()
                    qc_map[analyte]['msd_pct_rec_limits'] = parts[5].strip()
                    qc_map[analyte]['msd_pct_rpd'] = parts[6].strip()

    # Convert the map into a predictable list for template iteration
    final_list = []
    for analyte in STATIC_ANALYTES:
        data = qc_map[analyte]
        # Only include if the analyte was found in the data, or if we want the full static list
        # For QC, we must include all 14 analytes statically
        final_list.append({
            'analyte': analyte,
            'is_bps': analyte == "Bisphenol S",
            'is_pfas': analyte != "Bisphenol S",
            'sample_result': data.get('sample_result_units', ''),
            
            'mb_result': data.get('mb_result', ''),
            'mb_mrl': data.get('mb_mrl', ''),
            'mb_units': data.get('mb_units', ''),
            'mb_dilution': data.get('mb_dilution', ''),
            
            'ms1_result': data.get('ms1_result', ''),
            'ms1_mrl': data.get('ms1_mrl', ''),
            'ms1_units': data.get('ms1_units', ''),
            'ms1_dilution': data.get('ms1_dilution', ''),
            'ms1_fortified_level': data.get('ms1_fortified_level', ''),
            'ms1_pct_rec': data.get('ms1_pct_rec', ''),
            'ms1_pct_rec_limits': data.get('msd_pct_rec_limits', ''), # Use MSD limits as best available
            
            'msd_result': data.get('msd_result', ''),
            'msd_units': data.get('msd_units', ''),
            'msd_dilution': data.get('msd_dilution', ''),
            'msd_pct_rec': data.get('msd_pct_rec', ''),
            'msd_pct_rec_limits': data.get('msd_pct_rec_limits', ''),
            'msd_pct_rpd': data.get('msd_pct_rpd', ''),
            'msd_pct_rpd_limit': r.msd_pct_rpd_limit or '', # Static RPD limit
        })
        
    return final_list


# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
# ... (login route unchanged) ...

@app.route("/logout")
# ... (logout route unchanged) ...

@app.route("/dashboard")
# ... (dashboard route unchanged, still passes ORM objects) ...

@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    r = db.query(Report).get(report_id)
    db.close()
    if not r:
        flash("Report not found", "error")
        return redirect(url_for("dashboard"))
    if u["role"] == "client" and r.client != u["client_name"]:
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    def val(x): return "" if x is None else str(x)
    
    # --- CRITICAL FIX: Generate structured QC data in Python ---
    structured_qc_list = _get_structured_qc_data(r)
    
    p = {
        "client_info": {
            "client": val(r.client), "phone": val(r.phone), "email": val(r.email) or "support@envirolabsusa.com",
            "project_lead": val(r.project_lead), "address": val(r.address),
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "", "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": val(r.sample_name or r.lab_id), "prepared_by": val(r.prepared_by),
            "matrix": val(r.matrix), "prepared_date": val(r.prepared_date),
            "qualifiers": val(r.qualifiers), "asin": val(r.asin), "product_weight_g": val(r.product_weight_g),
        },
        "sample_results": {
            "analyte": val(r.test), "result_summary": val(r.pdf_url) or "N/A", 
            "mrl": val(r.sample_mrl), "units": val(r.sample_units),
            "dilution": val(r.sample_dilution), "analyzed": val(r.sample_analyzed),
            "qualifier": val(r.sample_qualifier),
        },
        # Pass the final structured list to the template
        "analyte_list": structured_qc_list,
        
        # Keep single-value QC for MSD limits (for template fallback)
        "matrix_spike_dup": {
            "pct_rpd_limit": val(r.msd_pct_rpd_limit),
        },
    }

    return render_template("report_detail.html", user=u, r=r, p=p)

# ... (rest of the upload and helper routes unchanged) ...
