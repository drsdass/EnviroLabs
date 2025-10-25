import os
import io
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any # Added Dict, Any

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text as sql_text
import pandas as pd
import json # For structured data handling

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
Base = declarative_declarative_base()

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
    pdf_url = Column(String, nullable=True) # <-- Sample Analyte Accumulation

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
    "SAmPAP","PFOSA","N-MeFOSA","N-MeFOSE","N-EtFOSA","N-EtFOSE","diSAmPAP"
]
PFAS_SET_UPPER = {a.upper() for a in PFAS_LIST}

# --- CRITICAL STATIC LIST FOR TEMPLATE STABILITY ---
STATIC_ANALYTES_LIST = [
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

# --- CRITICAL HELPER: Retrieves and structures QC data for the template ---
def _get_structured_qc_data(r: Report) -> Dict[str, Dict[str, Any]]:
    """
    Parses accumulation strings and reorganizes data into a final map for template use.
    The output structure guarantees that all STATIC_ANALYTES_LIST names are present as keys.
    """
    
    # Initialize maps for parsing from the accumulation strings
    sample_map = {}
    mb_map = {}
    ms1_map = {}
    msd_map = {}
    
    # --- 1. Sample Results Parsing (r.pdf_url: Analyte: ResultUnit | ...) ---
    if r.pdf_url:
        for item in r.pdf_url.split(' | '):
            if ': ' in item:
                analyte, result_unit = item.split(': ', 1)
                sample_map[analyte.strip()] = {'sample_result_units': result_unit.strip()}

    # --- 2. Method Blank Parsing (r.acq_datetime: Analyte|Result|MRL|Units|Dilution | ...) ---
    if r.acq_datetime:
        for item in r.acq_datetime.split(' | '):
            parts = item.split('|')
            if len(parts) >= 5:
                analyte = parts[0].strip()
                mb_map[analyte] = {
                    'mb_result': parts[1].strip(),
                    'mb_mrl': parts[2].strip(),
                    'mb_units': parts[3].strip(),
                    'mb_dilution': parts[4].strip(),
                }

    # --- 3. Matrix Spike 1 Parsing (r.sheet_name: Analyte|Result|MRL|Units|Dilution|FortifiedLevel|%REC | ...) ---
    if r.sheet_name:
        for item in r.sheet_name.split(' | '):
            parts = item.split('|')
            if len(parts) >= 7:
                analyte = parts[0].strip()
                ms1_map[analyte] = {
                    'ms1_result': parts[1].strip(),
                    'ms1_mrl': parts[2].strip(),
                    'ms1_units': parts[3].strip(),
                    'ms1_dilution': parts[4].strip(),
                    'ms1_fortified_level': parts[5].strip(),
                    'ms1_pct_rec': parts[6].strip(),
                }

    # --- 4. Matrix Spike Duplicate Parsing (r.ms1_pct_rec_limits: Analyte|Result|Units|Dilution|%REC|%REC Limits|%RPD | ...) ---
    if r.ms1_pct_rec_limits:
        for item in r.ms1_pct_rec_limits.split(' | '):
            parts = item.split('|')
            if len(parts) >= 7:
                analyte = parts[0].strip()
                msd_map[analyte] = {
                    'msd_result': parts[1].strip(),
                    'msd_units': parts[2].strip(),
                    'msd_dilution': parts[3].strip(),
                    'msd_pct_rec': parts[4].strip(),
                    'msd_pct_rec_limits': parts[5].strip(),
                    'msd_pct_rpd': parts[6].strip(),
                }

    # --- 5. Assemble Final List ---
    final_list = []
    
    # Iterate over the static list to ensure fixed row order and count
    for analyte_name in STATIC_ANALYTES_LIST:
        data = {}
        
        # Merge data from all maps, prioritizing what's available
        # Sample Result Data (from r.pdf_url)
        data.update(sample_map.get(analyte_name, {}))
        
        # QC Data (from accumulation strings)
        data.update(mb_map.get(analyte_name, {}))
        data.update(ms1_map.get(analyte_name, {}))
        data.update(msd_map.get(analyte_name, {}))

        # Build the final list object for the template
        final_list.append({
            'analyte': analyte_name,
            
            # Sample Result Fields
            'sample_result': data.get('sample_result_units', ''),
            'sample_mrl': r.sample_mrl or '', # Use report-level MRL as best guess
            'sample_units': r.sample_units or '',
            'sample_dilution': r.sample_dilution or '',
            'sample_analyzed': r.sample_analyzed or '',
            'sample_qualifier': r.sample_qualifier or '',
            
            # MB Fields
            'mb_result': data.get('mb_result', ''),
            'mb_mrl': data.get('mb_mrl', ''),
            'mb_units': data.get('mb_units', ''),
            'mb_dilution': data.get('mb_dilution', ''),
            
            # MS1 Fields
            'ms1_result': data.get('ms1_result', ''),
            'ms1_mrl': data.get('ms1_mrl', ''),
            'ms1_units': data.get('ms1_units', ''),
            'ms1_dilution': data.get('ms1_dilution', ''),
            'ms1_fortified_level': data.get('ms1_fortified_level', ''),
            'ms1_pct_rec': data.get('ms1_pct_rec', ''),
            'ms1_pct_rec_limits': r.ms1_pct_rec_limits or '', # Report level field
            
            # MSD Fields
            'msd_result': data.get('msd_result', ''),
            'msd_units': data.get('msd_units', ''),
            'msd_dilution': data.get('msd_dilution', ''),
            'msd_pct_rec': data.get('msd_pct_rec', ''),
            'msd_pct_rec_limits': data.get('msd_pct_rec_limits', ''),
            'msd_pct_rpd': data.get('msd_pct_rpd', ''),
            'msd_pct_rpd_limit': r.msd_pct_rpd_limit or '', # Report level field
        })
        
    return final_list


# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    # ... (login route unchanged) ...
    role = request.form.get("role")
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if role == "admin" and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"] = username
        session["role"] = "admin"
        session["client_name"] = None
        log_action(username, "admin", "login", "Admin logged in")
        return redirect(url_for("dashboard"))
    elif role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"] = username
        session["role"] = "client"
        session["client_name"] = CLIENT_NAME
        log_action(username, "client", "login", f"Client '{CLIENT_NAME}' logged in")
        return redirect(url_for("dashboard"))
    else:
        flash("Invalid credentials", "error")
        return redirect(url_for("home"))

@app.route("/logout")
# ... (logout route unchanged) ...
def logout():
    u = current_user()
    if u["username"]:
        log_action(u["username"], u["role"] or "unknown", "logout", "User logged out")
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
# ... (dashboard route unchanged) ...
def dashboard():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    lab_id = request.args.get("lab_id", "").strip()
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()

    db = SessionLocal()
    q = db.query(Report)
    if u["role"] == "client":
        q = q.filter(Report.client == u["client_name"])

    # Apply normalization here too, for searching
    if lab_id:
        normalized_lab_id = _normalize_lab_id(lab_id)
        q = q.filter(Report.lab_id == normalized_lab_id)
        
    if start:
        sd = parse_date(start)
        if sd:
            q = q.filter(Report.resulted_date >= sd)
    if end:
        ed = parse_date(end)
        if ed:
            q = q.filter(Report.resulted_date <= ed)

    try:
        reports = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).limit(500).all()
    except Exception:
        # Fallback for old SQLite versions that don't support NULLS LAST
        reports = q.order_by(Report.resulted_date.desc(), Report.id.desc()).limit(500).all()
    
    db.close()
    
    # Generate the HTML table rows string using the new helper
    reports_html = _generate_report_table_html(reports, app)

    # Pass the raw ORM objects (reports) to check for emptiness and the HTML string
    return render_template("dashboard.html", user=u, reports=reports, reports_html=reports_html)

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
    
    # --- CRITICAL FIX: Generate structured list for template iteration ---
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
        # Pass the structured list to the template for simple looping
        "analyte_list": structured_qc_list,
        
        # Keep single-value QC for MSD limits (for template fallback)
        "matrix_spike_dup": {
            "pct_rpd_limit": val(r.msd_pct_rpd_limit),
        },
    }

    return render_template("report_detail.html", user=u, r=r, p=p)

# ----------- CSV/Excel upload with robust header detection -----------
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    if u["role"] != "admin":
        flash("Only admins can upload CSV files", "error")
        return redirect(url_for("dashboard"))

    f = request.files.get("csv_file")
    if not f or f.filename.strip() == "":
        flash("No file uploaded", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(saved_path)
    keep = request.form.get("keep_original", "on") == "on"

    # Read without headers first to get all data and use custom header setting
    raw = None
    last_err = None
    for loader in (
        lambda: pd.read_csv(saved_path, header=None, dtype=str),
        lambda: pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl"),
    ):
        try:
            # Load all rows; columns are auto-named 0, 1, 2, ...
            raw = loader()
            break
        except Exception as e:
            last_err = e
            
    if raw is None:
        flash(f"Could not read file: {last_err}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    raw = raw.fillna("")
    
    # --- FIX for two-row header: Explicitly set header_row_idx to 1 (the second row) ---
    header_row_idx = 1
    
    if len(raw) <= header_row_idx:
        flash("File is too short to contain the required header (row 2) and data.", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))
    
    flash("Using Excel row 2 as the column header (to skip the thematic row).", "info")

    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    # df starts from the row *after* the header row
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    # Drop fully empty rows
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")
    flash("Header preview: " + ", ".join(df.columns[:12]), "info")

    msg = _ingest_master_upload(df, u, filename)
    flash(msg if not msg.lower().startswith("import failed") else msg, "success" if not msg.lower().startswith("import failed") else "error")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)
    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Data Consolidator File layout with new condensed headers.
    Groups all target analytes for a single (Normalized) Lab ID into one Report.
    Analyte results are accumulated into r.pdf_url.
    QC data is accumulated using a PIPE (|) separator for inner fields into r.acq_datetime and r.sheet_name.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # --- NEW HELPER: Find column index based on a list of names ---
    def find_any_col(names: List[str], fallback_tokens: List[str]) -> Optional[int]:
        """Tries to find a column by exact names first, then by tokens."""
        for name in names:
            idx = _find_exact(cols, name)
            if idx is not None:
                return idx
        
        # Fallback to token matching (original fuzzy logic)
        return _find_token_col(cols, *fallback_tokens)

    # ---- CORE SAMPLE / CLIENT MAPPING (Updated for new condensed headers) ----
    
    # Try new names first, then fallbacks
    idx_lab             = find_any_col(["Sample ID", "SampleID"], ["sample", "id"])
    idx_analyte_name    = find_any_col(["Name", "Analyte Name"], ["name", "analyte"]) # Corresponds to the analyte name (e.g., Bisphenol S)
    idx_final_conc      = find_any_col(["Final Conc."], ["final", "conc"]) # Maps to r.result
    idx_dilution        = find_any_col(["Dil.", "Dilution"], ["dilution"]) # Maps to r.sample_dilution
    idx_acq_datetime_orig = find_any_col(["Acq. Date-Time"], ["acq", "date"]) # Original Acq Date-Time (Date field for reported)

    idx_sample_name     = find_any_col(["Product Name"], ["product", "name"])
    idx_matrix          = find_any_col(["Matrix"], ["matrix"])
    idx_received_by     = find_any_col(["Received By"], ["received", "by"])
    idx_asin            = find_any_col(["ASIN (Identifier)", "Amazon ID"], ["asin", "identifier"])
    idx_weight          = find_any_col(["Weight (Grams)"], ["weight", "g"])
    idx_client          = find_any_col(["Client"], ["client"])
    idx_phone           = find_any_col(["Phone"], ["phone"])
    idx_email           = find_any_col(["Email"], ["email"])
    idx_project_lead    = find_any_col(["Project Lead"], ["project", "lead"])
    idx_address         = find_any_col(["Address"], ["address"])
    idx_sheet_name_orig = find_any_col(["SheetName"], ["sheetname"]) # Original SheetName field

    # ---- QUALITY CONTROL BLOCK MAPPING (Using new named headers) ----
    idx_mb_analyte      = find_any_col(["Analyte (MB)"], ["analyte", "mb"])
    idx_mb_result       = find_any_col(["Result (MB)"], ["result", "mb"])
    idx_mb_mrl          = find_any_col(["MRL (MB)"], ["mrl", "mb"])
    idx_mb_dilution     = find_any_col(["Dilution (MB)"], ["dilution", "mb"])
    
    idx_ms1_analyte     = find_any_col(["Analyte (MS1)"], ["analyte", "ms1"])
    idx_ms1_result      = find_any_col(["Result (MS1)"], ["result", "ms1"])
    idx_ms1_fort_level  = find_any_col(["Fortified Level (MS1)"], ["fortified", "level", "ms1"])
    idx_ms1_pct_rec     = find_any_col(["%REC (MS1)"], ["%rec", "ms1"])

    idx_msd_result      = find_any_col(["Result (MSD)"], ["result", "msd"])
    idx_msd_pct_rec     = find_any_col(["%REC (MSD)"], ["%rec", "msd"])
    idx_msd_rpd         = find_any_col(["%RPD (MSD)"], ["%rpd", "msd"])
    
    # Check for core required fields
    if idx_lab is None or idx_final_conc is None or idx_client is None:
        return "Import failed: Essential columns (Sample ID, Final Conc., Client) not found."


    created = 0
    updated = 0
    skipped_no_sample_name = 0
    skipped_analyte = 0
    
    db = SessionLocal()
    report_data = {} 

    try:
        for _, row in df.iterrows():
            original_lab_id = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""
            
            # --- CRITICAL FILTER CHECK: Sample Name must exist ---
            sample_name_value = str(row.iloc[idx_sample_name]).strip() if idx_sample_name is not None else ""

            if not sample_name_value:
                skipped_no_sample_name += 1
                continue
            
            # --- CRITICAL: Normalize Lab ID for the database key ---
            lab_id = _normalize_lab_id(original_lab_id)
            
            client = str(row.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME
            
            # Use the "Name" column as the primary analyte for logic
            sr_analyte = str(row.iloc[idx_analyte_name]).strip() if idx_analyte_name is not None else ""
            
            if not _target_analyte_ok(sr_analyte):
                skipped_analyte += 1
                continue
            
            is_pfas = _is_pfas_analyte(sr_analyte)
            
            # --- CRITICAL FIX 2: Grouping Logic Key ---
            db_key = lab_id 
            
            existing = report_data.get(db_key)
            if not existing:
                existing = db.query(Report).filter(
                    Report.lab_id == db_key
                ).one_or_none()
                
                if not existing:
                    test_name = "PFAS GROUP" if is_pfas else sr_analyte
                    existing = Report(lab_id=db_key, client=client, test=test_name)
                    existing.pdf_url = "" 
                    existing.sample_name = "" 
                    existing.acq_datetime = "" # MB
                    existing.sheet_name = "" # MS1
                    existing.ms1_pct_rec_limits = "" # MSD Accumulation
                    db.add(existing)
                    created += 1
                else:
                    updated += 1
                
                report_data[db_key] = existing

            r = existing 
            
            # --- General Info and Sample Summary ---
            r.client = client
            r.sample_name = sample_name_value 
            
            if idx_phone is not None:        r.phone = str(row.iloc[idx_phone]).strip()
            if idx_email is not None:        r.email = str(row.iloc[idx_email]).strip()
            if idx_project_lead is not None: r.project_lead = str(row.iloc[idx_project_lead]).strip()
            if idx_address is not None:      r.address = str(row.iloc[idx_address]).strip()
            
            if idx_acq_datetime_orig is not None: r.resulted_date = parse_date(row.iloc[idx_acq_datetime_orig]) # Use Acq. Date-Time as reported date
            r.collected_date = r.resulted_date # Assuming reported and received date are often the same in this consolidated format
            
            if idx_matrix is not None:       r.matrix = str(row.iloc[idx_matrix]).strip()
            if idx_asin is not None:         r.asin = str(row.iloc[idx_asin]).strip()
            if idx_weight is not None:        r.product_weight_g = str(row.iloc[idx_weight]).strip()

            # --- Sample Results Accumulation (Main fields inferred from the row) ---
            
            current_result = str(row.iloc[idx_final_conc]).strip() if idx_final_conc is not None else ""
            
            # Accumulate data into the r.pdf_url field
            r.pdf_url = r.pdf_url or ""
            
            # Format: Analyte: ResultUnit (Sample Results)
            accumulation_string = f"{sr_analyte}: {current_result} {r.sample_units or ''}"

            if r.pdf_url:
                 if accumulation_string not in r.pdf_url:
                     r.pdf_url += f" | {accumulation_string}"
            else:
                 r.pdf_url = accumulation_string

            # If BPS, set the primary fields
            if sr_analyte.upper() == "BISPHENOL S":
                r.test = sr_analyte
                r.result = current_result
            
            # If PFAS, set the primary fields
            elif is_pfas:
                r.test = "PFAS GROUP"
                r.result = "See Details" 
            
            # Update dilution using the new 'Dil.' column
            if idx_dilution is not None: r.sample_dilution = str(row.iloc[idx_dilution]).strip()


            # --- QC Blocks Accumulation ---
            
            # Fill MB
            if idx_mb_analyte is not None:
                try:
                    mb_analyte_val = str(row.iloc[idx_mb_analyte]).strip()
                    mb_result_val = str(row.iloc[idx_mb_result]).strip() if idx_mb_result is not None else ""
                    mb_mrl_val = str(row.iloc[idx_mb_mrl]).strip() if idx_mb_mrl is not None else ""
                    mb_dilution_val = str(row.iloc[idx_mb_dilution]).strip() if idx_mb_dilution is not None else ""
                    
                    # Store only the accumulation string in r.acq_datetime (repurposed for MB Accumulation)
                    # CRITICAL: Use PIPE (|) as the inner field separator
                    mb_accumulation_string = f"{mb_analyte_val}|{mb_result_val}|{mb_mrl_val}|{r.sample_units or ''}|{mb_dilution_val}"
                    
                    r.acq_datetime = r.acq_datetime or ""
                    if mb_analyte_val and mb_accumulation_string not in r.acq_datetime:
                         if r.acq_datetime:
                             r.acq_datetime += f" | {mb_accumulation_string}"
                         else:
                             r.acq_datetime = mb_accumulation_string

                    # Store the single MB result for display purposes (respecting ND/Blank)
                    if mb_result_val.upper() in ["#VALUE!", "NAN", "NOT FOUND"]:
                        r.mb_result = "" 
                    elif mb_result_val:
                        r.mb_result = mb_result_val 
                    else:
                        r.mb_result = "" 
                        
                    r.mb_analyte = mb_analyte_val
                    r.mb_mrl = mb_mrl_val
                    r.mb_dilution = mb_dilution_val

                except Exception:
                    pass

            # Fill MS1
            if idx_ms1_analyte is not None:
                try:
                    ms1_analyte_val = str(row.iloc[idx_ms1_analyte]).strip()
                    ms1_result_val = str(row.iloc[idx_ms1_result]).strip() if idx_ms1_result is not None else ""
                    ms1_fortified_level_val = str(row.iloc[idx_ms1_fort_level]).strip() if idx_ms1_fort_level is not None else ""
                    ms1_pct_rec_val = str(row.iloc[idx_ms1_pct_rec]).strip() if idx_ms1_pct_rec is not None else ""
                    
                    # Assume MRL and Dilution are the same as MB fields on the same row.
                    ms1_mrl_val = str(row.iloc[idx_mb_mrl]).strip() if idx_mb_mrl is not None else ""
                    ms1_dilution_val = str(row.iloc[idx_mb_dilution]).strip() if idx_mb_dilution is not None else ""

                    # Store only the accumulation string in r.sheet_name (repurposed for MS1 Accumulation)
                    # CRITICAL: Use PIPE (|) as the inner field separator
                    ms1_accumulation_string = f"{ms1_analyte_val}|{ms1_result_val}|{ms1_mrl_val}|{r.sample_units or ''}|{ms1_dilution_val}|{ms1_fortified_level_val}|{ms1_pct_rec_val}"
                    
                    r.sheet_name = r.sheet_name or ""
                    if ms1_analyte_val and ms1_accumulation_string not in r.sheet_name:
                        if r.sheet_name:
                            r.sheet_name += f" | {ms1_accumulation_string}"
                        else:
                            r.sheet_name = ms1_accumulation_string
                        
                    # Also update single-analyte fields (overwritten by last row)
                    r.ms1_analyte = ms1_analyte_val
                    r.ms1_result = ms1_result_val
                    r.ms1_mrl = ms1_mrl_val
                    r.ms1_units = r.sample_units
                    r.ms1_dilution = ms1_dilution_val
                    r.ms1_fortified_level = ms1_fortified_level_val
                    r.ms1_pct_rec = ms1_pct_rec_val

                except Exception:
                    pass

            # Fill MSD (Matrix Spike Duplicate)
            if idx_msd_result is not None:
                try:
                    msd_analyte_val = sr_analyte 
                    msd_result_val = str(row.iloc[idx_msd_result]).strip()
                    msd_pct_rec_val = str(row.iloc[idx_msd_pct_rec]).strip() if idx_msd_pct_rec is not None else ""
                    msd_rpd_val = str(row.iloc[idx_msd_rpd]).strip() if idx_msd_rpd is not None else ""
                    
                    # Inherit/Use MS1 values for shared fields
                    msd_units_val = r.ms1_units
                    msd_dilution_val = r.ms1_dilution
                    msd_pct_rec_limits_val = r.ms1_pct_rec_limits

                    # --- MSD ACCUMULATION LOGIC ---
                    # CRITICAL: Use ms1_pct_rec_limits field for MSD accumulation
                    msd_accumulation_string = f"{msd_analyte_val}|{msd_result_val}|{msd_units_val}|{msd_dilution_val}|{msd_pct_rec_val}|{msd_pct_rec_limits_val}|{msd_rpd_val}"
                    
                    r.ms1_pct_rec_limits = r.ms1_pct_rec_limits or ""
                    if msd_analyte_val and msd_accumulation_string not in r.ms1_pct_rec_limits:
                        if r.ms1_pct_rec_limits:
                            r.ms1_pct_rec_limits += f" | {msd_accumulation_string}"
                        else:
                            r.ms1_pct_rec_limits = msd_accumulation_string
                    
                    # Update single-analyte MSD fields (overwritten by last row)
                    r.msd_analyte = msd_analyte_val
                    r.msd_result = msd_result_val
                    r.msd_pct_rec = msd_pct_rec_val
                    r.msd_pct_rpd = msd_rpd_val
                    r.msd_units = msd_units_val
                    r.msd_dilution = msd_dilution_val
                    r.msd_pct_rec_limits = msd_pct_rec_limits_val

                except Exception:
                    pass

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: Critical database error. Details: {e}"
    finally:
        db.close()

    return (f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_no_sample_name} row(s) with missing Sample Name and {skipped_analyte} non-target analyte row(s).")

@app.route("/audit")
def audit():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    if u["role"] != "admin":
        flash("Admins only.", "error")
        return redirect(url_for("dashboard"))
    db = SessionLocal()
    rows = db.query(AuditLog).order_by(AuditLog.at.desc()).limit(500).all()
    db.close()
    return render_template("audit.html", user=u, rows=rows)

@app.route("/export_csv")
def export_csv():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    q = db.query(Report)
    if u["role"] == "client":
        q = q.filter(Report.client == u["client_name"])
    rows = q.all()
    db.close()

    data = [{
        "Lab ID": r.lab_id,
        "Client": r.client,
        "Analyte": r.test or "",
        "Result": r.result or "",
        "MRL": r.sample_mrl or "",
        "Units": r.sample_units or "",
        "Dilution": r.sample_dilution or "",
        "Analyzed": r.sample_analyzed or "",
        "Qualifier": r.sample_qualifier or "",
        "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
        "Received": r.collected_date.isoformat() if r.collected_date else "",
        # Use the accumulation field for detailed analyte data
        "Analyte Details": r.pdf_url or "", 
    } for r in rows]
    df = pd.DataFrame(data)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(data)} records")
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="reports_export.csv"
    )

# ----------- Health & errors -----------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal Server Error"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
