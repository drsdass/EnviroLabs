import os
import io
import re # <-- NEW: Needed for robust Lab ID normalization
from datetime import datetime, date
from typing import List, Optional

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
    test = Column(String, nullable=True)     # analyte (e.g., "Bisphenol S", "PFOA", ...)
    result = Column(String, nullable=True)   # numeric-as-text or textual

    collected_date = Column(Date, nullable=True) # "Received Date"
    resulted_date = Column(Date, nullable=True)  # "Reported Date"
    pdf_url = Column(String, nullable=True)

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

    # QC: Matrix Spike 1
    ms1_analyte = Column(String, nullable=True)
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)

    # QC: Matrix Spike Duplicate
    msd_analyte = Column(String, nullable=True)
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)

    # Misc
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)

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
        # meta
        "phone", "email", "project_lead", "address", "sample_name", "prepared_by",
        "matrix", "prepared_date", "qualifiers", "asin", "product_weight_g",
        # sample extras
        "sample_mrl", "sample_units", "sample_dilution", "sample_analyzed", "sample_qualifier",
        # MB
        "mb_analyte", "mb_result", "mb_mrl", "mb_units", "mb_dilution",
        # MS1
        "ms1_analyte", "ms1_result", "ms1_mrl", "ms1_units", "ms1_dilution",
        "ms1_fortified_level", "ms1_pct_rec", "ms1_pct_rec_limits",
        # MSD
        "msd_analyte", "msd_result", "msd_units", "msd_dilution",
        "msd_pct_rec", "msd_pct_rec_limits", "msd_pct_rpd", "msd_pct_rpd_limit",
        # misc
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

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

# --- MODIFIED HELPER FUNCTION: Using Regex for Robust Normalization ---
def _normalize_lab_id(lab_id: str) -> str:
    """
    Removes common Lab ID suffixes like ' 0.5ppb', ' 1ppb', ' 5ppb', ' 1', etc.
    using regular expressions for robust stripping.
    """
    s = (lab_id or "").strip()
    if not s:
        return s
        
    # Regex to find suffixes: 
    # Starts with a space, followed by an optional sign, a number (optional decimal), 
    # and then optional units (ppb, ppt, ng/g, etc.) until the end of the string.
    # This also handles simple number suffixes like " -1"
    pattern = r'\s+[\-\+]?\d*\.?\d+(?:ppb|ppt|ng\/g|ug\/g|\s\d*)?$'
    
    # Apply regex substitution to remove the matched suffix
    normalized = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
    
    # If the normalized string is not empty, return it; otherwise, return the original
    if not normalized:
        return s
        
    return normalized

def _target_analyte_ok(analyte: str) -> bool:
    if analyte is None:
        return False
    a = analyte.strip().upper()
    return (a == "BISPHENOL S") or (a in PFAS_SET_UPPER)

# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
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
def logout():
    u = current_user()
    if u["username"]:
        log_action(u["username"], u["role"] or "unknown", "logout", "User logged out")
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
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

    if lab_id:
        # Apply normalization for searching too
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
        reports = q.order_by(Report.resulted_date.desc(), Report.id.desc()).limit(500).all()

    db.close()
    
    # Passing reports as 'report_summaries' to match the user's working dashboard template
    return render_template("dashboard.html", user=u, report_summaries=reports)

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

    p = {
        "client_info": {
            "client": val(r.client),
            "phone": val(r.phone),
            "email": val(r.email) or "support@envirolabsusa.com",
            "project_lead": val(r.project_lead),
            "address": val(r.address),
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": val(r.sample_name or r.lab_id),
            "prepared_by": val(r.prepared_by),
            "matrix": val(r.matrix),
            "prepared_date": val(r.prepared_date),
            "qualifiers": val(r.qualifiers),
            "asin": val(r.asin),
            "product_weight_g": val(r.product_weight_g),
        },
        "sample_results": {
            "analyte": val(r.test), "result": val(r.result),
            "mrl": val(r.sample_mrl), "units": val(r.sample_units),
            "dilution": val(r.sample_dilution), "analyzed": val(r.sample_analyzed),
            "qualifier": val(r.sample_qualifier),
        },
        "method_blank": {
            "analyte": val(r.mb_analyte), "result": val(r.mb_result),
            "mrl": val(r.mb_mrl), "units": val(r.mb_units), "dilution": val(r.mb_dilution),
        },
        "matrix_spike_1": {
            "analyte": val(r.ms1_analyte), "result": val(r.ms1_result),
            "mrl": val(r.ms1_mrl), "units": val(r.ms1_units), "dilution": val(r.ms1_dilution),
            "fortified_level": val(r.ms1_fortified_level), "pct_rec": val(r.ms1_pct_rec),
            "pct_rec_limits": val(r.ms1_pct_rec_limits),
        },
        "matrix_spike_dup": {
            "analyte": val(r.msd_analyte), "result": val(r.msd_result),
            "units": val(r.msd_units), "dilution": val(r.msd_dilution),
            "pct_rec": val(r.msd_pct_rec), "pct_rec_limits": val(r.msd_pct_rec_limits),
            "pct_rpd": val(r.msd_pct_rpd), "pct_rpd_limit": val(r.msd_pct_rpd_limit),
        },
        "acq_datetime": val(r.acq_datetime),
        "sheet_name": val(r.sheet_name),
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
    
    # FIX for two-row header: Explicitly set header_row_idx to 1 (the second row)
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
    Parse the Master Upload File layout with repeated blocks.
    NEW LOGIC: Normalizes Lab ID and groups all PFAS/BPS analytes for a single Lab ID into one Report.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # ---- Prefer exact column names from your sheet ----
    def ex(name): return _find_exact(cols, name)

    idx_lab             = ex("Sample ID (Lab ID, Laboratory ID)") or _find_token_col(cols, "sample", "id")
    idx_client         = ex("Client") or _find_token_col(cols, "client")
    idx_phone          = ex("Phone") or _find_token_col(cols, "phone")
    idx_email          = ex("Email") or _find_token_col(cols, "email")
    idx_project_lead = ex("Project Lead") or _find_token_col(cols, "project", "lead")
    idx_address      = ex("Address") or _find_token_col(cols, "address")

    idx_reported       = ex("Reported") or _find_token_col(cols, "reported")
    idx_received       = ex("Received Date") or _find_token_col(cols, "received", "date")
    idx_sample_name    = ex("Sample Name") or _find_token_col(cols, "sample", "name")
    idx_prepared_by    = ex("Prepared By") or _find_token_col(cols, "prepared", "by")
    idx_matrix         = ex("Matrix") or _find_token_col(cols, "matrix")
    idx_prepared_date= ex("Prepared Date") or _find_token_col(cols, "prepared", "date")
    idx_qualifiers     = ex("Qualifiers") or _find_token_col(cols, "qualifiers")
    idx_asin           = ex("ASIN (Identifier)") or _find_token_col(cols, "asin")
    idx_weight         = ex("Product Weight (Grams)") or _find_token_col(cols, "product", "weight")

    idx_acq            = ex("Acq. Date-Time") or _find_token_col(cols, "acq", "date")
    idx_sheet          = ex("SheetName") or _find_token_col(cols, "sheetname")

    # Block starts 
    sr_seq  = ["Analyte","Result","MRL","Units","Dilution","Analyzed","Qualifier"]
    mb_seq  = ["Analyte","Result","MRL","Units","Dilution"]
    ms1_seq = ["Analyte","Result","MRL","Units","Dilution","Fortified Level","%REC","%REC Limits"]
    msd_seq = ["Analyte","Result","Units","Dilution","%REC","%REC Limits","%RPD","%RPD Limit"]

    sr_start  = _find_sequence(cols, sr_seq)
    mb_start  = _find_sequence(cols, mb_seq)
    ms1_start = _find_sequence(cols, ms1_seq)
    msd_start = _find_sequence(cols, msd_seq)

    created = 0
    updated = 0
    skipped_num = 0
    skipped_analyte = 0
    
    db = SessionLocal()
    report_data = {} # Dictionary to hold Report objects by normalized lab_id

    try:
        for _, row in df.iterrows():
            # Original Lab ID (e.g., "250819-B11 0.5ppb")
            original_lab_id = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""
            
            # --- CRITICAL FIX 1: Normalize Lab ID ---
            lab_id = _normalize_lab_id(original_lab_id)
            
            client = str(row.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME
            
            # Use the normalized ID for skipping checks
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            # Sample Results (per-row analyte)
            sr_analyte = ""
            if sr_start is not None:
                try:
                    sr_analyte = str(row.iloc[sr_start + 0]).strip()
                except Exception:
                    sr_analyte = ""

            # Only process target analytes
            if not _target_analyte_ok(sr_analyte):
                skipped_analyte += 1
                continue

            # --- CRITICAL FIX 2: Grouping Logic ---
            db_key = lab_id 
            
            # 1. Check if we already created a Report object for this Lab ID in this session
            existing = report_data.get(db_key)
            if not existing:
                # 2. If not in session, check the database for the unique normalized ID + placeholder
                existing = db.query(Report).filter(
                    Report.lab_id == db_key,
                    Report.test == "PFAS_BPS_GROUP" 
                ).one_or_none()
                
                if not existing:
                    # 3. If not in DB, create a new one using the placeholder test name
                    existing = Report(lab_id=db_key, client=client, test="PFAS_BPS_GROUP")
                    db.add(existing)
                    created += 1
                else:
                    updated += 1
                
                # Store it in the session dictionary for subsequent rows
                report_data[db_key] = existing

            r = existing # Report object for the current normalized Lab ID

            # Client info (updated/overwritten for every row)
            r.client = client
            if idx_phone is not None:        r.phone = str(row.iloc[idx_phone]).strip()
            if idx_email is not None:        r.email = str(row.iloc[idx_email]).strip()
            if idx_project_lead is not None: r.project_lead = str(row.iloc[idx_project_lead]).strip()
            if idx_address is not None:      r.address = str(row.iloc[idx_address]).strip()

            # Dates (updated/overwritten for every row)
            if idx_reported is not None: r.resulted_date = parse_date(row.iloc[idx_reported])
            if idx_received is not None: r.collected_date = parse_date(row.iloc[idx_received])

            # Sample summary (updated/overwritten for every row)
            r.sample_name = (str(row.iloc[idx_sample_name]).strip()
                             if idx_sample_name is not None else (r.sample_name or original_lab_id))
            if idx_prepared_by is not None:   r.prepared_by  = str(row.iloc[idx_prepared_by]).strip()
            if idx_matrix is not None:        r.matrix       = str(row.iloc[idx_matrix]).strip()
            if idx_prepared_date is not None: r.prepared_date= str(row.iloc[idx_prepared_date]).strip()
            if idx_qualifiers is not None:    r.qualifiers   = str(row.iloc[idx_qualifiers]).strip()
            if idx_asin is not None:          r.asin         = str(row.iloc[idx_asin]).strip()
            if idx_weight is not None:        r.product_weight_g = str(row.iloc[idx_weight]).strip()

            if idx_acq is not None:           r.acq_datetime = str(row.iloc[idx_acq]).strip()
            if idx_sheet is not None:         r.sheet_name   = str(row.iloc[idx_sheet]).strip()

            
            # --- Fixed Logic for Analyte Data Accumulation (prevents TypeError) ---
            if sr_start is not None:
                try:
                    current_result   = str(row.iloc[sr_start + 1]).strip()
                    current_mrl      = str(row.iloc[sr_start + 2]).strip()
                    current_units    = str(row.iloc[sr_start + 3]).strip()
                    current_dilution = str(row.iloc[sr_start + 4]).strip()
                    current_analyzed = str(row.iloc[sr_start + 5]).strip()
                    current_qualifier= str(row.iloc[sr_start + 6]).strip()
                    
                    # Ensure fields are strings/initialized before checking/concatenating
                    r.sample_name = r.sample_name or ""
                    r.test = r.test or "PFAS_BPS_GROUP"

                    if r.test == "PFAS_BPS_GROUP":
                        # FIRST ANALYTE: Set main fields
                        r.test = sr_analyte 
                        r.result = current_result
                        r.sample_mrl = current_mrl
                        r.sample_units = current_units
                        r.sample_dilution = current_dilution
                        r.sample_analyzed = current_analyzed
                        r.sample_qualifier = current_qualifier
                    else:
                        # SUBSEQUENT ANALYTES: Append to the sample name field
                        # This line was the source of the TypeError/silent failure.
                        r.sample_name += f" | {sr_analyte}: {current_result}{current_units}"

                except Exception:
                    # Pass on any indexing or string conversion errors for this analyte block
                    pass

            # Fill QC Blocks (Overwriting on each row, using the current row's QC data)
            # This is acceptable because QC data is generally repeated across analyte rows for a sample.

            # Fill MB
            if mb_start is not None:
                try:
                    r.mb_analyte  = str(row.iloc[mb_start + 0]).strip()
                    r.mb_result   = str(row.iloc[mb_start + 1]).strip()
                    r.mb_mrl      = str(row.iloc[mb_start + 2]).strip()
                    r.mb_units    = str(row.iloc[mb_start + 3]).strip()
                    r.mb_dilution = str(row.iloc[mb_start + 4]).strip()
                except Exception:
                    pass

            # Fill MS1
            if ms1_start is not None:
                try:
                    r.ms1_analyte         = str(row.iloc[ms1_start + 0]).strip()
                    r.ms1_result          = str(row.iloc[ms1_start + 1]).strip()
                    r.ms1_mrl             = str(row.iloc[ms1_start + 2]).strip()
                    r.ms1_units           = str(row.iloc[ms1_start + 3]).strip()
                    r.ms1_dilution        = str(row.iloc[ms1_start + 4]).strip()
                    r.ms1_fortified_level = str(row.iloc[ms1_start + 5]).strip()
                    r.ms1_pct_rec         = str(row.iloc[ms1_start + 6]).strip()
                    r.ms1_pct_rec_limits  = str(row.iloc[ms1_start + 7]).strip()
                except Exception:
                    pass

            # Fill MSD
            if msd_start is not None:
                try:
                    r.msd_analyte       = str(row.iloc[msd_start + 0]).strip()
                    r.msd_result        = str(row.iloc[msd_start + 1]).strip()
                    r.msd_units         = str(row.iloc[msd_start + 2]).strip()
                    r.msd_dilution      = str(row.iloc[msd_start + 3]).strip()
                    r.msd_pct_rec       = str(row.iloc[msd_start + 4]).strip()
                    r.msd_pct_rec_limits = str(row.iloc[msd_start + 5]).strip()
                    r.msd_pct_rpd       = str(row.iloc[msd_start + 6]).strip()
                    r.msd_pct_rpd_limit = str(row.iloc[msd_start + 7]).strip()
                except Exception:
                    pass

        db.commit()
    except Exception as e:
        db.rollback()
        # Ensure a specific error message is returned if a failure occurs
        return f"Import failed: Critical database error. Details: {e}"
    finally:
        db.close()

    return (f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_num} non-numeric Lab ID row(s) and {skipped_analyte} non-target analyte row(s).")

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
        "PDF URL": r.pdf_url or "",
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
