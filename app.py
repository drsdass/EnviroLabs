import os
import io
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
CLIENT_NAME     = os.getenv("CLIENT_NAME", "Artemis")

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

    # Identity
    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)

    # (compat)
    patient_name = Column(String, nullable=True)

    # Primary analyte for this row
    test = Column(String, nullable=True)      # analyte (BPS or one PFAS)
    result = Column(String, nullable=True)

    collected_date = Column(Date, nullable=True)  # "Received Date"
    resulted_date = Column(Date, nullable=True)   # "Reported Date"
    pdf_url = Column(String, nullable=True)

    # Client / header metadata (all strings)
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

    # Sample Results extras
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
    role = Column(String, nullable=False)  # 'admin' or 'client'
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
            cols.add(row[1])  # name
        missing = needed - cols
        for col in sorted(missing):
            conn.execute(sql_text(f"ALTER TABLE reports ADD COLUMN {col} TEXT"))

_ensure_report_columns()

# ------------------- Helpers -------------------
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

PFAS_SET = {
    "PFOA","PFOS","PFNA","FOSAA","N-MEFOSAA","N-ETFOSAA","SAMPAP","PFOSA",
    "N-MEFOSA","N-MEFOSE","N-ETFOSA","N-ETFOSE","DISAMPAP"
}

def _is_bps(analyte: str) -> bool:
    return analyte and _norm(analyte).replace(" ", "") in {"bisphenols", "bisphenols"}  # tolerant

def _is_pfas_member(analyte: str) -> bool:
    if not analyte:
        return False
    a = _norm(analyte).upper().replace(" ", "")
    return a in PFAS_SET

def _target_analyte_ok(analyte: str) -> bool:
    return _is_bps(analyte) or _is_pfas_member(analyte)

def _safe(row_val) -> str:
    return "" if row_val is None else str(row_val).strip()

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
        q = q.filter(Report.lab_id == lab_id)
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
    return render_template("dashboard.html", user=u, reports=reports)

@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    r = db.query(Report).get(report_id)
    if not r:
        db.close()
        flash("Report not found", "error")
        return redirect(url_for("dashboard"))

    # Pull ALL sibling rows for this Lab ID (PFAS panel => many rows)
    siblings = db.query(Report).filter(Report.lab_id == r.lab_id, Report.client == r.client).order_by(Report.id.asc()).all()
    db.close()

    # Build combined payload
    def v(x): return "" if x is None else str(x)

    # Prefer any sibling with values for client info; fallback to r
    base = next((s for s in siblings if any([s.phone, s.email, s.project_lead, s.address])), r)

    sample_results_rows = []
    qc_rows = []  # aligned by analyte for rendering QC per analyte

    for s in siblings:
        sample_results_rows.append({
            "analyte": v(s.test),
            "result": v(s.result),
            "mrl": v(s.sample_mrl),
            "units": v(s.sample_units),
            "dilution": v(s.sample_dilution),
            "analyzed": v(s.sample_analyzed),
            "qualifier": v(s.sample_qualifier),
        })
        qc_rows.append({
            "analyte": v(s.test),
            "method_blank": {
                "analyte": v(s.mb_analyte), "result": v(s.mb_result),
                "mrl": v(s.mb_mrl), "units": v(s.mb_units), "dilution": v(s.mb_dilution),
            },
            "matrix_spike_1": {
                "analyte": v(s.ms1_analyte), "result": v(s.ms1_result),
                "mrl": v(s.ms1_mrl), "units": v(s.ms1_units), "dilution": v(s.ms1_dilution),
                "fortified_level": v(s.ms1_fortified_level),
                "pct_rec": v(s.ms1_pct_rec),
                "pct_rec_limits": v(s.ms1_pct_rec_limits),
            },
            "matrix_spike_dup": {
                "analyte": v(s.msd_analyte), "result": v(s.msd_result),
                "units": v(s.msd_units), "dilution": v(s.msd_dilution),
                "pct_rec": v(s.msd_pct_rec), "pct_rec_limits": v(s.msd_pct_rec_limits),
                "pct_rpd": v(s.msd_pct_rpd), "pct_rpd_limit": v(s.msd_pct_rpd_limit),
            },
        })

    p = {
        "client_info": {
            "client": v(r.client),
            "phone": v(base.phone),
            "email": v(base.email) or "support@envirolabsusa.com",
            "project_lead": v(base.project_lead),
            "address": v(base.address),
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": v(base.sample_name or r.lab_id),
            "prepared_by": v(base.prepared_by), "matrix": v(base.matrix),
            "prepared_date": v(base.prepared_date), "qualifiers": v(base.qualifiers),
            "asin": v(base.asin), "product_weight_g": v(base.product_weight_g),
        },
        # Back-compat single row (template can keep using it)
        "sample_results": sample_results_rows[0] if sample_results_rows else {
            "analyte": v(r.test), "result": v(r.result),
            "mrl": v(r.sample_mrl), "units": v(r.sample_units),
            "dilution": v(r.sample_dilution), "analyzed": v(r.sample_analyzed),
            "qualifier": v(r.sample_qualifier),
        },
        # New: full multi-row
        "sample_results_rows": sample_results_rows,
        "qc_rows": qc_rows,
        "acq_datetime": v(base.acq_datetime or r.acq_datetime),
        "sheet_name": v(base.sheet_name or r.sheet_name),
    }

    return render_template("report_detail.html", user=u, r=r, p=p)

# ----------- CSV/Excel upload -----------
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

    # Try robust master upload parser first (header might be on a data row)
    try:
        raw = pd.read_csv(saved_path, header=None, dtype=str).fillna("")
    except Exception:
        try:
            raw = pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl").fillna("")
        except Exception as e:
            flash(f"Could not read file: {e}", "error")
            if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
                os.remove(saved_path)
            return redirect(url_for("dashboard"))

    # Find header row (contains "Sample ID")
    header_row_idx = None
    for i in range(min(10, len(raw))):
        row_vals = [str(x) for x in list(raw.iloc[i].values)]
        if any("sample id" in _norm(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        flash("Could not locate header row (looking for 'Sample ID').", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Build DataFrame with real headers from that row
    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    df = df.fillna("")
    # Drop fully empty rows
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]

    msg = _ingest_master_upload(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Master Upload File layout with multiple repeated captions.
    Creates one Report row per analyte (BPS or any PFAS in PFAS_SET).
    Skips Lab IDs that do not start with a digit.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # Single columns
    idx_lab = _find_token_col(cols, "sample", "id")
    idx_client = _find_token_col(cols, "client")
    idx_reported = _find_token_col(cols, "reported")
    idx_received = _find_token_col(cols, "received", "date")
    idx_sample_name = _find_token_col(cols, "sample", "name")
    idx_prepared_by = _find_token_col(cols, "prepared", "by")
    idx_matrix = _find_token_col(cols, "matrix")
    idx_prepared_date = _find_token_col(cols, "prepared", "date")
    idx_qualifiers = _find_token_col(cols, "qualifiers")
    idx_asin = _find_token_col(cols, "asin") or _find_token_col(cols, "identifier")
    idx_weight = _find_token_col(cols, "product", "weight") or _find_token_col(cols, "weight")

    # Client info (explicit)
    idx_phone = _find_token_col(cols, "phone")
    idx_email = _find_token_col(cols, "email")
    idx_project = _find_token_col(cols, "project", "lead")
    idx_address = _find_token_col(cols, "address")

    idx_acq = _find_token_col(cols, "acq", "date")  # "Acq. Date-Time"
    idx_sheet = _find_token_col(cols, "sheetname") or _find_token_col(cols, "sheet", "name")

    # Blocks
    sr_seq = ["analyte", "result", "mrl", "units", "dilution", "analyzed", "qualifier"]
    mb_seq = ["analyte", "result", "mrl", "units", "dilution"]
    ms1_seq = ["analyte", "result", "mrl", "units", "dilution", "fortified level", "%rec", "%rec limits"]
    msd_seq = ["analyte", "result", "units", "dilution", "%rec", "%rec limits", "%rpd", "%rpd limit"]

    # Find starts (by exact order)
    colnames_lower = [c.lower() for c in cols]
    sr_start = _find_sequence(colnames_lower, sr_seq)
    mb_start = _find_sequence(colnames_lower, mb_seq)
    ms1_start = _find_sequence(colnames_lower, ms1_seq)
    msd_start = _find_sequence(colnames_lower, msd_seq)

    created = 0
    updated = 0
    skipped_num = 0
    skipped_analyte = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = _safe(row.iloc[idx_lab]) if idx_lab is not None else ""
            client = _safe(row.iloc[idx_client]) if idx_client is not None else CLIENT_NAME

            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            # Extract sample results block for this row
            sr = {}
            analyte = ""
            if sr_start is not None:
                try:
                    analyte = _safe(row.iloc[sr_start + 0])
                    sr = {
                        "result": _safe(row.iloc[sr_start + 1]),
                        "mrl": _safe(row.iloc[sr_start + 2]),
                        "units": _safe(row.iloc[sr_start + 3]),
                        "dilution": _safe(row.iloc[sr_start + 4]),
                        "analyzed": _safe(row.iloc[sr_start + 5]),
                        "qualifier": _safe(row.iloc[sr_start + 6]),
                    }
                except Exception:
                    analyte = ""
                    sr = {}

            # Only accept BPS or any PFAS in PFAS_SET
            if not _target_analyte_ok(analyte):
                skipped_analyte += 1
                continue

            existing = db.query(Report).filter(
                Report.lab_id == lab_id,
                Report.client == client,
                Report.test == analyte
            ).one_or_none()

            if not existing:
                existing = Report(lab_id=lab_id, client=client, test=analyte)
                db.add(existing)
                created += 1
            else:
                updated += 1

            # Dates
            existing.resulted_date = parse_date(row.iloc[idx_reported]) if idx_reported is not None else existing.resulted_date
            existing.collected_date = parse_date(row.iloc[idx_received]) if idx_received is not None else existing.collected_date

            # Client info
            if idx_phone is not None:   existing.phone = _safe(row.iloc[idx_phone])
            if idx_email is not None:   existing.email = _safe(row.iloc[idx_email])
            if idx_project is not None: existing.project_lead = _safe(row.iloc[idx_project])
            if idx_address is not None: existing.address = _safe(row.iloc[idx_address])

            # Summary
            if idx_sample_name is not None: existing.sample_name = _safe(row.iloc[idx_sample_name]) or existing.sample_name
            if idx_prepared_by is not None: existing.prepared_by = _safe(row.iloc[idx_prepared_by]) or existing.prepared_by
            if idx_matrix is not None:      existing.matrix = _safe(row.iloc[idx_matrix]) or existing.matrix
            if idx_prepared_date is not None: existing.prepared_date = _safe(row.iloc[idx_prepared_date]) or existing.prepared_date
            if idx_qualifiers is not None:  existing.qualifiers = _safe(row.iloc[idx_qualifiers]) or existing.qualifiers
            if idx_asin is not None:        existing.asin = _safe(row.iloc[idx_asin]) or existing.asin
            if idx_weight is not None:      existing.product_weight_g = _safe(row.iloc[idx_weight]) or existing.product_weight_g

            if idx_acq is not None:   existing.acq_datetime = _safe(row.iloc[idx_acq]) or existing.acq_datetime
            if idx_sheet is not None: existing.sheet_name = _safe(row.iloc[idx_sheet]) or existing.sheet_name

            # Sample results
            existing.result = sr.get("result", existing.result)
            existing.sample_mrl = sr.get("mrl", existing.sample_mrl)
            existing.sample_units = sr.get("units", existing.sample_units)
            existing.sample_dilution = sr.get("dilution", existing.sample_dilution)
            existing.sample_analyzed = sr.get("analyzed", existing.sample_analyzed)
            existing.sample_qualifier = sr.get("qualifier", existing.sample_qualifier)

            # QC blocks for THIS analyte row
            if mb_start is not None:
                try:
                    existing.mb_analyte  = _safe(row.iloc[mb_start + 0])
                    existing.mb_result   = _safe(row.iloc[mb_start + 1])
                    existing.mb_mrl      = _safe(row.iloc[mb_start + 2])
                    existing.mb_units    = _safe(row.iloc[mb_start + 3])
                    existing.mb_dilution = _safe(row.iloc[mb_start + 4])
                except Exception:
                    pass

            if ms1_start is not None:
                try:
                    existing.ms1_analyte         = _safe(row.iloc[ms1_start + 0])
                    existing.ms1_result          = _safe(row.iloc[ms1_start + 1])
                    existing.ms1_mrl             = _safe(row.iloc[ms1_start + 2])
                    existing.ms1_units           = _safe(row.iloc[ms1_start + 3])
                    existing.ms1_dilution        = _safe(row.iloc[ms1_start + 4])
                    existing.ms1_fortified_level = _safe(row.iloc[ms1_start + 5])
                    existing.ms1_pct_rec         = _safe(row.iloc[ms1_start + 6])
                    existing.ms1_pct_rec_limits  = _safe(row.iloc[ms1_start + 7])
                except Exception:
                    pass

            if msd_start is not None:
                try:
                    existing.msd_analyte        = _safe(row.iloc[msd_start + 0])
                    existing.msd_result         = _safe(row.iloc[msd_start + 1])
                    existing.msd_units          = _safe(row.iloc[msd_start + 2])
                    existing.msd_dilution       = _safe(row.iloc[msd_start + 3])
                    existing.msd_pct_rec        = _safe(row.iloc[msd_start + 4])
                    existing.msd_pct_rec_limits = _safe(row.iloc[msd_start + 5])
                    existing.msd_pct_rpd        = _safe(row.iloc[msd_start + 6])
                    existing.msd_pct_rpd_limit  = _safe(row.iloc[msd_start + 7])
                except Exception:
                    pass

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
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
