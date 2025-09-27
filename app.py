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

    # Core identity / simple fields
    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)

    # (kept for compatibility but we won’t use patient fields)
    patient_name = Column(String, nullable=True)

    # Sample Results (primary)
    test = Column(String, nullable=True)      # analyte (e.g., "Bisphenol S", "PFAS")
    result = Column(String, nullable=True)    # numeric-as-text or textual

    collected_date = Column(Date, nullable=True)  # "Received Date"
    resulted_date = Column(Date, nullable=True)   # "Reported Date"
    pdf_url = Column(String, nullable=True)

    # ---- New optional metadata fields (all strings to keep SQLite simple) ----
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

def _target_analyte_ok(analyte: str) -> bool:
    if analyte is None:
        return False
    a = analyte.strip().upper()
    return a in {"BISPHENOL S", "PFAS"}

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
            "phone": val(r.phone), "email": val(r.email) or "mike@centerforconsumersafety.com",
            "project_lead": val(r.project_lead) or "Mike White", "address": val(r.address) or "2001 Addison St Berkeley, CA 94704",
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": val(r.sample_name or r.lab_id),
            "prepared_by": val(r.prepared_by), "matrix": val(r.matrix),
            "prepared_date": val(r.prepared_date), "qualifiers": val(r.qualifiers),
            "asin": val(r.asin), "product_weight_g": val(r.product_weight_g),
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

    # Find header row (the one containing "Sample ID")
    header_row_idx = None
    for i in range(min(10, len(raw))):
        row_vals = [str(x) for x in list(raw.iloc[i].values)]
        if any("sample id" in _norm(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        # Fallback to your old generic single-table importer
        df = _fallback_simple_table(saved_path)
        if isinstance(df, str):
            flash(df, "error")
            if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
                os.remove(saved_path)
            return redirect(url_for("dashboard"))
        processed_msg = _ingest_simple(df, u, filename)
        flash(processed_msg, "success")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Build DataFrame with real headers from that row
    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    # Drop fully empty rows
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]

    msg = _ingest_master_upload(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _fallback_simple_table(path) -> pd.DataFrame | str:
    """Old generic single-table fallback. Returns df or error string."""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        try:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")
        except Exception as e:
            return f"Could not read file: {e}"
    df = df.fillna("").copy()
    if df.empty:
        return "No rows found."
    return df

def _ingest_simple(df: pd.DataFrame, u, filename: str) -> str:
    """Very old flow: requires Lab ID and Client present in headers."""
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    c_lab = _find_token_col(cols, "lab", "id") or _find_token_col(cols, "sample", "id") or _find_token_col(cols, "sample")
    c_client = _find_token_col(cols, "client")
    if c_lab is None or c_client is None:
        preview = ", ".join(cols[:20])
        return ("CSV must include Lab ID (aka 'Sample ID') and Client columns. "
                f"Found columns: {preview}")

    created = 0
    updated = 0
    skipped_num = 0
    skipped_analyte = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.iloc[c_lab]).strip()
            client = str(row.iloc[c_client]).strip() or CLIENT_NAME
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue
            # We don't have analyte columns reliably here—accept
            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1
        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_num} non-numeric Lab ID row(s) and {skipped_analyte} non-target analyte row(s).")

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Master Upload File layout with multiple repeated captions.
    Only create reports where:
      - Lab ID starts with a digit, AND
      - Sample Results Analyte is 'Bisphenol S' or 'PFAS' (case-insensitive)
    QC sections are taken as-is from the row (Sheet computed values).
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # Locate key single columns
    idx_lab = _find_token_col(cols, "sample", "id")  # "Sample ID (Lab ID, Laboratory ID)"
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

    idx_acq = _find_token_col(cols, "acq", "date")  # "Acq. Date-Time"
    idx_sheet = _find_token_col(cols, "sheetname") or _find_token_col(cols, "sheet", "name")

    # Find the blocks by column sequences
    # Sample Results block has 7 captions in a row:
    sr_seq = ["analyte", "result", "mrl", "units", "dilution", "analyzed", "qualifier"]
    sr_start = _find_sequence([c.lower() for c in cols], sr_seq)

    # Method Blank
    mb_seq = ["analyte", "result", "mrl", "units", "dilution"]
    mb_start = _find_sequence([c.lower() for c in cols], mb_seq)

    # Matrix Spike 1
    ms1_seq = ["analyte", "result", "mrl", "units", "dilution", "fortified level", "%rec", "%rec limits"]
    ms1_start = _find_sequence([c.lower() for c in cols], ms1_seq)

    # Matrix Spike Duplicate
    msd_seq = ["analyte", "result", "units", "dilution", "%rec", "%rec limits", "%rpd", "%rpd limit"]
    msd_start = _find_sequence([c.lower() for c in cols], msd_seq)

    created = 0
    updated = 0
    skipped_num = 0
    skipped_analyte = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""
            client = str(row.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME

            if not _lab_id_is_numericish(lab_id):
                # This will naturally skip "Method Blank", "Calibration Blank", etc.
                skipped_num += 1
                continue

            # Sample Results analyte (must be Bisphenol S or PFAS)
            sr_analyte = ""
            sr_values = {}
            if sr_start is not None:
                try:
                    sr_analyte = str(row.iloc[sr_start + 0]).strip()
                    sr_values = {
                        "result": str(row.iloc[sr_start + 1]).strip(),
                        "mrl": str(row.iloc[sr_start + 2]).strip(),
                        "units": str(row.iloc[sr_start + 3]).strip(),
                        "dilution": str(row.iloc[sr_start + 4]).strip(),
                        "analyzed": str(row.iloc[sr_start + 5]).strip(),
                        "qualifier": str(row.iloc[sr_start + 6]).strip(),
                    }
                except Exception:
                    sr_values = {}
                    sr_analyte = ""
            # Filter on analyte
            if not _target_analyte_ok(sr_analyte):
                skipped_analyte += 1
                continue

            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

            # Top info
            existing.sample_name = str(row.iloc[idx_sample_name]).strip() if idx_sample_name is not None else lab_id
            existing.phone = ""  # optional; add mapping if you include it in CSV
            existing.email = ""  # default shown in template if missing
            existing.project_lead = ""
            existing.address = ""

            # Dates
            existing.resulted_date = parse_date(row.iloc[idx_reported]) if idx_reported is not None else None
            existing.collected_date = parse_date(row.iloc[idx_received]) if idx_received is not None else None

            existing.prepared_by = str(row.iloc[idx_prepared_by]).strip() if idx_prepared_by is not None else ""
            existing.matrix = str(row.iloc[idx_matrix]).strip() if idx_matrix is not None else ""
            existing.prepared_date = str(row.iloc[idx_prepared_date]).strip() if idx_prepared_date is not None else ""
            existing.qualifiers = str(row.iloc[idx_qualifiers]).strip() if idx_qualifiers is not None else ""
            existing.asin = str(row.iloc[idx_asin]).strip() if idx_asin is not None else ""
            existing.product_weight_g = str(row.iloc[idx_weight]).strip() if idx_weight is not None else ""

            existing.acq_datetime = str(row.iloc[idx_acq]).strip() if idx_acq is not None else ""
            existing.sheet_name = str(row.iloc[idx_sheet]).strip() if idx_sheet is not None else ""

            # Sample Results -> primary fields
            existing.test = sr_analyte
            existing.result = sr_values.get("result", "")
            existing.sample_mrl = sr_values.get("mrl", "")
            existing.sample_units = sr_values.get("units", "")
            existing.sample_dilution = sr_values.get("dilution", "")
            existing.sample_analyzed = sr_values.get("analyzed", "")
            existing.sample_qualifier = sr_values.get("qualifier", "")

            # Method Blank
            if mb_start is not None:
                try:
                    existing.mb_analyte  = str(row.iloc[mb_start + 0]).strip()
                    existing.mb_result   = str(row.iloc[mb_start + 1]).strip()
                    existing.mb_mrl      = str(row.iloc[mb_start + 2]).strip()
                    existing.mb_units    = str(row.iloc[mb_start + 3]).strip()
                    existing.mb_dilution = str(row.iloc[mb_start + 4]).strip()
                except Exception:
                    pass

            # Matrix Spike 1
            if ms1_start is not None:
                try:
                    existing.ms1_analyte         = str(row.iloc[ms1_start + 0]).strip()
                    existing.ms1_result          = str(row.iloc[ms1_start + 1]).strip()
                    existing.ms1_mrl             = str(row.iloc[ms1_start + 2]).strip()
                    existing.ms1_units           = str(row.iloc[ms1_start + 3]).strip()
                    existing.ms1_dilution        = str(row.iloc[ms1_start + 4]).strip()
                    existing.ms1_fortified_level = str(row.iloc[ms1_start + 5]).strip()
                    existing.ms1_pct_rec         = str(row.iloc[ms1_start + 6]).strip()
                    existing.ms1_pct_rec_limits  = str(row.iloc[ms1_start + 7]).strip()
                except Exception:
                    pass

            # Matrix Spike Duplicate
            if msd_start is not None:
                try:
                    existing.msd_analyte       = str(row.iloc[msd_start + 0]).strip()
                    existing.msd_result        = str(row.iloc[msd_start + 1]).strip()
                    existing.msd_units         = str(row.iloc[msd_start + 2]).strip()
                    existing.msd_dilution      = str(row.iloc[msd_start + 3]).strip()
                    existing.msd_pct_rec       = str(row.iloc[msd_start + 4]).strip()
                    existing.msd_pct_rec_limits= str(row.iloc[msd_start + 5]).strip()
                    existing.msd_pct_rpd       = str(row.iloc[msd_start + 6]).strip()
                    existing.msd_pct_rpd_limit = str(row.iloc[msd_start + 7]).strip()
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
