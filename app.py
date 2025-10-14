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

# ====== Main report header (one row per Lab ID & client) ======
class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)

    # identity
    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)

    # legacy fields (ok to keep)
    patient_name = Column(String, nullable=True)

    # a “lead” analyte/result (we still keep for backwards-compat)
    test = Column(String, nullable=True)
    result = Column(String, nullable=True)

    collected_date = Column(Date, nullable=True)
    resulted_date  = Column(Date, nullable=True)
    pdf_url = Column(String, nullable=True)

    # Metadata visible at the top of the report
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

    # legacy single-analyte extras (still present for back-compat)
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)

    # legacy QC (single)
    mb_analyte = Column(String, nullable=True)
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)

    ms1_analyte = Column(String, nullable=True)
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)

    msd_analyte = Column(String, nullable=True)
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)

    # misc
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ====== NEW: child table — many analytes per Report ======
class ReportResult(Base):
    __tablename__ = "report_results"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, index=True, nullable=False)
    analyte = Column(String, nullable=False)

    # sample results
    result = Column(String, nullable=True)
    mrl = Column(String, nullable=True)
    units = Column(String, nullable=True)
    dilution = Column(String, nullable=True)
    analyzed = Column(String, nullable=True)
    qualifier = Column(String, nullable=True)

    # QC: Method Blank
    mb_analyte  = Column(String, nullable=True)
    mb_result   = Column(String, nullable=True)
    mb_mrl      = Column(String, nullable=True)
    mb_units    = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)

    # QC: Matrix Spike 1
    ms1_analyte         = Column(String, nullable=True)
    ms1_result          = Column(String, nullable=True)
    ms1_mrl             = Column(String, nullable=True)
    ms1_units           = Column(String, nullable=True)
    ms1_dilution        = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec         = Column(String, nullable=True)
    ms1_pct_rec_limits  = Column(String, nullable=True)

    # QC: Matrix Spike Duplicate
    msd_analyte        = Column(String, nullable=True)
    msd_result         = Column(String, nullable=True)
    msd_units          = Column(String, nullable=True)
    msd_dilution       = Column(String, nullable=True)
    msd_pct_rec        = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd        = Column(String, nullable=True)
    msd_pct_rpd_limit  = Column(String, nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'admin' or 'client'
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# --- add columns if the DB was created earlier without the new fields ---
def _ensure_report_columns():
    needed = {
        "phone","email","project_lead","address","sample_name","prepared_by",
        "matrix","prepared_date","qualifiers","asin","product_weight_g",
        "sample_mrl","sample_units","sample_dilution","sample_analyzed","sample_qualifier",
        "mb_analyte","mb_result","mb_mrl","mb_units","mb_dilution",
        "ms1_analyte","ms1_result","ms1_mrl","ms1_units","ms1_dilution",
        "ms1_fortified_level","ms1_pct_rec","ms1_pct_rec_limits",
        "msd_analyte","msd_result","msd_units","msd_dilution",
        "msd_pct_rec","msd_pct_rec_limits","msd_pct_rpd","msd_pct_rpd_limit",
        "acq_datetime","sheet_name",
    }
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(reports)"))}
        for col in sorted(needed - cols):
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
        db.add(AuditLog(username=username or "system", role=role or "system",
                        action=action, details=details))
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
        if pd.isna(ts): return None
        return ts.date()
    except Exception:
        return None

def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _safe(x) -> str:
    return "" if x is None else str(x).strip()

def _find_token_col(cols: List[str], *needles: str) -> Optional[int]:
    tokens = [t.lower() for t in needles]
    for i, c in enumerate(cols):
        name = _norm(c)
        if all(tok in name for tok in tokens):
            return i
    return None

def _find_sequence(cols: List[str], seq: List[str]) -> Optional[int]:
    n, m = len(cols), len(seq)
    seq_l = [s.lower() for s in seq]
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            if cols[i + j].strip().lower() != seq_l[j]:
                ok = False; break
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
def _is_target_analyte(analyte: str) -> bool:
    if not analyte: return False
    a = analyte.strip().upper()
    if a == "BISPHENOL S": return True
    return a in PFAS_SET

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

    lab_id_q = request.args.get("lab_id", "").strip() or None
    start_q = request.args.get("start", "").strip()
    end_q   = request.args.get("end", "").strip()

    sd = parse_date(start_q) if start_q else None
    ed = parse_date(end_q) if end_q else None
    sd_iso = sd.isoformat() if sd else None
    ed_iso = ed.isoformat() if ed else None

    # One row per Lab ID (smallest id as canonical)
    sql = """
      SELECT * FROM reports
      WHERE id IN (
        SELECT MIN(id) FROM reports
        WHERE (:lab_id IS NULL OR lab_id = :lab_id)
          AND (:start IS NULL OR COALESCE(resulted_date,'0001-01-01') >= :start)
          AND (:end   IS NULL OR COALESCE(resulted_date,'9999-12-31') <= :end)
        GROUP BY lab_id, client
      )
      ORDER BY COALESCE(resulted_date,'0001-01-01') DESC, id DESC
      LIMIT 500
    """
    params = {"lab_id": lab_id_q, "start": sd_iso, "end": ed_iso}

    db = SessionLocal()
    try:
        rows = db.execute(sql_text(sql), params)
        reports = []
        for r in rows:
            obj = Report()
            for k, v in r._mapping.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            reports.append(obj)
    finally:
        db.close()

    return render_template("dashboard.html", user=u, reports=reports)

@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    try:
        header = db.query(Report).get(report_id)
        if not header:
            flash("Report not found", "error")
            return redirect(url_for("dashboard"))

        # New style: many analytes as children
        children = db.query(ReportResult).filter(
            ReportResult.report_id == header.id
        ).order_by(ReportResult.id.asc()).all()

        def v(x): return "" if x is None else str(x)

        sample_rows = []
        qc_rows = []

        if children:
            for c in children:
                sample_rows.append({
                    "analyte": v(c.analyte),
                    "result": v(c.result),
                    "mrl": v(c.mrl),
                    "units": v(c.units),
                    "dilution": v(c.dilution),
                    "analyzed": v(c.analyzed),
                    "qualifier": v(c.qualifier),
                })
                qc_rows.append({
                    "analyte": v(c.analyte),
                    "method_blank": {
                        "analyte": v(c.mb_analyte), "result": v(c.mb_result),
                        "mrl": v(c.mb_mrl), "units": v(c.mb_units), "dilution": v(c.mb_dilution),
                    },
                    "matrix_spike_1": {
                        "analyte": v(c.ms1_analyte), "result": v(c.ms1_result),
                        "mrl": v(c.ms1_mrl), "units": v(c.ms1_units), "dilution": v(c.ms1_dilution),
                        "fortified_level": v(c.ms1_fortified_level),
                        "pct_rec": v(c.ms1_pct_rec),
                        "pct_rec_limits": v(c.ms1_pct_rec_limits),
                    },
                    "matrix_spike_dup": {
                        "analyte": v(c.msd_analyte), "result": v(c.msd_result),
                        "units": v(c.msd_units), "dilution": v(c.msd_dilution),
                        "pct_rec": v(c.msd_pct_rec), "pct_rec_limits": v(c.msd_pct_rec_limits),
                        "pct_rpd": v(c.msd_pct_rpd), "pct_rpd_limit": v(c.msd_pct_rpd_limit),
                    },
                })
        else:
            # Back-compat: old DB (one row per analyte)
            siblings = db.query(Report).filter(
                Report.lab_id == header.lab_id,
                Report.client == header.client
            ).order_by(Report.id.asc()).all()
            for s in siblings:
                sample_rows.append({
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

        # Single row used by existing sections (kept)
        single = (sample_rows[0] if sample_rows else {
            "analyte": v(header.test), "result": v(header.result),
            "mrl": v(header.sample_mrl), "units": v(header.sample_units),
            "dilution": v(header.sample_dilution), "analyzed": v(header.sample_analyzed),
            "qualifier": v(header.sample_qualifier),
        })

        p = {
            "client_info": {
                "client": v(header.client),
                "phone": v(header.phone),
                "email": v(header.email) or "support@envirolabsusa.com",
                "project_lead": v(header.project_lead),
                "address": v(header.address),
            },
            "sample_summary": {
                "reported": header.resulted_date.isoformat() if header.resulted_date else "",
                "received_date": header.collected_date.isoformat() if header.collected_date else "",
                "sample_name": v(header.sample_name or header.lab_id),
                "prepared_by": v(header.prepared_by), "matrix": v(header.matrix),
                "prepared_date": v(header.prepared_date), "qualifiers": v(header.qualifiers),
                "asin": v(header.asin), "product_weight_g": v(header.product_weight_g),
            },
            "sample_results": single,
            "sample_results_rows": sample_rows,
            "qc_rows": qc_rows,
            "acq_datetime": v(header.acq_datetime),
            "sheet_name": v(header.sheet_name),
        }

        return render_template("report_detail.html", user=u, r=header, p=p)

    except Exception as e:
        log_action(u.get("username"), u.get("role"), "report_detail_error",
                   f"report_id={report_id} err={e}")
        flash("Could not render report.", "error")
        return redirect(url_for("dashboard"))
    finally:
        db.close()

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

    # Try to read with unknown header row (banner on row 0, headers on row 1)
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

    # Detect header row (the one that contains "Sample ID")
    header_row_idx = None
    for i in range(min(10, len(raw))):
        row_vals = [str(x) for x in list(raw.iloc[i].values)]
        if any("sample id" in _norm(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        flash("Could not find the header row (looking for 'Sample ID').", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Build a DataFrame from that header row
    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]  # drop empty rows

    msg = _ingest_master_upload(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Master Upload File.
    Create one Report (header) per Lab ID & client, and many ReportResult
    child rows (one per BPS/PFAS analyte).
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # single columns (by “contains” tokens)
    idx_lab         = _find_token_col(cols, "sample", "id")
    idx_client      = _find_token_col(cols, "client")
    idx_phone       = _find_token_col(cols, "phone")
    idx_email       = _find_token_col(cols, "email")
    idx_project     = _find_token_col(cols, "project", "lead")
    idx_address     = _find_token_col(cols, "address")

    idx_reported    = _find_token_col(cols, "reported")
    idx_received    = _find_token_col(cols, "received", "date")
    idx_sample_name = _find_token_col(cols, "sample", "name")
    idx_prepared_by = _find_token_col(cols, "prepared", "by")
    idx_matrix      = _find_token_col(cols, "matrix")
    idx_prepared_dt = _find_token_col(cols, "prepared", "date")
    idx_qualifiers  = _find_token_col(cols, "qualifiers")
    idx_asin        = _find_token_col(cols, "asin") or _find_token_col(cols, "identifier")
    idx_weight      = _find_token_col(cols, "product", "weight") or _find_token_col(cols, "weight")
    idx_acq         = _find_token_col(cols, "acq", "date")
    idx_sheet       = _find_token_col(cols, "sheetname") or _find_token_col(cols, "sheet", "name")

    # blocks (by exact sequence)
    sr_seq  = ["analyte","result","mrl","units","dilution","analyzed","qualifier"]
    mb_seq  = ["analyte","result","mrl","units","dilution"]
    ms1_seq = ["analyte","result","mrl","units","dilution","fortified level","%rec","%rec limits"]
    msd_seq = ["analyte","result","units","dilution","%rec","%rec limits","%rpd","%rpd limit"]

    cols_lower = [c.lower() for c in cols]
    sr_start  = _find_sequence(cols_lower, sr_seq)
    mb_start  = _find_sequence(cols_lower, mb_seq)
    ms1_start = _find_sequence(cols_lower, ms1_seq)
    msd_start = _find_sequence(cols_lower, msd_seq)

    created_children = 0
    created_headers  = 0
    updated_headers  = 0
    skipped_non_num  = 0
    skipped_non_tgt  = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = _safe(row.iloc[idx_lab]) if idx_lab is not None else ""
            client = _safe(row.iloc[idx_client]) if idx_client is not None else CLIENT_NAME

            if not _lab_id_is_numericish(lab_id):
                skipped_non_num += 1
                continue

            # Sample Results analyte
            analyte = ""
            sr = {}
            if sr_start is not None:
                try:
                    analyte = _safe(row.iloc[sr_start + 0])
                    sr = {
                        "result":    _safe(row.iloc[sr_start + 1]),
                        "mrl":       _safe(row.iloc[sr_start + 2]),
                        "units":     _safe(row.iloc[sr_start + 3]),
                        "dilution":  _safe(row.iloc[sr_start + 4]),
                        "analyzed":  _safe(row.iloc[sr_start + 5]),
                        "qualifier": _safe(row.iloc[sr_start + 6]),
                    }
                except Exception:
                    analyte = ""
                    sr = {}

            if not _is_target_analyte(analyte):
                skipped_non_tgt += 1
                continue

            # ===== header (one per lab_id+client) =====
            header = db.query(Report).filter(
                Report.lab_id == lab_id, Report.client == client
            ).order_by(Report.id.asc()).first()

            if not header:
                header = Report(lab_id=lab_id, client=client)
                db.add(header)
                db.flush()
                created_headers += 1
            else:
                updated_headers += 1

            # update header metadata (only if present)
            if idx_reported    is not None: header.resulted_date  = parse_date(row.iloc[idx_reported])  or header.resulted_date
            if idx_received    is not None: header.collected_date = parse_date(row.iloc[idx_received])  or header.collected_date
            if idx_phone       is not None: header.phone       = _safe(row.iloc[idx_phone])       or header.phone
            if idx_email       is not None: header.email       = _safe(row.iloc[idx_email])       or header.email
            if idx_project     is not None: header.project_lead= _safe(row.iloc[idx_project])     or header.project_lead
            if idx_address     is not None: header.address     = _safe(row.iloc[idx_address])     or header.address
            if idx_sample_name is not None: header.sample_name = _safe(row.iloc[idx_sample_name]) or header.sample_name
            if idx_prepared_by is not None: header.prepared_by = _safe(row.iloc[idx_prepared_by]) or header.prepared_by
            if idx_matrix      is not None: header.matrix      = _safe(row.iloc[idx_matrix])      or header.matrix
            if idx_prepared_dt is not None: header.prepared_date = _safe(row.iloc[idx_prepared_dt]) or header.prepared_date
            if idx_qualifiers  is not None: header.qualifiers  = _safe(row.iloc[idx_qualifiers])  or header.qualifiers
            if idx_asin        is not None: header.asin        = _safe(row.iloc[idx_asin])        or header.asin
            if idx_weight      is not None: header.product_weight_g = _safe(row.iloc[idx_weight]) or header.product_weight_g
            if idx_acq         is not None: header.acq_datetime= _safe(row.iloc[idx_acq])         or header.acq_datetime
            if idx_sheet       is not None: header.sheet_name  = _safe(row.iloc[idx_sheet])       or header.sheet_name

            # ===== child result (one per analyte) =====
            child = db.query(ReportResult).filter(
                ReportResult.report_id == header.id,
                ReportResult.analyte == analyte
            ).one_or_none()

            if not child:
                child = ReportResult(report_id=header.id, analyte=analyte)
                db.add(child)

            child.result    = sr.get("result")    or child.result
            child.mrl       = sr.get("mrl")       or child.mrl
            child.units     = sr.get("units")     or child.units
            child.dilution  = sr.get("dilution")  or child.dilution
            child.analyzed  = sr.get("analyzed")  or child.analyzed
            child.qualifier = sr.get("qualifier") or child.qualifier

            # Method Blank
            if mb_start is not None:
                try:
                    child.mb_analyte  = _safe(row.iloc[mb_start + 0]) or child.mb_analyte
                    child.mb_result   = _safe(row.iloc[mb_start + 1]) or child.mb_result
                    child.mb_mrl      = _safe(row.iloc[mb_start + 2]) or child.mb_mrl
                    child.mb_units    = _safe(row.iloc[mb_start + 3]) or child.mb_units
                    child.mb_dilution = _safe(row.iloc[mb_start + 4]) or child.mb_dilution
                except Exception:
                    pass

            # Matrix Spike 1
            if ms1_start is not None:
                try:
                    child.ms1_analyte         = _safe(row.iloc[ms1_start + 0]) or child.ms1_analyte
                    child.ms1_result          = _safe(row.iloc[ms1_start + 1]) or child.ms1_result
                    child.ms1_mrl             = _safe(row.iloc[ms1_start + 2]) or child.ms1_mrl
                    child.ms1_units           = _safe(row.iloc[ms1_start + 3]) or child.ms1_units
                    child.ms1_dilution        = _safe(row.iloc[ms1_start + 4]) or child.ms1_dilution
                    child.ms1_fortified_level = _safe(row.iloc[ms1_start + 5]) or child.ms1_fortified_level
                    child.ms1_pct_rec         = _safe(row.iloc[ms1_start + 6]) or child.ms1_pct_rec
                    child.ms1_pct_rec_limits  = _safe(row.iloc[ms1_start + 7]) or child.ms1_pct_rec_limits
                except Exception:
                    pass

            # Matrix Spike Duplicate
            if msd_start is not None:
                try:
                    child.msd_analyte        = _safe(row.iloc[msd_start + 0]) or child.msd_analyte
                    child.msd_result         = _safe(row.iloc[msd_start + 1]) or child.msd_result
                    child.msd_units          = _safe(row.iloc[msd_start + 2]) or child.msd_units
                    child.msd_dilution       = _safe(row.iloc[msd_start + 3]) or child.msd_dilution
                    child.msd_pct_rec        = _safe(row.iloc[msd_start + 4]) or child.msd_pct_rec
                    child.msd_pct_rec_limits = _safe(row.iloc[msd_start + 5]) or child.msd_pct_rec_limits
                    child.msd_pct_rpd        = _safe(row.iloc[msd_start + 6]) or child.msd_pct_rpd
                    child.msd_pct_rpd_limit  = _safe(row.iloc[msd_start + 7]) or child.msd_pct_rpd_limit
                except Exception:
                    pass

            created_children += 1

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (f"Imported/updated {created_children} analyte row(s) across "
            f"{created_headers + updated_headers} report(s). "
            f"Skipped {skipped_non_num} non-numeric Lab ID row(s) and "
            f"{skipped_non_tgt} non-target analyte row(s).")

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
    # export headers only (one per Lab ID)
    rows = db.execute(sql_text("""
        SELECT * FROM reports
        WHERE id IN (SELECT MIN(id) FROM reports GROUP BY lab_id, client)
    """))
    out = []
    for r in rows:
        m = r._mapping
        out.append({
            "Lab ID": m["lab_id"],
            "Client": m["client"],
            "Analyte (lead)": m["test"] or "",
            "Result (lead)": m["result"] or "",
            "Reported": m["resulted_date"] or "",
            "Received": m["collected_date"] or "",
            "PDF URL": m["pdf_url"] or "",
        })
    db.close()

    df = pd.DataFrame(out)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(out)} headers")
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
