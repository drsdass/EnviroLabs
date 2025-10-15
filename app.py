import os
import io
import re
from datetime import datetime, date
from typing import List, Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Text, ForeignKey
)
from sqlalchemy.orm import (
    sessionmaker, declarative_base, relationship
)
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

    # kept for compatibility (not used for PFAS multi-analyte)
    patient_name = Column(String, nullable=True)
    test = Column(String, nullable=True)
    result = Column(String, nullable=True)

    collected_date = Column(Date, nullable=True)   # "Received Date"
    resulted_date = Column(Date, nullable=True)    # "Reported Date"
    pdf_url = Column(String, nullable=True)

    # Optional metadata
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

    # legacy single-analyte extras (kept; not required now)
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)

    # Misc
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # NEW: one-to-many analytes
    analytes = relationship("ReportAnalyte", back_populates="report", cascade="all, delete-orphan")

class ReportAnalyte(Base):
    """
    One row per analyte for a given Report (Lab ID).
    Holds Sample Results + QC (MB, MS1, MSD) for that analyte.
    """
    __tablename__ = "report_analytes"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"), index=True, nullable=False)

    # canonical key for matching (“pfoa”, “pfos”, …, “bisphenol s”)
    analyte_key = Column(String, index=True, nullable=False)
    # pretty display name (“PFOA”, “PFOS”, …, “Bisphenol S”)
    display_name = Column(String, nullable=False)

    # Sample Results
    sample_result = Column(String, nullable=True)
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)

    # Method Blank (MB)
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)

    # Matrix Spike 1 (MS1)
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)

    # Matrix Spike Duplicate (MSD)
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)

    report = relationship("Report", back_populates="analytes")

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'admin' or 'client'
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

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

# ---------- analyte normalization / allow lists ----------
def _norm(s: str) -> str:
    """lowercase, replace non-alnum with space, collapse whitespace"""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

# Map of allowed keys -> required token sets for a match
_PFAS_TOKEN_MAP = {
    "pfoa": {"pfoa"},
    "pfos": {"pfos"},
    "pfna": {"pfna"},
    "fosaa": {"fosaa"},
    "n mefosaa": {"n", "mefosaa"},
    "n etfosaa": {"n", "etfosaa"},
    "sampap": {"sampap"},
    "pfosa": {"pfosa"},
    "n mefosa": {"n", "mefosa"},
    "n mefose": {"n", "mefose"},
    "n etfosa": {"n", "etfosa"},
    "n etfose": {"n", "etfose"},
    "disampap": {"disampap"},
}
# BPS aliases
_BPS_TOKEN_SETS = [ {"bisphenol", "s"}, {"bps"} ]

_isotope_prefix = re.compile(r"^\s*(\d+[A-Za-z]*|-?[dD]\d+)\s*-?\s*")

def _tokenize(s: str) -> List[str]:
    return _norm(s).split()

def _strip_isotope_prefix(name: str) -> str:
    return _isotope_prefix.sub("", str(name or "").strip())

def _normalize_analyte(raw: str) -> tuple[str, str]:
    """
    Normalize an analyte label to (analyte_key, display_name).
    - Strips isotope/surrogate prefixes (e.g., "13C4-", "D8-").
    - Accepts extras like "(C8)" or trailing descriptors.
    - Matches BPS and the EXACT 13 PFAS analytes by token-set containment.
    """
    s = str(raw or "").strip()
    if not s:
        return "", ""
    base = _strip_isotope_prefix(s)
    tokens = set(_tokenize(base))
    if not tokens:
        return "", ""

    # BPS?
    for ts in _BPS_TOKEN_SETS:
        if ts.issubset(tokens):
            return "bisphenol s", "Bisphenol S"

    # PFAS exact list by tokens subset
    for key, needed in _PFAS_TOKEN_MAP.items():
        if needed.issubset(tokens):
            # display name = original (without isotope) up to first " (" if present
            disp = base.split(" (", 1)[0].strip()
            return key, disp or key.upper()

    return "", ""

def _is_supported_analyte(raw: str) -> bool:
    akey, _ = _normalize_analyte(raw)
    return akey != ""

# ---------- header location helpers ----------
def _find_token_col(cols: List[str], *needles: str) -> Optional[int]:
    tokens = [t.lower() for t in needles]
    for i, c in enumerate(cols):
        name = _norm(c)
        if all(tok in name for tok in tokens):
            return i
    return None

def _find_sequence(cols: List[str], seq: List[str]) -> Optional[int]:
    """
    Find starting index of a consecutive sequence of captions (case-insensitive),
    used for blocks like Sample Results / MB / MS1 / MSD.
    """
    n = len(cols)
    m = len(seq)
    seq_l = [s.lower() for s in seq]
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            if str(cols[i + j]).strip().lower() != seq_l[j]:
                ok = False
                break
        if ok:
            return i
    return None

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

    ras = (
        db.query(ReportAnalyte)
        .filter(ReportAnalyte.report_id == report_id)
        .order_by(ReportAnalyte.display_name.asc())
        .all()
    )
    db.close()

    def val(x): return "" if x is None else str(x)
    first = ras[0] if ras else None

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
            "prepared_by": val(r.prepared_by), "matrix": val(r.matrix),
            "prepared_date": val(r.prepared_date), "qualifiers": val(r.qualifiers),
            "asin": val(r.asin), "product_weight_g": val(r.product_weight_g),
        },
        # legacy single row (template-safe)
        "sample_results": {
            "analyte": val(first.display_name) if first else val(r.test),
            "result": val(first.sample_result) if first else val(r.result),
            "mrl": val(first.sample_mrl) if first else val(r.sample_mrl),
            "units": val(first.sample_units) if first else val(r.sample_units),
            "dilution": val(first.sample_dilution) if first else val(r.sample_dilution),
            "analyzed": val(first.sample_analyzed) if first else val(r.sample_analyzed),
            "qualifier": val(first.sample_qualifier) if first else val(r.sample_qualifier),
        },
        "acq_datetime": val(r.acq_datetime),
        "sheet_name": val(r.sheet_name),
    }

    analytes_payload = []
    for a in ras:
        analytes_payload.append({
            "display_name": a.display_name,
            "sample": {
                "result": a.sample_result, "mrl": a.sample_mrl, "units": a.sample_units,
                "dilution": a.sample_dilution, "analyzed": a.sample_analyzed, "qualifier": a.sample_qualifier
            },
            "mb": {
                "result": a.mb_result, "mrl": a.mb_mrl, "units": a.mb_units, "dilution": a.mb_dilution
            },
            "ms1": {
                "result": a.ms1_result, "mrl": a.ms1_mrl, "units": a.ms1_units, "dilution": a.ms1_dilution,
                "fortified_level": a.ms1_fortified_level, "pct_rec": a.ms1_pct_rec, "pct_rec_limits": a.ms1_pct_rec_limits
            },
            "msd": {
                "result": a.msd_result, "units": a.msd_units, "dilution": a.msd_dilution,
                "pct_rec": a.msd_pct_rec, "pct_rec_limits": a.msd_pct_rec_limits,
                "pct_rpd": a.msd_pct_rpd, "pct_rpd_limit": a.msd_pct_rpd_limit
            }
        })

    # Your current template works; if you want a table of all analytes,
    # use 'analytes' in the template to render them.
    return render_template("report_detail.html", user=u, r=r, p=p, analytes=analytes_payload)

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

    # Try Master Upload (banner + header row) first
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

    # detect header row (contains "Sample ID")
    header_row_idx = None
    for i in range(min(10, len(raw))):
        row_vals = [str(x) for x in list(raw.iloc[i].values)]
        if any("sample id" in _norm(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        # fallback to old simple importer
        df = _fallback_simple_table(saved_path)
        if isinstance(df, str):
            flash(df, "error")
            if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
                os.remove(saved_path)
            return redirect(url_for("dashboard"))
        msg = _ingest_simple(df, u, filename)
        flash(msg, "success")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # build DataFrame with that header
    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    # drop fully empty rows
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]

    msg = _ingest_master_upload(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _fallback_simple_table(path) -> pd.DataFrame | str:
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

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.iloc[c_lab]).strip()
            client = str(row.iloc[c_client]).strip() or CLIENT_NAME
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue
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
            f"Skipped {skipped_num} non-numeric Lab ID row(s).")

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Master Upload File; create one Report per Lab ID (numeric-leading),
    and one ReportAnalyte per supported analyte row (BPS + EXACT 13 PFAS).
    If any QC values are missing, we still import the analyte and leave them blank.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # Single columns we care about
    idx_lab = _find_token_col(cols, "sample", "id")
    idx_client = _find_token_col(cols, "client")
    idx_phone = _find_token_col(cols, "phone")
    idx_email = _find_token_col(cols, "email")
    idx_lead = _find_token_col(cols, "project", "lead")
    idx_addr = _find_token_col(cols, "address")

    idx_reported = _find_token_col(cols, "reported")
    idx_received = _find_token_col(cols, "received", "date")
    idx_sample_name = _find_token_col(cols, "sample", "name")
    idx_prepared_by = _find_token_col(cols, "prepared", "by")
    idx_matrix = _find_token_col(cols, "matrix")
    idx_prepared_date = _find_token_col(cols, "prepared", "date")
    idx_qualifiers = _find_token_col(cols, "qualifiers")
    idx_asin = _find_token_col(cols, "asin") or _find_token_col(cols, "identifier")
    idx_weight = _find_token_col(cols, "product", "weight") or _find_token_col(cols, "weight")

    idx_acq = _find_token_col(cols, "acq", "date")
    idx_sheet = _find_token_col(cols, "sheetname") or _find_token_col(cols, "sheet", "name")

    # Blocks (exact consecutive captions)
    sr_seq  = ["analyte", "result", "mrl", "units", "dilution", "analyzed", "qualifier"]
    mb_seq  = ["analyte", "result", "mrl", "units", "dilution"]
    ms1_seq = ["analyte", "result", "mrl", "units", "dilution", "fortified level", "%rec", "%rec limits"]
    msd_seq = ["analyte", "result", "units", "dilution", "%rec", "%rec limits", "%rpd", "%rpd limit"]

    cols_lower = [str(c).lower().strip() for c in cols]
    sr_start  = _find_sequence(cols_lower, sr_seq)
    mb_start  = _find_sequence(cols_lower, mb_seq)
    ms1_start = _find_sequence(cols_lower, ms1_seq)
    msd_start = _find_sequence(cols_lower, msd_seq)

    created_reports = 0
    updated_reports = 0
    created_analytes = 0
    updated_analytes = 0
    skipped_num = 0
    skipped_analyte = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            def get(idx):
                return "" if idx is None else str(row.iloc[idx]).strip()

            lab_id  = get(idx_lab)
            client  = get(idx_client) or CLIENT_NAME

            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            # Upsert report per Lab ID
            rpt = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if rpt is None:
                rpt = Report(lab_id=lab_id, client=client)
                db.add(rpt)
                created_reports += 1
            else:
                rpt.client = client
                updated_reports += 1

            # Client info / summary
            rpt.phone        = get(idx_phone)
            rpt.email        = get(idx_email)
            rpt.project_lead = get(idx_lead)
            rpt.address      = get(idx_addr)

            rpt.resulted_date  = parse_date(get(idx_reported)) if idx_reported is not None else None
            rpt.collected_date = parse_date(get(idx_received)) if idx_received is not None else None

            rpt.sample_name     = get(idx_sample_name) or lab_id
            rpt.prepared_by     = get(idx_prepared_by)
            rpt.matrix          = get(idx_matrix)
            rpt.prepared_date   = get(idx_prepared_date)
            rpt.qualifiers      = get(idx_qualifiers)
            rpt.asin            = get(idx_asin)
            rpt.product_weight_g= get(idx_weight)

            rpt.acq_datetime    = get(idx_acq)
            rpt.sheet_name      = get(idx_sheet)

            # Must have SR block to read analyte row
            if sr_start is None:
                continue

            sr_analyte_raw = str(row.iloc[sr_start + 0]).strip()
            if not _is_supported_analyte(sr_analyte_raw):
                skipped_analyte += 1
                continue

            akey, display_name = _normalize_analyte(sr_analyte_raw)

            # Upsert analyte row for this report+akey
            ra = (
                db.query(ReportAnalyte)
                .filter(ReportAnalyte.report_id == rpt.id, ReportAnalyte.analyte_key == akey)
                .one_or_none()
            )
            if ra is None:
                ra = ReportAnalyte(report=rpt, analyte_key=akey, display_name=display_name)
                db.add(ra)
                created_analytes += 1
            else:
                ra.display_name = display_name
                updated_analytes += 1

            # Sample Results (always set; blanks ok)
            ra.sample_result    = str(row.iloc[sr_start + 1]).strip()
            ra.sample_mrl       = str(row.iloc[sr_start + 2]).strip()
            ra.sample_units     = str(row.iloc[sr_start + 3]).strip()
            ra.sample_dilution  = str(row.iloc[sr_start + 4]).strip()
            ra.sample_analyzed  = str(row.iloc[sr_start + 5]).strip()
            ra.sample_qualifier = str(row.iloc[sr_start + 6]).strip()

            # MB (if present in header; allow blanks)
            if mb_start is not None:
                ra.mb_result   = str(row.iloc[mb_start + 1]).strip()
                ra.mb_mrl      = str(row.iloc[mb_start + 2]).strip()
                ra.mb_units    = str(row.iloc[mb_start + 3]).strip()
                ra.mb_dilution = str(row.iloc[mb_start + 4]).strip()

            # MS1
            if ms1_start is not None:
                ra.ms1_result          = str(row.iloc[ms1_start + 1]).strip()
                ra.ms1_mrl             = str(row.iloc[ms1_start + 2]).strip()
                ra.ms1_units           = str(row.iloc[ms1_start + 3]).strip()
                ra.ms1_dilution        = str(row.iloc[ms1_start + 4]).strip()
                ra.ms1_fortified_level = str(row.iloc[ms1_start + 5]).strip()
                ra.ms1_pct_rec         = str(row.iloc[ms1_start + 6]).strip()
                ra.ms1_pct_rec_limits  = str(row.iloc[ms1_start + 7]).strip()

            # MSD
            if msd_start is not None:
                ra.msd_result         = str(row.iloc[msd_start + 1]).strip()
                ra.msd_units          = str(row.iloc[msd_start + 2]).strip()
                ra.msd_dilution       = str(row.iloc[msd_start + 3]).strip()
                ra.msd_pct_rec        = str(row.iloc[msd_start + 4]).strip()
                ra.msd_pct_rec_limits = str(row.iloc[msd_start + 5]).strip()
                ra.msd_pct_rpd        = str(row.iloc[msd_start + 6]).strip()
                ra.msd_pct_rpd_limit  = str(row.iloc[msd_start + 7]).strip()

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (
        f"Reports (created={created_reports}, updated={updated_reports}). "
        f"Analytes (created={created_analytes}, updated={updated_analytes}). "
        f"Skipped {skipped_num} non-numeric Lab ID row(s) and {skipped_analyte} non-supported analyte row(s)."
    )

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
        "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
        "Received": r.collected_date.isoformat() if r.collected_date else "",
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
