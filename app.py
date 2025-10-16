import os
import io
from typing import Optional, List, Dict
from datetime import datetime, date

from flask import (
    Flask, render_template, render_template_string, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename

from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text as sql_text

import pandas as pd

# =========================
# Config
# =========================
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

# =========================
# App
# =========================
app = Flask(__name__)
app.secret_key = SECRET_KEY

# =========================
# DB
# =========================
DB_PATH = os.path.join(BASE_DIR, "app.db")
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)

    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)

    # for compatibility
    patient_name = Column(String, nullable=True)

    # primary fields
    test = Column(String, nullable=True)    # analyte name (e.g., "Bisphenol S", "PFOA", ...)
    result = Column(String, nullable=True)

    collected_date = Column(Date, nullable=True)  # "Received"
    resulted_date = Column(Date, nullable=True)   # "Reported"
    pdf_url = Column(String, nullable=True)

    # Client/summary metadata
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

    # Method Blank
    mb_analyte = Column(String, nullable=True)
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)

    # Matrix Spike 1
    ms1_analyte = Column(String, nullable=True)
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)

    # Matrix Spike Duplicate
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

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# add columns if missing (for older dbs)
def _ensure_report_columns():
    needed = {
        "phone","email","project_lead","address","sample_name","prepared_by","matrix",
        "prepared_date","qualifiers","asin","product_weight_g",
        "sample_mrl","sample_units","sample_dilution","sample_analyzed","sample_qualifier",
        "mb_analyte","mb_result","mb_mrl","mb_units","mb_dilution",
        "ms1_analyte","ms1_result","ms1_mrl","ms1_units","ms1_dilution","ms1_fortified_level",
        "ms1_pct_rec","ms1_pct_rec_limits",
        "msd_analyte","msd_result","msd_units","msd_dilution","msd_pct_rec",
        "msd_pct_rec_limits","msd_pct_rpd","msd_pct_rpd_limit",
        "acq_datetime","sheet_name",
    }
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(reports)"))}
        for col in sorted(needed - existing):
            conn.execute(sql_text(f"ALTER TABLE reports ADD COLUMN {col} TEXT"))
_ensure_report_columns()

# =========================
# Constants / helpers
# =========================

PFAS_DISPLAY = [
    "PFOA","PFOS","PFNA","FOSAA","N-MeFOSAA","N-EtFOSAA","SAmPAP",
    "PFOSA","N-MeFOSA","N-MeFOSE","N-EtFOSA","N-EtFOSE","diSAmPAP"
]
def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

PFAS_KEYS = [ _norm(x) for x in PFAS_DISPLAY ]
PFAS_MAP: Dict[str,str] = { _norm(x): x for x in PFAS_DISPLAY }

def _pfas_key(name: str) -> Optional[str]:
    if not name: return None
    k = _norm(name)
    return k if k in PFAS_MAP else None

def _is_internal_standard(name: str) -> bool:
    """Filter out internal standards (do not render or store as analytes)."""
    if not name: return False
    n = _norm(name)
    return ("13c12" in n and "bps" in n) or ("d8" in n and "bps" in n)

def _is_bps(name: str) -> bool:
    if not name: return False
    n = _norm(name)
    return "bisphenol" in n and "s" in n and not _is_internal_standard(name)

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

def parse_date(val):
    if val is None: return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan","none"}:
        return None
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%m/%d/%y","%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        ts = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except Exception:
        return None

def current_user():
    return {
        "username": session.get("username"),
        "role": session.get("role"),
        "client_name": session.get("client_name"),
    }

def require_login(role=None):
    def deco(fn):
        def wrapper(*a, **k):
            if "username" not in session:
                return redirect(url_for("home"))
            if role and session.get("role") != role:
                flash("Unauthorized", "error")
                return redirect(url_for("dashboard"))
            return fn(*a, **k)
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco

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

def find_header_row(raw: pd.DataFrame) -> Optional[int]:
    """Find the line that contains the actual headers (banner rows will be above)."""
    limit = min(20, len(raw))
    for i in range(limit):
        vals = [str(x) for x in raw.iloc[i].values]
        line = " ".join(_norm(v) for v in vals)
        if "sample" in line and "id" in line:
            return i
    return None

def _first_nonempty(*vals) -> str:
    for v in vals:
        if v is None: continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none"}:
            return s
    return ""

# =========================
# Routes: auth + base
# =========================
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    role = request.form.get("role")
    username = request.form.get("username","").strip()
    password = request.form.get("password","").strip()
    if role == "admin" and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"] = username
        session["role"] = "admin"
        session["client_name"] = None
        log_action(username, "admin", "login", "Admin logged in")
        return redirect(url_for("dashboard"))
    if role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"] = username
        session["role"] = "client"
        session["client_name"] = CLIENT_NAME
        log_action(username, "client", "login", f"Client '{CLIENT_NAME}' logged in")
        return redirect(url_for("dashboard"))
    flash("Invalid credentials", "error")
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    u = current_user()
    if u["username"]:
        log_action(u["username"], u["role"] or "unknown", "logout", "User logged out")
    session.clear()
    return redirect(url_for("home"))

# =========================
# Dashboard
# =========================
@app.route("/dashboard")
def dashboard():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    lab_id = request.args.get("lab_id","").strip()
    start  = request.args.get("start","").strip()
    end    = request.args.get("end","").strip()

    db = SessionLocal()
    q = db.query(Report)
    if u["role"] == "client":
        q = q.filter(Report.client == u["client_name"])

    if lab_id:
        q = q.filter(Report.lab_id == lab_id)
    if start:
        sd = parse_date(start)
        if sd: q = q.filter(Report.resulted_date >= sd)
    if end:
        ed = parse_date(end)
        if ed: q = q.filter(Report.resulted_date <= ed)

    try:
        reports = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).limit(1000).all()
    except Exception:
        reports = q.order_by(Report.resulted_date.desc(), Report.id.desc()).limit(1000).all()

    db.close()
    return render_template("dashboard.html", user=u, reports=reports)

# =========================
# Report detail (group PFAS)
# =========================
@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    base = db.query(Report).get(report_id)
    if not base:
        db.close()
        flash("Report not found", "error")
        return redirect(url_for("dashboard"))
    if u["role"] == "client" and base.client != u["client_name"]:
        db.close()
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    rows = db.query(Report).filter(Report.lab_id == base.lab_id).all()
    db.close()

    has_bps = any(_is_bps(r.test) for r in rows)

    if has_bps:
        mode = "BPS"
        # Pick a single BPS row
        bps_rows = [r for r in rows if _is_bps(r.test)]
        bps_rows.sort(key=lambda r: (str(r.result or "") == "", r.id))
        r = bps_rows[0]

        sample_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(r.result),
            "mrl": _first_nonempty(r.sample_mrl),
            "units": _first_nonempty(r.sample_units),
            "dilution": _first_nonempty(r.sample_dilution),
            "analyzed": _first_nonempty(r.sample_analyzed),
            "qualifier": _first_nonempty(r.sample_qualifier),
        }]
        mb_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(r.mb_result),
            "mrl": _first_nonempty(r.mb_mrl),
            "units": _first_nonempty(r.mb_units),
            "dilution": _first_nonempty(r.mb_dilution),
        }]
        ms1_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(r.ms1_result),
            "mrl": _first_nonempty(r.ms1_mrl),
            "units": _first_nonempty(r.ms1_units),
            "dilution": _first_nonempty(r.ms1_dilution),
            "fortified_level": _first_nonempty(r.ms1_fortified_level),
            "pct_rec": _first_nonempty(r.ms1_pct_rec),
            "pct_rec_limits": _first_nonempty(r.ms1_pct_rec_limits),
        }]
        msd_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(r.msd_result),
            "units": _first_nonempty(r.msd_units),
            "dilution": _first_nonempty(r.msd_dilution),
            "pct_rec": _first_nonempty(r.msd_pct_rec),
            "pct_rec_limits": _first_nonempty(r.msd_pct_rec_limits),
            "pct_rpd": _first_nonempty(r.msd_pct_rpd),
            "pct_rpd_limit": _first_nonempty(r.msd_pct_rpd_limit),
        }]

        # ensure summary info
        base.sample_name = _first_nonempty(base.sample_name, base.lab_id)
    else:
        mode = "PFAS"
        # group rows by analyte key, choose best row per analyte
        groups: Dict[str, List[Report]] = {}
        for r in rows:
            if _is_internal_standard(r.test):  # ignore IS
                continue
            k = _pfas_key(r.test)
            if not k:
                continue
            groups.setdefault(k, []).append(r)

        chosen: Dict[str, Report] = {}
        for k, items in groups.items():
            items.sort(key=lambda rr: (str(rr.result or "") == "", rr.id))
            chosen[k] = items[0]

        sample_rows, mb_rows, ms1_rows, msd_rows = [], [], [], []
        for k in PFAS_KEYS:
            r = chosen.get(k)
            if not r:
                # skip analytes not present in the file for this sample
                continue
            name = PFAS_MAP[k]
            sample_rows.append({
                "analyte": name,
                "result": _first_nonempty(r.result),
                "mrl": _first_nonempty(r.sample_mrl),
                "units": _first_nonempty(r.sample_units),
                "dilution": _first_nonempty(r.sample_dilution),
                "analyzed": _first_nonempty(r.sample_analyzed),
                "qualifier": _first_nonempty(r.sample_qualifier),
            })
            mb_rows.append({
                "analyte": name,
                "result": _first_nonempty(r.mb_result),
                "mrl": _first_nonempty(r.mb_mrl),
                "units": _first_nonempty(r.mb_units),
                "dilution": _first_nonempty(r.mb_dilution),
            })
            ms1_rows.append({
                "analyte": name,
                "result": _first_nonempty(r.ms1_result),
                "mrl": _first_nonempty(r.ms1_mrl),
                "units": _first_nonempty(r.ms1_units),
                "dilution": _first_nonempty(r.ms1_dilution),
                "fortified_level": _first_nonempty(r.ms1_fortified_level),
                "pct_rec": _first_nonempty(r.ms1_pct_rec),
                "pct_rec_limits": _first_nonempty(r.ms1_pct_rec_limits),
            })
            msd_rows.append({
                "analyte": name,
                "result": _first_nonempty(r.msd_result),
                "units": _first_nonempty(r.msd_units),
                "dilution": _first_nonempty(r.msd_dilution),
                "pct_rec": _first_nonempty(r.msd_pct_rec),
                "pct_rec_limits": _first_nonempty(r.msd_pct_rec_limits),
                "pct_rpd": _first_nonempty(r.msd_pct_rpd),
                "pct_rpd_limit": _first_nonempty(r.msd_pct_rpd_limit),
            })

        # borrow summary fields
        pick = (list(chosen.values()) or rows)[0]
        base.sample_name      = _first_nonempty(base.sample_name, pick.sample_name, base.lab_id)
        base.prepared_by      = _first_nonempty(base.prepared_by, pick.prepared_by)
        base.matrix           = _first_nonempty(base.matrix, pick.matrix)
        base.prepared_date    = _first_nonempty(base.prepared_date, pick.prepared_date)
        base.asin             = _first_nonempty(base.asin, pick.asin)
        base.product_weight_g = _first_nonempty(base.product_weight_g, pick.product_weight_g)
        base.qualifiers       = _first_nonempty(base.qualifiers, pick.qualifiers)
        base.phone            = _first_nonempty(base.phone, pick.phone)
        base.email            = _first_nonempty(base.email, pick.email)
        base.project_lead     = _first_nonempty(base.project_lead, pick.project_lead)
        base.address          = _first_nonempty(base.address, pick.address)

    return render_template(
        "report_detail.html",
        user=u, r=base,
        sample_rows=sample_rows, mb_rows=mb_rows, ms1_rows=ms1_rows, msd_rows=msd_rows,
        mode=mode
    )

# =========================
# Upload (robust)
# =========================
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

    raw = None
    err = None
    for reader in (
        lambda: pd.read_csv(saved_path, header=None, dtype=str),
        lambda: pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl"),
        lambda: pd.read_csv(saved_path, header=0, dtype=str),
        lambda: pd.read_excel(saved_path, header=0, dtype=str, engine="openpyxl"),
    ):
        try:
            raw = reader().fillna("")
            break
        except Exception as e:
            err = e

    if raw is None:
        flash(f"Could not read file: {err}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    header_row = find_header_row(raw)
    if header_row is None and list(raw.columns) and isinstance(raw.columns[0], str):
        header_row = 0
    if header_row is None:
        flash("Could not locate header row with 'Sample ID'.", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    headers = [str(x).strip() for x in raw.iloc[header_row].values]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")

    msg = _ingest_master_upload(df, u, filename, debug=True)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)
    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, u, filename: str, debug: bool=False) -> str:
    cols = list(df.columns)
    def c(*tokens):
        tl = [t.lower() for t in tokens]
        for name in cols:
            n = _norm(name)
            if all(t in n for t in tl):
                return name
        return None

    # single columns
    col_lab   = c("sample","id")
    col_client= c("client")
    col_phone = c("phone")
    col_email = c("email")
    col_pjlead= c("project","lead")
    col_addr  = c("address")
    col_rep   = c("reported")
    col_recv  = c("received","date")
    col_sname = c("sample","name")
    col_prepby= c("prepared","by")
    col_matrix= c("matrix")
    col_prepdt= c("prepared","date")
    col_qual  = c("qualifiers")
    col_asin  = c("asin") or c("identifier")
    col_weight= c("product","weight")
    col_acq   = c("acq","date")
    col_sheet = c("sheetname") or c("sheet","name")

    norm_cols = [_norm(x) for x in cols]
    analyte_idx = [i for i,n in enumerate(norm_cols) if n == "analyte"]

    def looks_like(start: int, wanted: List[str]) -> bool:
        slice_norm = norm_cols[start:start+len(wanted)]
        return len(slice_norm) == len(wanted) and all(a == b for a,b in zip(slice_norm, wanted))

    def looks_sr(i): return looks_like(i, ["analyte","result","mrl","units","dilution","analyzed","qualifier"])
    def looks_mb(i): return looks_like(i, ["analyte","result","mrl","units","dilution"])
    def looks_ms1(i):return looks_like(i, ["analyte","result","mrl","units","dilution","fortified level","%rec","%rec limits"])
    def looks_msd(i):return looks_like(i, ["analyte","result","units","dilution","%rec","%rec limits","%rpd","%rpd limit"])

    sr_start = mb_start = ms1_start = msd_start = None
    for idx in analyte_idx:
        if sr_start is None and looks_sr(idx): sr_start = idx; continue
        if sr_start is not None and mb_start is None and looks_mb(idx): mb_start = idx; continue
        if mb_start is not None and ms1_start is None and looks_ms1(idx): ms1_start = idx; continue
        if ms1_start is not None and msd_start is None and looks_msd(idx): msd_start = idx; continue

    if sr_start is None and analyte_idx:
        sr_start = analyte_idx[0]
    if mb_start is None and sr_start is not None:
        mb_start = sr_start + 7
    if ms1_start is None and mb_start is not None:
        ms1_start = mb_start + 5
    if msd_start is None and ms1_start is not None:
        msd_start = ms1_start + 8

    def block(start, length):
        return [cols[start + j] if start is not None and start + j < len(cols) else None for j in range(length)]

    SR  = block(sr_start, 7)
    MB  = block(mb_start, 5)
    MS1 = block(ms1_start, 8)
    MSD = block(msd_start, 8)

    if debug:
        flash(f"Header OK. SR@{sr_start} MB@{mb_start} MS1@{ms1_start} MSD@{msd_start}; rows: {len(df)}", "info")

    created = updated = skipped_num = 0
    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.get(col_lab, "")).strip()
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            client = str(row.get(col_client, CLIENT_NAME)).strip() or CLIENT_NAME

            sr_analyte = str(row.get(SR[0], "")).strip() if SR[0] else ""
            if _is_internal_standard(sr_analyte):
                continue

            is_bps = _is_bps(sr_analyte)
            k = _pfas_key(sr_analyte)
            if not is_bps and not k:
                # ignore unrelated analytes
                continue

            existing = db.query(Report).filter(
                Report.lab_id == lab_id,
                Report.test == (sr_analyte or "")
            ).one_or_none()

            if not existing:
                existing = Report(lab_id=lab_id, client=client, test=sr_analyte)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

            # meta
            existing.sample_name      = str(row.get(col_sname, lab_id)).strip()
            existing.phone            = str(row.get(col_phone, "")).strip()
            existing.email            = str(row.get(col_email, "")).strip()
            existing.project_lead     = str(row.get(col_pjlead, "")).strip()
            existing.address          = str(row.get(col_addr, "")).strip()
            existing.resulted_date    = parse_date(row.get(col_rep))
            existing.collected_date   = parse_date(row.get(col_recv))
            existing.prepared_by      = str(row.get(col_prepby, "")).strip()
            existing.matrix           = str(row.get(col_matrix, "")).strip()
            existing.prepared_date    = str(row.get(col_prepdt, "")).strip()
            existing.qualifiers       = str(row.get(col_qual, "")).strip()
            existing.asin             = str(row.get(col_asin, "")).strip()
            existing.product_weight_g = str(row.get(col_weight, "")).strip()
            existing.acq_datetime     = str(row.get(col_acq, "")).strip() if col_acq else ""
            existing.sheet_name       = str(row.get(col_sheet, "")).strip() if col_sheet else ""

            # SR
            existing.result            = str(row.get(SR[1], "")).strip() if SR[1] else ""
            existing.sample_mrl        = str(row.get(SR[2], "")).strip() if SR[2] else ""
            existing.sample_units      = str(row.get(SR[3], "")).strip() if SR[3] else ""
            existing.sample_dilution   = str(row.get(SR[4], "")).strip() if SR[4] else ""
            existing.sample_analyzed   = str(row.get(SR[5], "")).strip() if SR[5] else ""
            existing.sample_qualifier  = str(row.get(SR[6], "")).strip() if SR[6] else ""

            # MB
            existing.mb_analyte  = str(row.get(MB[0], "")).strip() if MB[0] else ""
            existing.mb_result   = str(row.get(MB[1], "")).strip() if MB[1] else ""
            existing.mb_mrl      = str(row.get(MB[2], "")).strip() if MB[2] else ""
            existing.mb_units    = str(row.get(MB[3], "")).strip() if MB[3] else ""
            existing.mb_dilution = str(row.get(MB[4], "")).strip() if MB[4] else ""

            # MS1
            existing.ms1_analyte         = str(row.get(MS1[0], "")).strip() if MS1[0] else ""
            existing.ms1_result          = str(row.get(MS1[1], "")).strip() if MS1[1] else ""
            existing.ms1_mrl             = str(row.get(MS1[2], "")).strip() if MS1[2] else ""
            existing.ms1_units           = str(row.get(MS1[3], "")).strip() if MS1[3] else ""
            existing.ms1_dilution        = str(row.get(MS1[4], "")).strip() if MS1[4] else ""
            existing.ms1_fortified_level = str(row.get(MS1[5], "")).strip() if MS1[5] else ""
            existing.ms1_pct_rec         = str(row.get(MS1[6], "")).strip() if MS1[6] else ""
            existing.ms1_pct_rec_limits  = str(row.get(MS1[7], "")).strip() if MS1[7] else ""

            # MSD
            existing.msd_analyte         = str(row.get(MSD[0], "")).strip() if MSD[0] else ""
            existing.msd_result          = str(row.get(MSD[1], "")).strip() if MSD[1] else ""
            existing.msd_units           = str(row.get(MSD[2], "")).strip() if MSD[2] else ""
            existing.msd_dilution        = str(row.get(MSD[3], "")).strip() if MSD[3] else ""
            existing.msd_pct_rec         = str(row.get(MSD[4], "")).strip() if MSD[4] else ""
            existing.msd_pct_rec_limits  = str(row.get(MSD[5], "")).strip() if MSD[5] else ""
            existing.msd_pct_rpd         = str(row.get(MSD[6], "")).strip() if MSD[6] else ""
            existing.msd_pct_rpd_limit   = str(row.get(MSD[7], "")).strip() if MSD[7] else ""

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (f"Imported {created} new and updated {updated} row(s). "
            f"Skipped {skipped_num} non-numeric Lab ID row(s).")

# =========================
# Audit / Export / Health
# =========================
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
