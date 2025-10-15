import os
import io
import json
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

    # Compatibility fields
    patient_name = Column(String, nullable=True)

    # Single-analyte (Bisphenol S) fields
    test = Column(String, nullable=True)      # e.g., "Bisphenol S" or "PFAS Panel"
    result = Column(String, nullable=True)
    collected_date = Column(Date, nullable=True)  # "Received Date"
    resulted_date = Column(Date, nullable=True)   # "Reported Date"
    pdf_url = Column(String, nullable=True)

    # Client info
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    project_lead = Column(String, nullable=True)
    address = Column(String, nullable=True)

    # Sample summary
    sample_name = Column(String, nullable=True)
    prepared_by = Column(String, nullable=True)
    matrix = Column(String, nullable=True)
    prepared_date = Column(String, nullable=True)
    qualifiers = Column(String, nullable=True)
    asin = Column(String, nullable=True)
    product_weight_g = Column(String, nullable=True)

    # Sample results extras (BPS)
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)

    # QC for single-analyte (BPS)
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

    # Misc
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)

    # NEW: PFAS bundle (JSON text: list of analyte dicts)
    pfas_json = Column(Text, nullable=True)  # stores JSON list with 13 PFAS rows + QC per analyte

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

# one-time add columns if DB existed before
def _ensure_report_columns():
    needed = {
        "phone","email","project_lead","address","sample_name","prepared_by","matrix",
        "prepared_date","qualifiers","asin","product_weight_g","sample_mrl","sample_units",
        "sample_dilution","sample_analyzed","sample_qualifier","mb_analyte","mb_result",
        "mb_mrl","mb_units","mb_dilution","ms1_analyte","ms1_result","ms1_mrl","ms1_units",
        "ms1_dilution","ms1_fortified_level","ms1_pct_rec","ms1_pct_rec_limits","msd_analyte",
        "msd_result","msd_units","msd_dilution","msd_pct_rec","msd_pct_rec_limits","msd_pct_rpd",
        "msd_pct_rpd_limit","acq_datetime","sheet_name","pfas_json"
    }
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(reports)"))}
        for col in sorted(needed - cols):
            conn.execute(sql_text(f"ALTER TABLE reports ADD COLUMN {col} TEXT"))

_ensure_report_columns()

# ------------------- Helpers -------------------
PFAS_NAMES = {
    "pfoa","pfos","pfna","fosaa","n-mefosaa","n-etfosaa","sampap",
    "pfosa","n-mefosa","n-mefose","n-etfosa","n-etfose","disampap"
}
def _norm_space(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

def _parse_date(val):
    if val is None: return None
    s = str(val).strip()
    if not s or s.lower() in {"nan","none"}: return None
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%m/%d/%y","%d-%b-%Y"):
        try: return datetime.strptime(s, fmt).date()
        except: pass
    try:
        ts = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except: return None

def _find_row_header_index(raw: pd.DataFrame) -> Optional[int]:
    for i in range(min(10, len(raw))):
        if any("sample id" in _norm_space(x) for x in raw.iloc[i].tolist()):
            return i
    return None

def _find_seq(cols: List[str], seq: List[str]) -> Optional[int]:
    m = [c.strip().lower() for c in cols]
    seq = [s.lower() for s in seq]
    for i in range(0, len(m)-len(seq)+1):
        ok = True
        for j in range(len(seq)):
            if m[i+j] != seq[j]: ok = False; break
        if ok: return i
    return None

def _find_col(cols: List[str], *tokens) -> Optional[int]:
    toks = [t.lower() for t in tokens]
    for i,c in enumerate(cols):
        if all(t in _norm_space(c) for t in toks): return i
    return None

def _pfas_key(analyte: str) -> Optional[str]:
    k = _norm_space(analyte).replace(" ", "")
    return k if k in PFAS_NAMES else None

def _load_pfas_list(text: Optional[str]) -> List[dict]:
    if not text: return []
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else []
    except: return []

def _save_pfas_list(lst: List[dict]) -> str:
    return json.dumps(lst, ensure_ascii=False)

def _merge_pfas(pfas_list: List[dict], row: dict) -> List[dict]:
    """Insert or replace PFAS analyte by name key."""
    key = _pfas_key(row.get("analyte",""))
    if not key: return pfas_list
    out = []
    replaced = False
    for r in pfas_list:
        if _pfas_key(r.get("analyte","")) == key:
            out.append(row); replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(row)
    # keep a stable order by analyte name
    return sorted(out, key=lambda d: d.get("analyte","").lower())

def current_user():
    return {"username": session.get("username"),
            "role": session.get("role"),
            "client_name": session.get("client_name")}

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
    except:
        db.rollback()
    finally:
        db.close()

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
        session.update({"username": username, "role":"admin", "client_name": None})
        log_action(username, "admin", "login", "Admin logged in")
        return redirect(url_for("dashboard"))
    elif role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session.update({"username": username, "role":"client", "client_name": CLIENT_NAME})
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
        sd = _parse_date(start)
        if sd: q = q.filter(Report.resulted_date >= sd)
    if end:
        ed = _parse_date(end)
        if ed: q = q.filter(Report.resulted_date <= ed)

    try:
        reports = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).limit(500).all()
    except:
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

    def v(x): return "" if x is None else str(x)

    p = {
        "client_info": {
            "client": v(r.client), "phone": v(r.phone), "email": v(r.email) or "support@envirolabsusa.com",
            "project_lead": v(r.project_lead), "address": v(r.address)
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": v(r.sample_name or r.lab_id),
            "prepared_by": v(r.prepared_by), "matrix": v(r.matrix),
            "prepared_date": v(r.prepared_date), "qualifiers": v(r.qualifiers),
            "asin": v(r.asin), "product_weight_g": v(r.product_weight_g),
        },
        "sample_results": {  # used for BPS report
            "analyte": v(r.test), "result": v(r.result),
            "mrl": v(r.sample_mrl), "units": v(r.sample_units),
            "dilution": v(r.sample_dilution), "analyzed": v(r.sample_analyzed),
            "qualifier": v(r.sample_qualifier),
        },
        "method_blank": {
            "analyte": v(r.mb_analyte), "result": v(r.mb_result),
            "mrl": v(r.mb_mrl), "units": v(r.mb_units), "dilution": v(r.mb_dilution),
        },
        "matrix_spike_1": {
            "analyte": v(r.ms1_analyte), "result": v(r.ms1_result),
            "mrl": v(r.ms1_mrl), "units": v(r.ms1_units), "dilution": v(r.ms1_dilution),
            "fortified_level": v(r.ms1_fortified_level), "pct_rec": v(r.ms1_pct_rec),
            "pct_rec_limits": v(r.ms1_pct_rec_limits),
        },
        "matrix_spike_dup": {
            "analyte": v(r.msd_analyte), "result": v(r.msd_result),
            "units": v(r.msd_units), "dilution": v(r.msd_dilution),
            "pct_rec": v(r.msd_pct_rec), "pct_rec_limits": v(r.msd_pct_rec_limits),
            "pct_rpd": v(r.msd_pct_rpd), "pct_rpd_limit": v(r.msd_pct_rpd_limit),
        },
        "acq_datetime": v(r.acq_datetime),
        "sheet_name": v(r.sheet_name),
        # NEW: PFAS bundle prepared for template
        "pfas_rows": _load_pfas_list(r.pfas_json),
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

    # Read raw with no header, find real header row
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

    header_row = _find_row_header_index(raw)
    if header_row is None:
        flash("Could not locate the header row (looking for 'Sample ID').", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    headers = [str(x).strip() for x in raw.iloc[header_row].values]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")
    cols = list(df.columns)

    # locate single columns
    idx_lab = _find_col(cols, "sample", "id")
    idx_client = _find_col(cols, "client")
    idx_reported = _find_col(cols, "reported")
    idx_received = _find_col(cols, "received", "date")
    idx_sample_name = _find_col(cols, "sample", "name")
    idx_prepared_by = _find_col(cols, "prepared", "by")
    idx_matrix = _find_col(cols, "matrix")
    idx_prepared_date = _find_col(cols, "prepared", "date")
    idx_qualifiers = _find_col(cols, "qualifiers")
    idx_asin = _find_col(cols, "asin") or _find_col(cols, "identifier")
    idx_weight = _find_col(cols, "product", "weight") or _find_col(cols, "weight")
    idx_acq = _find_col(cols, "acq", "date")
    idx_sheet = _find_col(cols, "sheetname") or _find_col(cols, "sheet", "name")

    # blocks
    sr_seq  = ["analyte","result","mrl","units","dilution","analyzed","qualifier"]
    mb_seq  = ["analyte","result","mrl","units","dilution"]
    ms1_seq = ["analyte","result","mrl","units","dilution","fortified level","%rec","%rec limits"]
    msd_seq = ["analyte","result","units","dilution","%rec","%rec limits","%rpd","%rpd limit"]

    sr_start  = _find_seq(cols, sr_seq)
    mb_start  = _find_seq(cols, mb_seq)
    ms1_start = _find_seq(cols, ms1_seq)
    msd_start = _find_seq(cols, msd_seq)

    created = updated = skipped_num = skipped_analyte = 0

    db = SessionLocal()
    try:
        for _, rw in df.iterrows():
            lab_id = str(rw.iloc[idx_lab]).strip() if idx_lab is not None else ""
            client = str(rw.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            # sample results
            sr_analyte = ""
            sr = {}
            if sr_start is not None:
                try:
                    sr_analyte = str(rw.iloc[sr_start + 0]).strip()
                    sr = {
                        "result": str(rw.iloc[sr_start + 1]).strip(),
                        "mrl": str(rw.iloc[sr_start + 2]).strip(),
                        "units": str(rw.iloc[sr_start + 3]).strip(),
                        "dilution": str(rw.iloc[sr_start + 4]).strip(),
                        "analyzed": str(rw.iloc[sr_start + 5]).strip(),
                        "qualifier": str(rw.iloc[sr_start + 6]).strip(),
                    }
                except Exception:
                    sr, sr_analyte = {}, ""

            # PFAS or Bisphenol S?
            norm_analyte = _norm_space(sr_analyte)
            is_pfas = _pfas_key(sr_analyte) is not None
            is_bps  = ("bisphenol" in norm_analyte and "s" in norm_analyte)

            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing); created += 1
            else:
                existing.client = client; updated += 1

            # common metadata (seed once if empty)
            if not existing.sample_name:
                existing.sample_name   = str(rw.iloc[idx_sample_name]).strip() if idx_sample_name is not None else lab_id
                existing.prepared_by   = str(rw.iloc[idx_prepared_by]).strip() if idx_prepared_by is not None else ""
                existing.matrix        = str(rw.iloc[idx_matrix]).strip() if idx_matrix is not None else ""
                existing.prepared_date = str(rw.iloc[idx_prepared_date]).strip() if idx_prepared_date is not None else ""
                existing.qualifiers    = str(rw.iloc[idx_qualifiers]).strip() if idx_qualifiers is not None else ""
                existing.asin          = str(rw.iloc[idx_asin]).strip() if idx_asin is not None else ""
                existing.product_weight_g = str(rw.iloc[idx_weight]).strip() if idx_weight is not None else ""
                existing.acq_datetime  = str(rw.iloc[idx_acq]).strip() if idx_acq is not None else ""
                existing.sheet_name    = str(rw.iloc[idx_sheet]).strip() if idx_sheet is not None else ""

                # (phone/email/project lead/address can be wired similarly if included)

            existing.resulted_date = _parse_date(rw.iloc[idx_reported]) if idx_reported is not None else existing.resulted_date
            existing.collected_date = _parse_date(rw.iloc[idx_received]) if idx_received is not None else existing.collected_date

            if is_pfas:
                # Build a per-analyte record including QC
                row_pfas = {"analyte": sr_analyte,
                            "result": sr.get("result",""),
                            "mrl": sr.get("mrl",""),
                            "units": sr.get("units",""),
                            "dilution": sr.get("dilution",""),
                            "analyzed": sr.get("analyzed",""),
                            "qualifier": sr.get("qualifier","")}

                if mb_start is not None:
                    try:
                        row_pfas.update({
                            "mb_analyte": str(rw.iloc[mb_start+0]).strip(),
                            "mb_result":  str(rw.iloc[mb_start+1]).strip(),
                            "mb_mrl":     str(rw.iloc[mb_start+2]).strip(),
                            "mb_units":   str(rw.iloc[mb_start+3]).strip(),
                            "mb_dilution":str(rw.iloc[mb_start+4]).strip(),
                        })
                    except: pass
                if ms1_start is not None:
                    try:
                        row_pfas.update({
                            "ms1_analyte": str(rw.iloc[ms1_start+0]).strip(),
                            "ms1_result":  str(rw.iloc[ms1_start+1]).strip(),
                            "ms1_mrl":     str(rw.iloc[ms1_start+2]).strip(),
                            "ms1_units":   str(rw.iloc[ms1_start+3]).strip(),
                            "ms1_dilution":str(rw.iloc[ms1_start+4]).strip(),
                            "ms1_fortified_level": str(rw.iloc[ms1_start+5]).strip(),
                            "ms1_pct_rec": str(rw.iloc[ms1_start+6]).strip(),
                            "ms1_pct_rec_limits": str(rw.iloc[ms1_start+7]).strip(),
                        })
                    except: pass
                if msd_start is not None:
                    try:
                        row_pfas.update({
                            "msd_analyte": str(rw.iloc[msd_start+0]).strip(),
                            "msd_result":  str(rw.iloc[msd_start+1]).strip(),
                            "msd_units":   str(rw.iloc[msd_start+2]).strip(),
                            "msd_dilution":str(rw.iloc[msd_start+3]).strip(),
                            "msd_pct_rec": str(rw.iloc[msd_start+4]).strip(),
                            "msd_pct_rec_limits": str(rw.iloc[msd_start+5]).strip(),
                            "msd_pct_rpd": str(rw.iloc[msd_start+6]).strip(),
                            "msd_pct_rpd_limit": str(rw.iloc[msd_start+7]).strip(),
                        })
                    except: pass

                lst = _load_pfas_list(existing.pfas_json)
                lst = _merge_pfas(lst, row_pfas)
                existing.pfas_json = _save_pfas_list(lst)
                # mark report as PFAS panel (for dashboard readability)
                existing.test = "PFAS Panel"
                existing.result = ""  # panel has multiple results
            elif is_bps:
                # keep existing BPS mapping
                existing.test = "Bisphenol S"
                existing.result = sr.get("result","")
                existing.sample_mrl = sr.get("mrl","")
                existing.sample_units = sr.get("units","")
                existing.sample_dilution = sr.get("dilution","")
                existing.sample_analyzed = sr.get("analyzed","")
                existing.sample_qualifier = sr.get("qualifier","")

                if mb_start is not None:
                    try:
                        existing.mb_analyte  = str(rw.iloc[mb_start+0]).strip()
                        existing.mb_result   = str(rw.iloc[mb_start+1]).strip()
                        existing.mb_mrl      = str(rw.iloc[mb_start+2]).strip()
                        existing.mb_units    = str(rw.iloc[mb_start+3]).strip()
                        existing.mb_dilution = str(rw.iloc[mb_start+4]).strip()
                    except: pass
                if ms1_start is not None:
                    try:
                        existing.ms1_analyte = str(rw.iloc[ms1_start+0]).strip()
                        existing.ms1_result  = str(rw.iloc[ms1_start+1]).strip()
                        existing.ms1_mrl     = str(rw.iloc[ms1_start+2]).strip()
                        existing.ms1_units   = str(rw.iloc[ms1_start+3]).strip()
                        existing.ms1_dilution= str(rw.iloc[ms1_start+4]).strip()
                        existing.ms1_fortified_level = str(rw.iloc[ms1_start+5]).strip()
                        existing.ms1_pct_rec = str(rw.iloc[ms1_start+6]).strip()
                        existing.ms1_pct_rec_limits = str(rw.iloc[ms1_start+7]).strip()
                    except: pass
                if msd_start is not None:
                    try:
                        existing.msd_analyte = str(rw.iloc[msd_start+0]).strip()
                        existing.msd_result  = str(rw.iloc[msd_start+1]).strip()
                        existing.msd_units   = str(rw.iloc[msd_start+2]).strip()
                        existing.msd_dilution= str(rw.iloc[msd_start+3]).strip()
                        existing.msd_pct_rec = str(rw.iloc[msd_start+4]).strip()
                        existing.msd_pct_rec_limits = str(rw.iloc[msd_start+5]).strip()
                        existing.msd_pct_rpd = str(rw.iloc[msd_start+6]).strip()
                        existing.msd_pct_rpd_limit = str(rw.iloc[msd_start+7]).strip()
                    except: pass
            else:
                # Not a targeted analyte — skip quietly
                skipped_analyte += 1
                continue

        db.commit()
        flash(
            f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_num} non-numeric Lab ID row(s) and {skipped_analyte} non-target analyte row(s).",
            "success"
        )
        log_action(u["username"], u["role"], "upload_csv",
                   f"{filename} -> created {created}, updated {updated}, skipped_non_numeric={skipped_num}, skipped_nontarget={skipped_analyte}")
    except Exception as e:
        db.rollback()
        flash(f"Import failed: {e}", "error")
    finally:
        db.close()

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

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

    data = []
    for r in rows:
        if _load_pfas_list(r.pfas_json):
            # export one row per PFAS analyte (flat)
            for a in _load_pfas_list(r.pfas_json):
                data.append({
                    "Lab ID": r.lab_id, "Client": r.client, "Analyte": a.get("analyte",""),
                    "Result": a.get("result",""), "MRL": a.get("mrl",""), "Units": a.get("units",""),
                    "Dilution": a.get("dilution",""), "Analyzed": a.get("analyzed",""),
                    "Qualifier": a.get("qualifier",""),
                    "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
                    "Received": r.collected_date.isoformat() if r.collected_date else "",
                })
        else:
            data.append({
                "Lab ID": r.lab_id, "Client": r.client, "Analyte": r.test or "",
                "Result": r.result or "", "MRL": r.sample_mrl or "", "Units": r.sample_units or "",
                "Dilution": r.sample_dilution or "", "Analyzed": r.sample_analyzed or "",
                "Qualifier": r.sample_qualifier or "",
                "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
                "Received": r.collected_date.isoformat() if r.collected_date else "",
            })

    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(data)} lines")
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
