import os
import io
from datetime import datetime, date
from typing import List, Optional, Dict

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

    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)

    # High-level sample/analyte row (one row per analyte for a Lab ID)
    test = Column(String, nullable=True)      # analyte (e.g., "Bisphenol S", "PFOS", etc.)
    result = Column(String, nullable=True)    # numeric-as-text or textual

    collected_date = Column(Date, nullable=True)  # "Received Date"
    resulted_date = Column(Date, nullable=True)   # "Reported Date"
    pdf_url = Column(String, nullable=True)

    # Client & sample summary (strings to keep simple)
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

# --- one-time add columns if the DB existed earlier without new fields ---
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
PFAS_KEYS = [
    "pfoa","pfos","pfna","fosaa","n-mefosaa","n-etfosaa",
    "sampap","pfosa","n-mefosa","n-mefose","n-etfosa","n-etfose","disampap"
]
PFAS_KEY_TO_DISPLAY = {
    "pfoa":"PFOA","pfos":"PFOS","pfna":"PFNA","fosaa":"FOSAA","n-mefosaa":"N-MeFOSAA",
    "n-etfosaa":"N-EtFOSAA","sampap":"SAmPAP","pfosa":"PFOSA","n-mefosa":"N-MeFOSA",
    "n-mefose":"N-MeFOSE","n-etfosa":"N-EtFOSA","n-etfose":"N-EtFOSE","disampap":"diSAmPAP"
}
INTERNAL_STANDARDS = {"13c12 bps","13c12 bps.","13c12-bps","d8-bps","d8 bps","13c12bps","d8bps"}

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
    if val is None: return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan","none"}: return None
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%m/%d/%y","%d-%b-%Y"):
        try: return datetime.strptime(s, fmt).date()
        except Exception: pass
    try:
        ts = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except Exception:
        return None

def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

def _is_bps(analyte: str) -> bool:
    return _norm(analyte) == "bisphenol s"

def _is_internal_standard(analyte: str) -> bool:
    return _norm(analyte) in INTERNAL_STANDARDS

def _pfas_key(analyte: str) -> Optional[str]:
    n = _norm(analyte)
    for k in PFAS_KEYS:
        if n == k:
            return k
    return None

def _is_pfas(analyte: str) -> bool:
    return _pfas_key(analyte) is not None

def _first_nonempty(*vals):
    for v in vals:
        if v is None: 
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""

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

# ---------- Dashboard: ONE row per Lab ID ----------
@app.route("/dashboard")
def dashboard():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    lab_id_q = request.args.get("lab_id", "").strip()
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()

    db = SessionLocal()
    try:
        q = db.query(Report)
        if u["role"] == "client":
            q = q.filter(Report.client == u["client_name"])

        if lab_id_q:
            q = q.filter(Report.lab_id == lab_id_q)
        if start:
            sd = parse_date(start)
            if sd: q = q.filter(Report.resulted_date >= sd)
        if end:
            ed = parse_date(end)
            if ed: q = q.filter(Report.resulted_date <= ed)

        # Pull candidate rows, then collapse by lab_id in Python
        rows = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).all()

        grouped: Dict[str, Dict] = {}
        for r in rows:
            g = grouped.setdefault(r.lab_id, {
                "lab_id": r.lab_id,
                "client": r.client,
                "sample_name": _first_nonempty(r.sample_name, r.lab_id),
                "reported": r.resulted_date,
                "mode": None,  # "BPS" or "PFAS"
            })
            # Determine mode for this Lab ID
            if _is_bps(r.test):
                g["mode"] = "BPS"
            elif _is_pfas(r.test):
                if g["mode"] != "BPS":  # BPS wins if present
                    g["mode"] = "PFAS"
            # Keep freshest reported date
            if r.resulted_date and (not g["reported"] or r.resulted_date > g["reported"]):
                g["reported"] = r.resulted_date

        # Build summary rows for template
        summary_rows = []
        for lab_id, g in grouped.items():
            mode = g["mode"] or "BPS"  # default
            display_analyte = "Bisphenol S" if mode == "BPS" else "PFAS Panel"
            summary_rows.append({
                "lab_id": lab_id,
                "client": g["client"],
                "sample_name": g["sample_name"],
                "analyte": display_analyte,
                "result": "",  # leave blank on dashboard; details in report page
                "units": "",
                "reported": g["reported"].isoformat() if g["reported"] else "",
                # which actual record to link? any id for this lab_id will do
                "report_id": db.query(Report.id).filter(Report.lab_id == lab_id).order_by(Report.id.desc()).first()[0],
            })

    finally:
        db.close()

    return render_template("dashboard.html", user=u, report_summaries=summary_rows)

# ---------- Report detail: consolidate 13 PFAS or single BPS ----------
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

    # Pull all rows for this Lab ID
    rows = db.query(Report).filter(Report.lab_id == base.lab_id).all()
    db.close()

    has_bps = any(_is_bps(r.test) for r in rows)
    if has_bps:
        mode = "BPS"
        sel = [r for r in rows if _is_bps(r.test)]
        # pick row that has a non-empty result, else first
        sel.sort(key=lambda rr: (str(rr.result or "") == "", rr.id))
        primary = sel[0]

        sample_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(primary.result),
            "mrl": _first_nonempty(primary.sample_mrl),
            "units": _first_nonempty(primary.sample_units),
            "dilution": _first_nonempty(primary.sample_dilution),
            "analyzed": _first_nonempty(primary.sample_analyzed),
            "qualifier": _first_nonempty(primary.sample_qualifier),
        }]
        mb_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(primary.mb_result),
            "mrl": _first_nonempty(primary.mb_mrl),
            "units": _first_nonempty(primary.mb_units),
            "dilution": _first_nonempty(primary.mb_dilution),
        }]
        ms1_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(primary.ms1_result),
            "mrl": _first_nonempty(primary.ms1_mrl),
            "units": _first_nonempty(primary.ms1_units),
            "dilution": _first_nonempty(primary.ms1_dilution),
            "fortified_level": _first_nonempty(primary.ms1_fortified_level),
            "pct_rec": _first_nonempty(primary.ms1_pct_rec),
            "pct_rec_limits": _first_nonempty(primary.ms1_pct_rec_limits),
        }]
        msd_rows = [{
            "analyte": "Bisphenol S",
            "result": _first_nonempty(primary.msd_result),
            "units": _first_nonempty(primary.msd_units),
            "dilution": _first_nonempty(primary.msd_dilution),
            "pct_rec": _first_nonempty(primary.msd_pct_rec),
            "pct_rec_limits": _first_nonempty(primary.msd_pct_rec_limits),
            "pct_rpd": _first_nonempty(primary.msd_pct_rpd),
            "pct_rpd_limit": _first_nonempty(primary.msd_pct_rpd_limit),
        }]

        # ensure top summary fields are filled
        base.sample_name = _first_nonempty(base.sample_name, base.lab_id)
    else:
        mode = "PFAS"
        # group PFAS rows by analyte key and choose best row per analyte
        by_key: Dict[str, List[Report]] = {}
        for r in rows:
            k = _pfas_key(r.test)
            if not k:
                continue
            by_key.setdefault(k, []).append(r)

        chosen: Dict[str, Report] = {}
        for k, items in by_key.items():
            items.sort(key=lambda rr: (str(rr.result or "") == "", rr.id))
            chosen[k] = items[0]

        sample_rows, mb_rows, ms1_rows, msd_rows = [], [], [], []
        for k in PFAS_KEYS:
            r = chosen.get(k)
            if not r:
                # If an analyte is truly absent from the file, skip it (keeps to present values only)
                continue
            disp = PFAS_KEY_TO_DISPLAY[k]
            sample_rows.append({
                "analyte": disp,
                "result": _first_nonempty(r.result),
                "mrl": _first_nonempty(r.sample_mrl),
                "units": _first_nonempty(r.sample_units),
                "dilution": _first_nonempty(r.sample_dilution),
                "analyzed": _first_nonempty(r.sample_analyzed),
                "qualifier": _first_nonempty(r.sample_qualifier),
            })
            mb_rows.append({
                "analyte": disp,
                "result": _first_nonempty(r.mb_result),
                "mrl": _first_nonempty(r.mb_mrl),
                "units": _first_nonempty(r.mb_units),
                "dilution": _first_nonempty(r.mb_dilution),
            })
            ms1_rows.append({
                "analyte": disp,
                "result": _first_nonempty(r.ms1_result),
                "mrl": _first_nonempty(r.ms1_mrl),
                "units": _first_nonempty(r.ms1_units),
                "dilution": _first_nonempty(r.ms1_dilution),
                "fortified_level": _first_nonempty(r.ms1_fortified_level),
                "pct_rec": _first_nonempty(r.ms1_pct_rec),
                "pct_rec_limits": _first_nonempty(r.ms1_pct_rec_limits),
            })
            msd_rows.append({
                "analyte": disp,
                "result": _first_nonempty(r.msd_result),
                "units": _first_nonempty(r.msd_units),
                "dilution": _first_nonempty(r.msd_dilution),
                "pct_rec": _first_nonempty(r.msd_pct_rec),
                "pct_rec_limits": _first_nonempty(r.msd_pct_rec_limits),
                "pct_rpd": _first_nonempty(r.msd_pct_rpd),
                "pct_rpd_limit": _first_nonempty(r.msd_pct_rpd_limit),
            })

        # Prefer summary fields from any PFAS row that has them
        pick = rows[0]
        base.sample_name = _first_nonempty(base.sample_name, pick.sample_name, base.lab_id)
        base.prepared_by = _first_nonempty(base.prepared_by, pick.prepared_by)
        base.matrix = _first_nonempty(base.matrix, pick.matrix)
        base.prepared_date = _first_nonempty(base.prepared_date, pick.prepared_date)
        base.asin = _first_nonempty(base.asin, pick.asin)
        base.product_weight_g = _first_nonempty(base.product_weight_g, pick.product_weight_g)
        base.qualifiers = _first_nonempty(base.qualifiers, pick.qualifiers)
        base.phone = _first_nonempty(base.phone, pick.phone)
        base.email = _first_nonempty(base.email, pick.email)
        base.project_lead = _first_nonempty(base.project_lead, pick.project_lead)
        base.address = _first_nonempty(base.address, pick.address)

    return render_template(
        "report_detail.html",
        user=u, r=base,
        sample_rows=sample_rows,
        mb_rows=mb_rows,
        ms1_rows=ms1_rows,
        msd_rows=msd_rows,
        mode=mode,
    )

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

    # read as "raw" to locate header row that contains "Sample ID"
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

    header_row = None
    for i in range(min(10, len(raw))):
        vals = [str(x) for x in raw.iloc[i].values]
        if any("sample id" in _norm(v) for v in vals):
            header_row = i
            break

    if header_row is None:
        flash("Could not locate header row with 'Sample ID'.", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    headers = [str(x).strip() for x in raw.iloc[header_row].values]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")

    msg = _ingest_master_upload(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Expected header row 2 (actual):
    Sample ID ... | Client | Phone | Email | Project Lead | Address |
    Reported | Received Date | Sample Name | Prepared By | Matrix | Prepared Date | Qualifiers | ASIN (Identifier) | Product Weight (Grams) |
    Analyte | Result | MRL | Units | Dilution | Analyzed | Qualifier |
    Analyte | Result | MRL | Units | Dilution | Analyzed | Qualifier |   <-- (duplicated by vendor sometimes)
    Analyte | Result | MRL | Units | Dilution |                              <-- Method Blank
    ... (MS1 block) ...
    ... (MSD block) ...
    """
    # map needed columns by exact or fuzzy contains
    cols = list(df.columns)
    def c(*tokens):
        tl = [t.lower() for t in tokens]
        for i, name in enumerate(cols):
            n = _norm(name)
            if all(t in n for t in tl):
                return name
        return None

    col_lab   = c("sample","id")  # "Sample ID (Lab ID, Laboratory ID)"
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

    # Locate the repeating blocks by counting forward from the first "Analyte"
    analyte_cols = [i for i, name in enumerate(cols) if _norm(name) == "analyte"]
    # We expect: SR (7), MB (5), MS1 (8), MSD (8) — same as before
    def block(start, length):
        return [cols[start + j] if start is not None and start + j < len(cols) else None for j in range(length)]

    # Identify first SR block
    sr_start = analyte_cols[0] if analyte_cols else None
    SR = block(sr_start, 7)
    # Next "Analyte" after SR for MB
    mb_start = analyte_cols[1] if len(analyte_cols) > 1 else None
    MB = block(mb_start, 5)
    # Next for MS1
    ms1_start = analyte_cols[2] if len(analyte_cols) > 2 else None
    MS1 = block(ms1_start, 8)
    # Next for MSD
    msd_start = analyte_cols[3] if len(analyte_cols) > 3 else None
    MSD = block(msd_start, 8)

    created = updated = skipped_num = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.get(col_lab, "")).strip()
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            client = str(row.get(col_client, CLIENT_NAME)).strip() or CLIENT_NAME

            # --- Sample Results row analyte ---
            sr_analyte = str(row.get(SR[0], "")).strip() if SR[0] else ""
            if _is_internal_standard(sr_analyte):
                # skip internal standard rows entirely
                continue

            is_bps = _is_bps(sr_analyte)
            pfas_k = _pfas_key(sr_analyte)

            # Ignore any non-target rows (neither BPS nor PFAS 13)
            if not is_bps and not pfas_k:
                continue

            existing = db.query(Report).filter(Report.lab_id == lab_id, Report.test == (sr_analyte or "")).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client, test=sr_analyte)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

            # Top info / meta
            existing.sample_name   = str(row.get(col_sname, lab_id)).strip()
            existing.phone         = str(row.get(col_phone, "")).strip()
            existing.email         = str(row.get(col_email, "")).strip()
            existing.project_lead  = str(row.get(col_pjlead, "")).strip()
            existing.address       = str(row.get(col_addr, "")).strip()
            existing.resulted_date = parse_date(row.get(col_rep))
            existing.collected_date= parse_date(row.get(col_recv))
            existing.prepared_by   = str(row.get(col_prepby, "")).strip()
            existing.matrix        = str(row.get(col_matrix, "")).strip()
            existing.prepared_date = str(row.get(col_prepdt, "")).strip()
            existing.qualifiers    = str(row.get(col_qual, "")).strip()
            existing.asin          = str(row.get(col_asin, "")).strip()
            existing.product_weight_g = str(row.get(col_weight, "")).strip()

            existing.acq_datetime  = str(row.get(c("acq","date"), "")).strip()
            existing.sheet_name    = str(row.get(c("sheetname") or c("sheet","name"), "")).strip()

            # Sample Results data
            existing.result         = str(row.get(SR[1], "")).strip() if SR[1] else ""
            existing.sample_mrl     = str(row.get(SR[2], "")).strip() if SR[2] else ""
            existing.sample_units   = str(row.get(SR[3], "")).strip() if SR[3] else ""
            existing.sample_dilution= str(row.get(SR[4], "")).strip() if SR[4] else ""
            existing.sample_analyzed= str(row.get(SR[5], "")).strip() if SR[5] else ""
            existing.sample_qualifier= str(row.get(SR[6], "")).strip() if SR[6] else ""

            # Method Blank
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

# ----------- Audit & export (unchanged) ----------
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
