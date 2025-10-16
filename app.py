import os
import io
from datetime import datetime, date
from typing import List, Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify, render_template_string
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

    # (legacy field, harmless)
    patient_name = Column(String, nullable=True)

    # Sample Results (primary)
    test = Column(String, nullable=True)      # analyte (e.g., "Bisphenol S", "PFOA", etc.)
    result = Column(String, nullable=True)    # numeric-as-text or textual

    collected_date = Column(Date, nullable=True)  # "Received Date"
    resulted_date = Column(Date, nullable=True)   # "Reported Date"
    pdf_url = Column(String, nullable=True)

    # ---- Optional metadata (strings for SQLite simplicity) ----
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

# ----- PFAS analyte mapping -----
PFAS_DISPLAY = [
    "PFOA","PFOS","PFNA","FOSAA","N-MeFOSAA","N-EtFOSAA","SAmPAP",
    "PFOSA","N-MeFOSA","N-MeFOSE","N-EtFOSA","N-EtFOSE","diSAmPAP"
]
def _norm_analyte_key(name: str) -> str:
    return (_norm(name or "")
            .replace("n me", "n-me")
            .replace("n et", "n-et")
            .replace("disampap", "diSAmPAP".lower()))

PFAS_KEYS = [ _norm_analyte_key(x) for x in PFAS_DISPLAY ]
PFAS_KEY_TO_DISPLAY = { _norm_analyte_key(x): x for x in PFAS_DISPLAY }

def _is_pfas(analyte: str) -> bool:
    key = _norm_analyte_key(analyte)
    return key in PFAS_KEYS

def _is_bps(analyte: str) -> bool:
    return "bisphenol s" in _norm(analyte)

def _first_nonempty(*vals: Optional[str]) -> str:
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

# >>> FIXED: include the path parameter in the route <<<
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

    # All rows for this Lab ID (PFAS: up to 13 rows; BPS: 1 row)
    rows = db.query(Report).filter(Report.lab_id == base.lab_id).all()
    db.close()

    # Decide panel: if any BPS row exists -> BPS report; else PFAS report
    has_bps = any(_is_bps(r.test) for r in rows)
    if has_bps:
        mode = "BPS"
        # Keep ONLY BPS rows; choose the "best" one (has result preferred)
        sel = [r for r in rows if _is_bps(r.test)]
        sel.sort(key=lambda r: (str(r.result or "") == "", r.id))
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

        # Client/summary fallbacks
        base.sample_name = _first_nonempty(base.sample_name, base.lab_id)

    else:
        mode = "PFAS"
        # Keep ONLY the 13 PFAS rows; group by analyte key; pick best row per analyte
        pfas_rows_by_key = {}
        for r in rows:
            if not _is_pfas(r.test):
                continue
            key = _norm_analyte_key(r.test)
            pfas_rows_by_key.setdefault(key, []).append(r)

        chosen = {}
        for key, items in pfas_rows_by_key.items():
            items.sort(key=lambda rr: (str(rr.result or "") == "", rr.id))
            chosen[key] = items[0]

        sample_rows, mb_rows, ms1_rows, msd_rows = [], [], [], []
        for key in PFAS_KEYS:
            r = chosen.get(key)
            if not r:
                # If an analyte is missing in the file, skip it (keeps the list clean)
                continue
            disp = PFAS_KEY_TO_DISPLAY.get(key, r.test or "")
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
        user=u, r=base, mode=mode,
        sample_rows=sample_rows, mb_rows=mb_rows, ms1_rows=ms1_rows, msd_rows=msd_rows
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

    # Read with header=None (banner row safe); find true header row by “Sample ID”
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

    header_row_idx = None
    for i in range(min(10, len(raw))):
        row_vals = [str(x) for x in list(raw.iloc[i].values)]
        if any("sample id" in _norm(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        flash("Could not locate header row (looking for 'Sample ID')", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]

    msg = _ingest_master_upload(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Master Upload File layout (banner row + true header).
    Only create report rows when Lab ID starts with a digit.
    PFAS rows are stored individually (one per analyte) and merged at render time.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    # Locate single columns by tokens
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

    # Blocks by column sequences
    sr_seq  = ["analyte", "result", "mrl", "units", "dilution", "analyzed", "qualifier"]
    mb_seq  = ["analyte", "result", "mrl", "units", "dilution"]
    ms1_seq = ["analyte", "result", "mrl", "units", "dilution", "fortified level", "%rec", "%rec limits"]
    msd_seq = ["analyte", "result", "units", "dilution", "%rec", "%rec limits", "%rpd", "%rpd limit"]

    sr_start  = _find_sequence([c.lower() for c in cols], sr_seq)
    mb_start  = _find_sequence([c.lower() for c in cols], mb_seq)
    ms1_start = _find_sequence([c.lower() for c in cols], ms1_seq)
    msd_start = _find_sequence([c.lower() for c in cols], msd_seq)

    created = updated = skipped_num = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            client = str(row.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME

            # Sample Results (one row per analyte present in this row)
            analyte = ""
            sr = {}
            if sr_start is not None:
                try:
                    analyte = str(row.iloc[sr_start + 0]).strip()
                    sr = {
                        "result": str(row.iloc[sr_start + 1]).strip(),
                        "mrl": str(row.iloc[sr_start + 2]).strip(),
                        "units": str(row.iloc[sr_start + 3]).strip(),
                        "dilution": str(row.iloc[sr_start + 4]).strip(),
                        "analyzed": str(row.iloc[sr_start + 5]).strip(),
                        "qualifier": str(row.iloc[sr_start + 6]).strip(),
                    }
                except Exception:
                    analyte, sr = "", {}

            # Create/update a DB row for this analyte (PFAS or BPS)
            existing = db.query(Report).filter(Report.lab_id == lab_id, Report.test == analyte).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client, test=analyte)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

            # top metadata (same on every analyte row; harmless duplicates)
            existing.sample_name = str(row.iloc[idx_sample_name]).strip() if idx_sample_name is not None else lab_id
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

            # sample
            existing.result = sr.get("result", "")
            existing.sample_mrl = sr.get("mrl", "")
            existing.sample_units = sr.get("units", "")
            existing.sample_dilution = sr.get("dilution", "")
            existing.sample_analyzed = sr.get("analyzed", "")
            existing.sample_qualifier = sr.get("qualifier", "")

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
            f"Skipped {skipped_num} non-numeric Lab ID row(s).")

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
