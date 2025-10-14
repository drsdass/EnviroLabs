import os
import io
from datetime import datetime, date
from typing import List, Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
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

# ------------------- Models -------------------
class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)

    # Identity
    lab_id = Column(String, nullable=False, index=True, unique=True)
    client = Column(String, nullable=False, index=True)

    # Legacy “patient” field (unused for Enviro Labs)
    patient_name = Column(String, nullable=True)

    # Sample-level dates / pdf-url
    collected_date = Column(Date, nullable=True)  # Received
    resulted_date  = Column(Date, nullable=True)  # Reported
    pdf_url = Column(String, nullable=True)

    # Client info coming from CSV (row 2 area)
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

    # Misc
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # NEW: a report has many analytes
    analytes = relationship("ReportAnalyte", back_populates="report", cascade="all, delete-orphan", lazy="selectin")

class ReportAnalyte(Base):
    """
    One row per analyte (BPS or one of the 13 PFAS) for a given Report.
    Holds the Sample Results + QC (MB, MS1, MSD) for that analyte.
    """
    __tablename__ = "report_analytes"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)

    # normalized analyte key for uniqueness; display_name keeps original
    analyte_key = Column(String, nullable=False)
    display_name = Column(String, nullable=False)

    # Sample Results
    result = Column(String, nullable=True)
    mrl = Column(String, nullable=True)
    units = Column(String, nullable=True)
    dilution = Column(String, nullable=True)
    analyzed = Column(String, nullable=True)
    qualifier = Column(String, nullable=True)

    # Method Blank
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)

    # Matrix Spike 1
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)

    # Matrix Spike Duplicate
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    report = relationship("Report", back_populates="analytes")

    __table_args__ = (
        UniqueConstraint("report_id", "analyte_key", name="uq_report_analyte"),
    )

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

def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

# Accepted sets
_PFAS_SET = {
    "pfoa","pfos","pfna","fosaa","n mefosaa","n etfosaa","sampap",
    "pfosa","n mefosa","n mefose","n etfosa","n etfose","disampap"
}
_BPS_ALIASES = {"bisphenol s", "bps"}

def _is_supported_analyte(analyte: str) -> bool:
    a = _norm(analyte)
    return (a in _BPS_ALIASES) or (a in _PFAS_SET) or ("pfas" in a)

def _analyte_key(analyte: str) -> str:
    a = _norm(analyte)
    # collapse common spellings to our keys
    if a in _BPS_ALIASES:
        return "bisphenol s"
    if a in _PFAS_SET:
        return a
    if "pfas" in a:
        return "pfas"  # generic bucket if ever present
    return a

# --- tiny column-finders for the Master Upload header row parsing ---
def _find_token_col(cols: List[str], *needles: str) -> Optional[int]:
    tokens = [t.lower() for t in needles]
    for i, c in enumerate(cols):
        name = _norm(c)
        if all(tok in name for tok in tokens):
            return i
    return None

def _find_sequence(cols_lower: List[str], seq: List[str]) -> Optional[int]:
    n = len(cols_lower)
    m = len(seq)
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            if cols_lower[i + j].strip() != seq[j]:
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
    analytes = db.query(ReportAnalyte).filter(ReportAnalyte.report_id == report_id).order_by(ReportAnalyte.display_name.asc()).all()
    db.close()

    # Pack for template (keeps your existing structure but swaps arrays into tables)
    p = {
        "client_info": {
            "client": r.client or "",
            "phone": r.phone or "",
            "email": r.email or "support@envirolabsusa.com",
            "project_lead": r.project_lead or "",
            "address": r.address or "",
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": r.sample_name or r.lab_id or "",
            "prepared_by": r.prepared_by or "",
            "matrix": r.matrix or "",
            "prepared_date": r.prepared_date or "",
            "qualifiers": r.qualifiers or "",
            "asin": r.asin or "",
            "product_weight_g": r.product_weight_g or "",
        },
        "acq_datetime": r.acq_datetime or "",
        "sheet_name": r.sheet_name or "",
    }

    return render_template("report_detail.html", user=u, r=r, p=p, analytes=analytes)

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

    # Read as raw (no header), find header row that contains "Sample ID"
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
    for i in range(min(12, len(raw))):
        row_vals = [str(x) for x in list(raw.iloc[i].values)]
        if any("sample id" in _norm(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        flash("Could not detect header row (no 'Sample ID' row found).", "error")
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
    Master Upload parser -> creates/updates one Report per Sample/Lab ID.
    For each row that has a supported analyte (BPS or 13 PFAS), upserts a ReportAnalyte.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)
    cols_lower = [c.lower().strip() for c in cols]

    # Single columns (positions may vary)
    idx_lab          = _find_token_col(cols, "sample", "id")
    idx_client       = _find_token_col(cols, "client")
    idx_reported     = _find_token_col(cols, "reported")
    idx_received     = _find_token_col(cols, "received", "date")
    idx_sample_name  = _find_token_col(cols, "sample", "name")
    idx_prepared_by  = _find_token_col(cols, "prepared", "by")
    idx_matrix       = _find_token_col(cols, "matrix")
    idx_prepared_date= _find_token_col(cols, "prepared", "date")
    idx_qualifiers   = _find_token_col(cols, "qualifiers")
    idx_asin         = _find_token_col(cols, "asin") or _find_token_col(cols, "identifier")
    idx_weight       = _find_token_col(cols, "product", "weight") or _find_token_col(cols, "weight")
    idx_acq          = _find_token_col(cols, "acq", "date")  # "Acq. Date-Time"
    idx_sheet        = _find_token_col(cols, "sheetname") or _find_token_col(cols, "sheet", "name")

    # Blocks by repeated caption sequences
    sr_seq   = ["analyte", "result", "mrl", "units", "dilution", "analyzed", "qualifier"]
    mb_seq   = ["analyte", "result", "mrl", "units", "dilution"]
    ms1_seq  = ["analyte", "result", "mrl", "units", "dilution", "fortified level", "%rec", "%rec limits"]
    msd_seq  = ["analyte", "result", "units", "dilution", "%rec", "%rec limits", "%rpd", "%rpd limit"]

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
            lab_id = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            # client
            client = str(row.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME

            # Sample Results — analyte name is what we key on
            if sr_start is None:
                continue  # no sample results section found

            analyte_name = str(row.iloc[sr_start + 0]).strip()
            if not _is_supported_analyte(analyte_name):
                skipped_analyte += 1
                continue

            akey = _analyte_key(analyte_name)
            # Upsert Report
            rpt = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if rpt is None:
                rpt = Report(lab_id=lab_id, client=client)
                db.add(rpt)
                created_reports += 1
            else:
                rpt.client = client
                updated_reports += 1

            # Fill sample-level info only if not already present (so later rows don’t wipe earlier)
            if rpt.sample_name in (None, "") and idx_sample_name is not None:
                rpt.sample_name = str(row.iloc[idx_sample_name]).strip() or lab_id
            if rpt.prepared_by in (None, "") and idx_prepared_by is not None:
                rpt.prepared_by = str(row.iloc[idx_prepared_by]).strip()
            if rpt.matrix in (None, "") and idx_matrix is not None:
                rpt.matrix = str(row.iloc[idx_matrix]).strip()
            if rpt.prepared_date in (None, "") and idx_prepared_date is not None:
                rpt.prepared_date = str(row.iloc[idx_prepared_date]).strip()
            if rpt.qualifiers in (None, "") and idx_qualifiers is not None:
                rpt.qualifiers = str(row.iloc[idx_qualifiers]).strip()
            if rpt.asin in (None, "") and idx_asin is not None:
                rpt.asin = str(row.iloc[idx_asin]).strip()
            if rpt.product_weight_g in (None, "") and idx_weight is not None:
                rpt.product_weight_g = str(row.iloc[idx_weight]).strip()
            if rpt.resulted_date is None and idx_reported is not None:
                rpt.resulted_date = parse_date(row.iloc[idx_reported])
            if rpt.collected_date is None and idx_received is not None:
                rpt.collected_date = parse_date(row.iloc[idx_received])
            if rpt.acq_datetime in (None, "") and idx_acq is not None:
                rpt.acq_datetime = str(row.iloc[idx_acq]).strip()
            if rpt.sheet_name in (None, "") and idx_sheet is not None:
                rpt.sheet_name = str(row.iloc[idx_sheet]).strip()

            # Per-analyte upsert
            ra = db.query(ReportAnalyte).filter(
                ReportAnalyte.report_id == rpt.id,
                ReportAnalyte.analyte_key == akey
            ).one_or_none()

            if ra is None:
                ra = ReportAnalyte(report=rpt, analyte_key=akey, display_name=analyte_name)
                db.add(ra)
                created_analytes += 1
            else:
                ra.display_name = analyte_name
                updated_analytes += 1

            # Sample results values
            ra.result    = str(row.iloc[sr_start + 1]).strip()
            ra.mrl       = str(row.iloc[sr_start + 2]).strip()
            ra.units     = str(row.iloc[sr_start + 3]).strip()
            ra.dilution  = str(row.iloc[sr_start + 4]).strip()
            ra.analyzed  = str(row.iloc[sr_start + 5]).strip()
            ra.qualifier = str(row.iloc[sr_start + 6]).strip()

            # Method Blank (aligned by position with its own “Analyte” column; we don’t need MB analyte name
            if mb_start is not None:
                ra.mb_result  = str(row.iloc[mb_start + 1]).strip()
                ra.mb_mrl     = str(row.iloc[mb_start + 2]).strip()
                ra.mb_units   = str(row.iloc[mb_start + 3]).strip()
                ra.mb_dilution= str(row.iloc[mb_start + 4]).strip()

            # Matrix Spike 1
            if ms1_start is not None:
                ra.ms1_result          = str(row.iloc[ms1_start + 1]).strip()
                ra.ms1_mrl             = str(row.iloc[ms1_start + 2]).strip()
                ra.ms1_units           = str(row.iloc[ms1_start + 3]).strip()
                ra.ms1_dilution        = str(row.iloc[ms1_start + 4]).strip()
                ra.ms1_fortified_level = str(row.iloc[ms1_start + 5]).strip()
                ra.ms1_pct_rec         = str(row.iloc[ms1_start + 6]).strip()
                ra.ms1_pct_rec_limits  = str(row.iloc[ms1_start + 7]).strip()

            # Matrix Spike Duplicate
            if msd_start is not None:
                ra.msd_result        = str(row.iloc[msd_start + 1]).strip()
                ra.msd_units         = str(row.iloc[msd_start + 2]).strip()
                ra.msd_dilution      = str(row.iloc[msd_start + 3]).strip()
                ra.msd_pct_rec       = str(row.iloc[msd_start + 4]).strip()
                ra.msd_pct_rec_limits= str(row.iloc[msd_start + 5]).strip()
                ra.msd_pct_rpd       = str(row.iloc[msd_start + 6]).strip()
                ra.msd_pct_rpd_limit = str(row.iloc[msd_start + 7]).strip()

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (
        f"Reports: +{created_reports}/~{updated_reports} updated. "
        f"Analytes: +{created_analytes}/~{updated_analytes} updated. "
        f"Skipped {skipped_num} non-numeric Lab ID row(s) and {skipped_analyte} non-target analyte row(s)."
    )

# ----------- Audit & Export -----------
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

    # Export flattened analytes (one row per analyte)
    db = SessionLocal()
    q = db.query(Report).all()
    rows = []
    for r in q:
        for a in r.analytes:
            rows.append({
                "Lab ID": r.lab_id,
                "Client": r.client,
                "Sample Name": r.sample_name or "",
                "Analyte": a.display_name,
                "Result": a.result or "",
                "MRL": a.mrl or "",
                "Units": a.units or "",
                "Dilution": a.dilution or "",
                "Analyzed": a.analyzed or "",
                "Qualifier": a.qualifier or "",
                "MB Result": a.mb_result or "",
                "MB MRL": a.mb_mrl or "",
                "MB Units": a.mb_units or "",
                "MB Dilution": a.mb_dilution or "",
                "MS1 Result": a.ms1_result or "",
                "MS1 MRL": a.ms1_mrl or "",
                "MS1 Units": a.ms1_units or "",
                "MS1 Dilution": a.ms1_dilution or "",
                "MS1 Fortified Level": a.ms1_fortified_level or "",
                "MS1 %REC": a.ms1_pct_rec or "",
                "MS1 %REC Limits": a.ms1_pct_rec_limits or "",
                "MSD Result": a.msd_result or "",
                "MSD Units": a.msd_units or "",
                "MSD Dilution": a.msd_dilution or "",
                "MSD %REC": a.msd_pct_rec or "",
                "MSD %REC Limits": a.msd_pct_rec_limits or "",
                "MSD %RPD": a.msd_pct_rpd or "",
                "MSD %RPD Limit": a.msd_pct_rpd_limit or "",
                "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
                "Received": r.collected_date.isoformat() if r.collected_date else "",
            })
    db.close()

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(rows)} rows (flattened analytes)")
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
