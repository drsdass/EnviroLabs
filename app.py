import os
import io
from datetime import datetime, date
from typing import List, Optional, Tuple

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Text,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
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
    lab_id = Column(String, nullable=False, index=True, unique=True)
    client = Column(String, nullable=False, index=True)

    # sample-level / client info
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

    collected_date = Column(Date, nullable=True)  # Received Date
    resulted_date  = Column(Date, nullable=True)  # Reported
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)

    pdf_url = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    analytes = relationship("ReportAnalyte", back_populates="report",
                            cascade="all, delete-orphan", lazy="selectin")

class ReportAnalyte(Base):
    __tablename__ = "report_analytes"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)

    analyte_key = Column(String, nullable=False)   # normalized
    display_name = Column(String, nullable=False)

    # SAMPLE RESULTS
    result = Column(String, nullable=True)
    mrl = Column(String, nullable=True)
    units = Column(String, nullable=True)
    dilution = Column(String, nullable=True)
    analyzed = Column(String, nullable=True)
    qualifier = Column(String, nullable=True)

    # METHOD BLANK
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)

    # MATRIX SPIKE 1
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)

    # MATRIX SPIKE DUPLICATE
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
    __table_args__ = (UniqueConstraint("report_id", "analyte_key", name="uq_report_analyte"),)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)
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
        db.add(AuditLog(username=username or "system", role=role or "system", action=action, details=details))
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

_PFAS_SET = {
    "pfoa","pfos","pfna","fosaa","n mefosaa","n etfosaa","sampap",
    "pfosa","n mefosa","n mefose","n etfosa","n etfose","disampap"
}
_BPS_ALIASES = {"bisphenol s", "bps"}
def _is_supported_analyte(analyte: str) -> bool:
    a = _norm(analyte)
    return (a in _BPS_ALIASES) or (a in _PFAS_SET)
def _akey(analyte: str) -> str:
    return _norm(analyte)

# ---------- two-row header tools ----------
def _detect_two_row_header(raw: pd.DataFrame) -> Tuple[int,int]:
    """
    Returns (row1_index, row2_index) where:
      - row1 contains section titles like 'CLIENT INFORMATION', 'SAMPLE SUMMARY' ...
      - row2 contains real column names including 'Sample ID', 'Analyte', etc.
    """
    r2 = None
    for i in range(min(15, len(raw))):
        vals = [str(x) for x in raw.iloc[i].values]
        line = " | ".join(vals).lower()
        if ("sample id" in line) and ("analyte" in line):
            r2 = i
            break
    if r2 is None:
        # fallback: row with 'Sample ID' only
        for i in range(min(15, len(raw))):
            vals = [str(x) for x in raw.iloc[i].values]
            if any("sample id" in _norm(v) for v in vals):
                r2 = i
                break
    if r2 is None or r2 - 1 < 0:
        raise ValueError("Could not detect the two-row header (need section row above the 'Sample ID' row).")
    r1 = r2 - 1
    return r1, r2

def _build_section_columns(header1: List[str], header2: List[str]) -> List[Tuple[str,str]]:
    """
    Given row1 (section titles) and row2 (column names), build a list of (Section, Field).
    Section names are forward-filled to the right.
    """
    sec = ""
    out = []
    for s, f in zip(header1, header2):
        if str(s).strip() != "":
            sec = str(s).strip()
        out.append((sec, str(f).strip()))
    return out

def _find_col_idx(cols: List[Tuple[str,str]], section_hint: str, field_hint: str) -> Optional[int]:
    sh = _norm(section_hint)
    fh = _norm(field_hint)
    for i, (sec, fld) in enumerate(cols):
        if sh in _norm(sec) and fh in _norm(fld):
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
    analytes = r.analytes[:]  # loaded via relationship (selectin)
    db.close()

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

    # Read without header; we will detect the two-row header
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

    try:
        r1, r2 = _detect_two_row_header(raw)
    except Exception as e:
        flash(str(e), "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    header1 = [str(x).strip() for x in raw.iloc[r1].values]
    header2 = [str(x).strip() for x in raw.iloc[r2].values]
    cols = _build_section_columns(header1, header2)

    df = raw.iloc[r2+1:].copy()
    df.columns = pd.MultiIndex.from_tuples(cols, names=["Section","Field"])
    # drop fully empty rows
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]

    msg = _ingest_with_sections(df)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _ingest_with_sections(df: pd.DataFrame) -> str:
    """
    df columns are MultiIndex (Section, Field) as parsed from the two-row header.
    We import one Report per Lab ID and a ReportAnalyte per row's SAMPLE RESULTS analyte.
    """
    def get(r, sec, fld):
        try:
            return str(r[(sec, fld)])
        except Exception:
            return ""

    created_reports = updated_reports = 0
    created_analytes = updated_analytes = 0
    skipped_num = skipped_analyte = 0

    # Helpful section labels (case insensitive matching)
    def secn(s): return s  # keep original
    CI  = "CLIENT INFORMATION"
    SS  = "SAMPLE SUMMARY"
    SR  = "SAMPLE RESULTS"
    MB  = "METHOD BLANK"
    MS1 = "MATRIX SPIKE 1"
    MSD = "MATRIX SPIKE DUPLICATE"

    # Column names we expect on row 2 (flexibly matched downstream by exact text)
    # You provided exact field labels, so we’ll use those directly.
    # CLIENT INFORMATION
    f_lab     = "Sample ID (Lab ID, Laboratory ID)"
    f_client  = "Client"
    f_phone   = "Phone"
    f_email   = "Email"
    f_pjlead  = "Project Lead"
    f_addr    = "Address"

    # SAMPLE SUMMARY
    f_reported = "Reported"
    f_received = "Received Date"
    f_sample   = "Sample Name"
    f_prepby   = "Prepared By"
    f_matrix   = "Matrix"
    f_prepdate = "Prepared Date"
    f_quals    = "Qualifiers"
    f_asin     = "ASIN (Identifier)"
    f_weight   = "Product Weight (Grams)"

    # SR block
    f_analyte  = "Analyte"
    f_result   = "Result"
    f_mrl      = "MRL"
    f_units    = "Units"
    f_dil      = "Dilution"
    f_analyzed = "Analyzed"
    f_qual     = "Qualifier"

    # MS1 extras
    f_fort     = "Fortified Level"
    f_pctrec   = "%REC"
    f_pctrecl  = "%REC Limits"

    # MSD extras
    f_msdpctrec   = "%REC"
    f_msdpctrecl  = "%REC Limits"
    f_rpd         = "%RPD"
    f_rpdl        = "%RPD Limit"

    # trailing misc (outside the 4 blocks)
    f_acq     = "Acq. Date-Time"
    f_sheet   = "SheetName"

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = get(row, CI, f_lab).strip()
            if not _lab_id_is_numericish(lab_id):
                skipped_num += 1
                continue

            client = (get(row, CI, f_client) or CLIENT_NAME).strip()

            # sample results analyte drives whether we keep the row
            analyte_name = get(row, SR, f_analyte).strip()
            if not _is_supported_analyte(analyte_name):
                skipped_analyte += 1
                continue
            akey = _akey(analyte_name)

            # upsert report
            rpt = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if rpt is None:
                rpt = Report(lab_id=lab_id, client=client)
                db.add(rpt)
                created_reports += 1
            else:
                rpt.client = client
                updated_reports += 1

            # fill sample-level info (first non-empty wins)
            def set_if_empty(attr, value):
                cur = getattr(rpt, attr)
                if (cur is None or cur == "") and (value not in (None, "")):
                    setattr(rpt, attr, value)

            set_if_empty("phone",         get(row, CI, f_phone).strip())
            set_if_empty("email",         get(row, CI, f_email).strip())
            set_if_empty("project_lead",  get(row, CI, f_pjlead).strip())
            set_if_empty("address",       get(row, CI, f_addr).strip())

            set_if_empty("sample_name",   (get(row, SS, f_sample).strip() or lab_id))
            set_if_empty("prepared_by",   get(row, SS, f_prepby).strip())
            set_if_empty("matrix",        get(row, SS, f_matrix).strip())
            set_if_empty("prepared_date", get(row, SS, f_prepdate).strip())
            set_if_empty("qualifiers",    get(row, SS, f_quals).strip())
            set_if_empty("asin",          get(row, SS, f_asin).strip())
            set_if_empty("product_weight_g", get(row, SS, f_weight).strip())

            if rpt.resulted_date is None:
                rpt.resulted_date = parse_date(get(row, SS, f_reported))
            if rpt.collected_date is None:
                rpt.collected_date = parse_date(get(row, SS, f_received))

            set_if_empty("acq_datetime", get(row, "", f_acq).strip() if (("", f_acq) in df.columns) else get(row, SS, f_acq).strip())
            set_if_empty("sheet_name",   get(row, "", f_sheet).strip() if (("", f_sheet) in df.columns) else get(row, SS, f_sheet).strip())

            # upsert analyte
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

            # SR fields
            ra.result    = get(row, SR, f_result).strip()
            ra.mrl       = get(row, SR, f_mrl).strip()
            ra.units     = get(row, SR, f_units).strip()
            ra.dilution  = get(row, SR, f_dil).strip()
            ra.analyzed  = get(row, SR, f_analyzed).strip()
            ra.qualifier = get(row, SR, f_qual).strip()

            # MB fields
            ra.mb_result   = get(row, MB, f_result).strip()
            ra.mb_mrl      = get(row, MB, f_mrl).strip()
            ra.mb_units    = get(row, MB, f_units).strip()
            ra.mb_dilution = get(row, MB, f_dil).strip()

            # MS1 fields
            ra.ms1_result          = get(row, MS1, f_result).strip()
            ra.ms1_mrl             = get(row, MS1, f_mrl).strip()
            ra.ms1_units           = get(row, MS1, f_units).strip()
            ra.ms1_dilution        = get(row, MS1, f_dil).strip()
            ra.ms1_fortified_level = get(row, MS1, f_fort).strip()
            ra.ms1_pct_rec         = get(row, MS1, f_pctrec).strip()
            ra.ms1_pct_rec_limits  = get(row, MS1, f_pctrecl).strip()

            # MSD fields
            ra.msd_result         = get(row, MSD, f_result).strip()
            ra.msd_units          = get(row, MSD, f_units).strip()
            ra.msd_dilution       = get(row, MSD, f_dil).strip()
            ra.msd_pct_rec        = get(row, MSD, f_msdpctrec).strip()
            ra.msd_pct_rec_limits = get(row, MSD, f_msdpctrecl).strip()
            ra.msd_pct_rpd        = get(row, MSD, f_rpd).strip()
            ra.msd_pct_rpd_limit  = get(row, MSD, f_rpdl).strip()

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (
        f"Reports +{created_reports} / updated ~{updated_reports}. "
        f"Analytes +{created_analytes} / updated ~{updated_analytes}. "
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

    db = SessionLocal()
    reports = db.query(Report).all()
    rows = []
    for r in reports:
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
