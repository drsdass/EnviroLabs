import os
import io
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, ForeignKey, Index
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

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)

    # identity
    lab_id = Column(String, nullable=False, index=True, unique=True)  # ensure one row per Lab ID
    client = Column(String, nullable=False, index=True)

    # meta (client info)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    project_lead = Column(String, nullable=True)
    address = Column(String, nullable=True)

    # Summary/meta
    sample_name = Column(String, nullable=True)
    prepared_by = Column(String, nullable=True)
    matrix = Column(String, nullable=True)
    prepared_date = Column(String, nullable=True)
    qualifiers = Column(String, nullable=True)
    asin = Column(String, nullable=True)
    product_weight_g = Column(String, nullable=True)

    # Dates / misc
    collected_date = Column(Date, nullable=True)   # Received Date
    resulted_date = Column(Date, nullable=True)    # Reported
    acq_datetime = Column(String, nullable=True)
    sheet_name = Column(String, nullable=True)
    pdf_url = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # child analyte rows
    rows = relationship("ResultRow", cascade="all, delete-orphan", back_populates="report")

class ResultRow(Base):
    """
    kind: 'sample' | 'mb' | 'ms1' | 'msd'
    We store analyte row data for Sample Results and QC sections.
    """
    __tablename__ = "result_rows"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"), index=True, nullable=False)

    kind = Column(String, nullable=False)          # sample / mb / ms1 / msd
    analyte = Column(String, nullable=True)
    result = Column(String, nullable=True)
    mrl = Column(String, nullable=True)
    units = Column(String, nullable=True)
    dilution = Column(String, nullable=True)
    analyzed = Column(String, nullable=True)       # only for sample / sometimes blank for QC
    qualifier = Column(String, nullable=True)      # only for sample

    # extra QC fields
    fortified_level = Column(String, nullable=True)    # ms1
    pct_rec = Column(String, nullable=True)            # ms1 / msd
    pct_rec_limits = Column(String, nullable=True)     # ms1 / msd
    pct_rpd = Column(String, nullable=True)            # msd
    pct_rpd_limit = Column(String, nullable=True)      # msd

    report = relationship("Report", back_populates="rows")

Index("ix_rows_report_kind", ResultRow.report_id, ResultRow.kind)

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
PFAS_TARGETS = {
    "pfoa","pfos","pfna","fosaa","n-mefosaa","n-etfosaa","sampap",
    "pfosa","n-mefosa","n-mefose","n-etfosa","n-etfose","disampap"
}
BPS_NAMES = {"bisphenol s", "bps"}  # allow both spellings

def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())

def _lab_id_is_numericish(lab_id: str) -> bool:
    s = (lab_id or "").strip()
    return len(s) > 0 and s[0].isdigit()

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
        db.add(AuditLog(username=username or "-", role=role or "-", action=action, details=details))
        db.commit()
    except Exception:
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

    if u["role"] == "client" and r.client != u["client_name"]:
        db.close()
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    # Split child rows by kind for the template
    sample_rows = [row for row in r.rows if row.kind == "sample"]
    mb_rows     = [row for row in r.rows if row.kind == "mb"]
    ms1_rows    = [row for row in r.rows if row.kind == "ms1"]
    msd_rows    = [row for row in r.rows if row.kind == "msd"]

    db.close()
    return render_template(
        "report_detail.html",
        user=u, r=r,
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

    # Read with no header first so we can detect the real header row
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

    # Find the real header row (look for "Sample ID")
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

    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    # drop empty rows
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].copy()

    msg = _ingest_master_file_grouped(df, u, filename)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

def _ingest_master_file_grouped(df: pd.DataFrame, u, filename: str) -> str:
    """
    Reads your 'Master Upload File' with the two-row header.
    Groups rows by Lab ID, creates/updates ONE Report per Lab ID,
    and saves all analyte/QC rows into ResultRow.
    """
    df = df.fillna("").copy()

    # Exact column captions from your sheet (row 2 actual headers)
    # Left banner section names are ignored; we use the captions below.
    C_LABID   = "Sample ID (Lab ID, Laboratory ID)"
    C_CLIENT  = "Client"
    C_PHONE   = "Phone"
    C_EMAIL   = "Email"
    C_LEAD    = "Project Lead"
    C_ADDR    = "Address"

    C_REPORTED   = "Reported"
    C_RECEIVED   = "Received Date"
    C_NAME       = "Sample Name"
    C_PREP_BY    = "Prepared By"
    C_MATRIX     = "Matrix"
    C_PREP_DATE  = "Prepared Date"
    C_QUALS      = "Qualifiers"
    C_ASIN       = "ASIN (Identifier)"
    C_WEIGHT     = "Product Weight (Grams)"
    C_ACQ        = "Acq. Date-Time"
    C_SHEET      = "SheetName"

    # Blocks (in fixed order in your sheet)
    # Sample Results
    SR = ["Analyte","Result","MRL","Units","Dilution","Analyzed","Qualifier"]
    # Method Blank
    MB = ["Analyte","Result","MRL","Units","Dilution"]
    # Matrix Spike 1
    MS1 = ["Analyte","Result","MRL","Units","Dilution","Fortified Level","%REC","%REC Limits"]
    # Matrix Spike Duplicate
    MSD = ["Analyte","Result","Units","Dilution","%REC","%REC Limits","%RPD","%RPD Limit"]

    # Build indices for the repeating blocks by scanning columns from left to right.
    cols = list(df.columns)

    def find_seq(start_idx: int, seq: List[str]) -> int:
        """Return start index of 'seq' (exact case match) at/after start_idx, else -1."""
        n = len(cols); m=len(seq)
        for i in range(start_idx, n - m + 1):
            if all(cols[i+j].strip().lower() == seq[j].strip().lower() for j in range(m)):
                return i
        return -1

    # Find the first SR sequence, then subsequent blocks must follow.
    sr_start = find_seq(0, SR)
    if sr_start < 0:
        return "Upload parsed, but could not locate the 'Sample Results' block."

    mb_start  = find_seq(sr_start + len(SR), MB)
    ms1_start = find_seq(mb_start + len(MB) if mb_start>=0 else sr_start+len(SR), MS1)
    msd_start = find_seq(ms1_start + len(MS1) if ms1_start>=0 else (mb_start+len(MB) if mb_start>=0 else sr_start+len(SR)), MSD)

    # Group rows by Lab ID
    created = 0
    updated = 0
    skipped_non_numeric = 0

    db = SessionLocal()
    try:
        for lab_id, grp in df.groupby(C_LABID):
            lab_id = str(lab_id).strip()
            if not _lab_id_is_numericish(lab_id):
                skipped_non_numeric += len(grp)
                continue

            # client info & meta — take the first non-empty among the group's rows
            def first_non_empty(col: str) -> str:
                if col not in grp.columns: return ""
                for v in grp[col].tolist():
                    s = str(v).strip()
                    if s: return s
                return ""

            client = first_non_empty(C_CLIENT) or CLIENT_NAME

            # upsert one Report row per Lab ID
            rpt = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not rpt:
                rpt = Report(lab_id=lab_id, client=client)
                db.add(rpt)
                created += 1
            else:
                rpt.client = client
                updated += 1

            rpt.phone = first_non_empty(C_PHONE)
            rpt.email = first_non_empty(C_EMAIL)
            rpt.project_lead = first_non_empty(C_LEAD)
            rpt.address = first_non_empty(C_ADDR)

            rpt.sample_name = first_non_empty(C_NAME) or lab_id
            rpt.prepared_by = first_non_empty(C_PREP_BY)
            rpt.matrix = first_non_empty(C_MATRIX)
            rpt.prepared_date = first_non_empty(C_PREP_DATE)
            rpt.qualifiers = first_non_empty(C_QUALS)
            rpt.asin = first_non_empty(C_ASIN)
            rpt.product_weight_g = first_non_empty(C_WEIGHT)

            rpt.resulted_date = parse_date(first_non_empty(C_REPORTED))
            rpt.collected_date = parse_date(first_non_empty(C_RECEIVED))
            rpt.acq_datetime = first_non_empty(C_ACQ)
            rpt.sheet_name = first_non_empty(C_SHEET)

            # Clear existing child rows and rebuild for this upload (simplest + consistent)
            for old in list(rpt.rows):
                db.delete(old)

            # For each physical row in the group, read Sample Results + QC blocks
            for _, row in grp.iterrows():
                # --- Sample Results ---
                if sr_start >= 0:
                    analyte = str(row.iloc[sr_start + 0]).strip()
                    if analyte:
                        analyte_norm = _norm(analyte)
                        is_bps = analyte_norm in BPS_NAMES
                        is_pfas = analyte_norm in PFAS_TARGETS
                        if is_bps or is_pfas:
                            rr = ResultRow(
                                report=rpt, kind="sample",
                                analyte=analyte,
                                result=str(row.iloc[sr_start + 1]).strip(),
                                mrl=str(row.iloc[sr_start + 2]).strip(),
                                units=str(row.iloc[sr_start + 3]).strip(),
                                dilution=str(row.iloc[sr_start + 4]).strip(),
                                analyzed=str(row.iloc[sr_start + 5]).strip(),
                                qualifier=str(row.iloc[sr_start + 6]).strip(),
                            )
                            db.add(rr)

                # --- Method Blank ---
                if mb_start >= 0:
                    mb_analyte = str(row.iloc[mb_start + 0]).strip()
                    if mb_analyte:
                        mb = ResultRow(
                            report=rpt, kind="mb",
                            analyte=mb_analyte,
                            result=str(row.iloc[mb_start + 1]).strip(),
                            mrl=str(row.iloc[mb_start + 2]).strip(),
                            units=str(row.iloc[mb_start + 3]).strip(),
                            dilution=str(row.iloc[mb_start + 4]).strip(),
                        )
                        db.add(mb)

                # --- Matrix Spike 1 ---
                if ms1_start >= 0:
                    ms1_analyte = str(row.iloc[ms1_start + 0]).strip()
                    if ms1_analyte:
                        ms1 = ResultRow(
                            report=rpt, kind="ms1",
                            analyte=ms1_analyte,
                            result=str(row.iloc[ms1_start + 1]).strip(),
                            mrl=str(row.iloc[ms1_start + 2]).strip(),
                            units=str(row.iloc[ms1_start + 3]).strip(),
                            dilution=str(row.iloc[ms1_start + 4]).strip(),
                            fortified_level=str(row.iloc[ms1_start + 5]).strip(),
                            pct_rec=str(row.iloc[ms1_start + 6]).strip(),
                            pct_rec_limits=str(row.iloc[ms1_start + 7]).strip(),
                        )
                        db.add(ms1)

                # --- Matrix Spike Duplicate ---
                if msd_start >= 0:
                    msd_analyte = str(row.iloc[msd_start + 0]).strip()
                    if msd_analyte:
                        msd = ResultRow(
                            report=rpt, kind="msd",
                            analyte=msd_analyte,
                            result=str(row.iloc[msd_start + 1]).strip(),
                            units=str(row.iloc[msd_start + 2]).strip(),
                            dilution=str(row.iloc[msd_start + 3]).strip(),
                            pct_rec=str(row.iloc[msd_start + 4]).strip(),
                            pct_rec_limits=str(row.iloc[msd_start + 5]).strip(),
                            pct_rpd=str(row.iloc[msd_start + 6]).strip(),
                            pct_rpd_limit=str(row.iloc[msd_start + 7]).strip(),
                        )
                        db.add(msd)

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_non_numeric} non-numeric Lab ID row(s).")

# ----------- Export / Audit / Health -----------
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

    # Flatten to a simple CSV: one line per analyte row for the report
    out = []
    for r in rows:
        for rr in r.rows:
            out.append({
                "Lab ID": r.lab_id,
                "Client": r.client,
                "Kind": rr.kind,
                "Analyte": rr.analyte or "",
                "Result": rr.result or "",
                "MRL": rr.mrl or "",
                "Units": rr.units or "",
                "Dilution": rr.dilution or "",
                "Analyzed": rr.analyzed or "",
                "Qualifier": rr.qualifier or "",
                "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
                "Received": r.collected_date.isoformat() if r.collected_date else "",
            })
    db.close()

    df = pd.DataFrame(out)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(out)} rows")
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
