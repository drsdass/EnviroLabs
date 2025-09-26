import os
import io
import re
from datetime import datetime, date
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

# ------------------- Config -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME     = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = str(os.getenv("KEEP_UPLOADED_CSVS", "true")).lower() == "true"

# Toggle analyte filter (true/false)
ENFORCE_ANALYTE_FILTER = str(os.getenv("ENFORCE_ANALYTE_FILTER", "true")).lower() == "true"
TARGET_ANALYTES = {"bisphenol s", "pfas"}

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
    patient_name = Column(String, nullable=True)  # unused, kept for compatibility
    test = Column(String, nullable=True)          # e.g., "Bisphenol S", "PFAS"
    result = Column(String, nullable=True)        # text or numeric-as-text
    collected_date = Column(Date, nullable=True)  # received date
    resulted_date = Column(Date, nullable=True)   # reported date
    pdf_url = Column(String, nullable=True)
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
    """Best-effort parse to date, tolerant of NaT/None/blank."""
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
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        if hasattr(dt, "to_pydatetime"):
            return dt.to_pydatetime().date()
        if isinstance(dt, datetime):
            return dt.date()
        return None
    except Exception:
        return None

def norm(s: str):
    """Normalize header strings to a list of lowercase 'words' (alnum/space only)."""
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(s)).split()

def header_contains(col: str, tokens):
    """True if all tokens appear (as words) in the normalized column name."""
    words = norm(col)
    return all(tok in words for tok in tokens)

# Strict aliases: we avoid loose 'sample'/'name' guesses for Lab ID.
ALIASES = {
    "lab_id": [
        ["sample", "id"],            # "Sample ID (Lab ID, Laboratory ID)"
        ["lab", "id"],               # "Lab ID"
        ["laboratory", "id"],        # "Laboratory ID"
        ["accession", "id"],
    ],
    "client": [
        ["client"], ["client", "name"], ["account"], ["facility"]
    ],
    "collected_date": [
        ["received", "date"], ["collected", "date"], ["collection", "date"], ["collected"]
    ],
    "resulted_date": [
        ["reported", "date"], ["resulted", "date"], ["finalized"], ["result", "date"]
    ],
    "pdf_url": [
        ["pdf", "url"], ["pdf"], ["report", "link"]
    ],
}

def find_col(df: pd.DataFrame, logical_name: str):
    aliases = ALIASES.get(logical_name, [])
    for tokens in aliases:
        for c in df.columns:
            if header_contains(c, tokens):
                return c
    for tokens in aliases:
        needle = " ".join(tokens)
        for c in df.columns:
            if needle in " ".join(norm(c)):
                return c
    return None

def _smart_read_table(saved_path: str) -> pd.DataFrame:
    """
    Read CSV/XLSX and auto-detect the real header row.
    Handles files where row 0 contains group titles and the actual headers are on row 1..N.
    """
    ext = os.path.splitext(saved_path)[1].lower()

    # Try simple read first
    try:
        if ext in [".xlsx", ".xls"]:
            df0 = pd.read_excel(saved_path, engine="openpyxl")
        else:
            df0 = pd.read_csv(saved_path)
    except Exception:
        df0 = None

    def _has_core(d: pd.DataFrame):
        d.columns = [str(c).strip() for c in d.columns]
        return bool(find_col(d, "lab_id") and find_col(d, "client"))

    if df0 is not None and _has_core(df0):
        return df0

    # Scan for header row
    try:
        if ext in [".xlsx", ".xls"]:
            raw = pd.read_excel(saved_path, engine="openpyxl", header=None, dtype=str)
        else:
            raw = pd.read_csv(saved_path, header=None, dtype=str, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(f"Could not read file: {e}")

    for r in range(min(10, len(raw))):
        header_candidate = [("" if pd.isna(x) else str(x)).strip() for x in list(raw.iloc[r])]
        if all(h == "" or h.lower().startswith("unnamed") for h in header_candidate):
            continue
        df_try = raw.iloc[r+1:].copy()
        df_try.columns = header_candidate
        df_try = df_try.rename(columns=lambda c: " ".join(str(c).split()))
        if _has_core(df_try):
            return df_try

    # Fallbacks
    for r in range(min(5, len(raw))):
        header = [("" if pd.isna(x) else str(x)).strip() for x in list(raw.iloc[r])]
        if any(h for h in header):
            df_fallback = raw.iloc[r+1:].copy()
            df_fallback.columns = header
            return df_fallback.rename(columns=lambda c: " ".join(str(c).split()))

    df_last = raw.copy()
    df_last.columns = [("" if pd.isna(x) else str(x)).strip() for x in list(raw.iloc[0])]
    return df_last.iloc[1:].rename(columns=lambda c: " ".join(str(c).split()))

def _force_true_labid_column(df: pd.DataFrame, current_col: str | None) -> str | None:
    """If chosen Lab ID column doesn't look like an ID header, replace with a header that does."""
    def looks_like_id_header(colname: str) -> bool:
        w = set(norm(colname))
        return ("id" in w) and (("lab" in w) or ("laboratory" in w) or ("sample" in w))

    if current_col and looks_like_id_header(current_col):
        return current_col

    candidates = [c for c in df.columns if looks_like_id_header(c)]
    if candidates:
        candidates.sort(key=lambda x: len(str(x)), reverse=True)
        return candidates[0]
    return current_col

def _normalize_leading(s: str) -> str:
    """Strip BOM/zero-width chars and spaces from the start."""
    if s is None:
        return ""
    return str(s).replace("\ufeff", "").replace("\u200b", "").lstrip()

def _starts_with_digit(s: str) -> bool:
    s2 = _normalize_leading(s)
    return bool(re.match(r"^\d", s2))

def _row_cell(row: pd.Series, col):
    """
    Safe getter that copes with duplicate column labels.
    If pandas returns a Series (duplicate headers), pick the first non-empty / non-'Not Found' value.
    """
    try:
        v = row[col]
    except Exception:
        v = row.get(col, None)
    if isinstance(v, pd.Series):
        for vv in v.values:
            if pd.notna(vv):
                s = str(vv).strip()
                if s and s.lower() != "not found":
                    return vv
        return None
    return v

def _find_sample_results_block(df: pd.DataFrame):
    """
    Find the columns that comprise the SAMPLE RESULTS block:
    Analyte, Result, MRL, Units, Dilution, Analyzed, Qualifier (in that vicinity).
    Returns a dict with keys: analyte, result, mrl, units, dilution, analyzed, qualifier
    or {} if not found.
    """
    cols = list(df.columns)
    ncols = len(cols)

    def next_by_name(start_idx, token):
        for j in range(start_idx + 1, min(start_idx + 8, ncols)):
            if header_contains(cols[j], [token]):
                return cols[j]
        return None

    for i, c in enumerate(cols):
        if header_contains(c, ["analyte"]):
            result   = next_by_name(i, "result")
            mrl      = next_by_name(i, "mrl")
            units    = next_by_name(i, "units")
            dilution = next_by_name(i, "dilution")
            analyzed = next_by_name(i, "analyzed")
            qualifier= next_by_name(i, "qualifier")
            if result and mrl and units and dilution and analyzed and qualifier:
                return {
                    "analyte": c, "result": result, "mrl": mrl,
                    "units": units, "dilution": dilution,
                    "analyzed": analyzed, "qualifier": qualifier
                }
    return {}

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

    # Minimal payload for template (QC behavior unchanged)
    def empty_payload():
        return {
            "client_info": {
                "client": r.client or "",
                "phone": "", "email": "", "project_lead": "", "address": ""
            },
            "sample_summary": {
                "reported": r.resulted_date.isoformat() if r.resulted_date else "",
                "received_date": r.collected_date.isoformat() if r.collected_date else "",
                "sample_name": r.lab_id or "",
                "prepared_by": "", "matrix": "", "prepared_date": "",
                "qualifiers": "", "asin": "", "product_weight_g": ""
            },
            "sample_results": {
                "analyte": r.test or "", "result": r.result or "", "mrl": "", "units": "",
                "dilution": "", "analyzed": "", "qualifier": ""
            },
            "method_blank": {"analyte": "", "result": "", "mrl": "", "units": "", "dilution": ""},
            "matrix_spike_1": {
                "analyte": "", "result": "", "mrl": "", "units": "",
                "dilution": "", "fortified_level": "", "pct_rec": "", "pct_rec_limits": ""
            },
            "matrix_spike_dup": {
                "analyte": "", "result": "", "units": "", "dilution": "",
                "pct_rec": "", "pct_rec_limits": "", "pct_rpd": "", "pct_rpd_limit": ""
            },
            "acq_datetime": "",
            "sheet_name": ""
        }

    p = empty_payload()
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

    # Use smart reader to auto-detect the real header row
    try:
        df = _smart_read_table(saved_path)
    except Exception as e:
        flash(str(e), "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # Find core columns
    c_lab_id = find_col(df, "lab_id")
    c_client = find_col(df, "client")

    # Extra rescue for explicit long header variants
    if not c_lab_id:
        for c in df.columns:
            title = " ".join(norm(c))
            if ("sample" in title and "id" in title) or ("lab" in title and "id" in title) or ("laboratory" in title and "id" in title):
                c_lab_id = c
                break

    if not c_client:
        for c in df.columns:
            if "client" in " ".join(norm(c)):
                c_client = c
                break

    # Force Lab ID to a true *ID* header
    c_lab_id = _force_true_labid_column(df, c_lab_id)

    if not c_lab_id or not c_client:
        preview = ", ".join(df.columns[:20])
        flash("CSV must include Lab ID (aka 'Sample ID') and Client columns (various names accepted). "
              f"Found columns: {preview}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Prefer the "SAMPLE RESULTS" block for analyte/result
    sr = _find_sample_results_block(df)
    c_test      = sr.get("analyte", None)
    c_result    = sr.get("result", None)
    c_collected = find_col(df, "collected_date")  # "Received Date"
    c_resulted  = find_col(df, "resulted_date")   # "Reported"
    c_pdf       = find_col(df, "pdf_url")

    db = SessionLocal()
    created, updated = 0, 0
    skipped_non_numeric = 0
    skipped_non_target  = 0

    try:
        for _, row in df.iterrows():
            raw_lab = _row_cell(row, c_lab_id)
            lab_id = "" if pd.isna(raw_lab) else str(raw_lab)

            if not lab_id or lab_id.strip().lower() == "nan":
                continue

            if not _starts_with_digit(lab_id):
                skipped_non_numeric += 1
                continue

            raw_client = _row_cell(row, c_client)
            client = "" if (raw_client is None or (isinstance(raw_client, float) and pd.isna(raw_client))) else str(raw_client).strip() or CLIENT_NAME

            analyte = ""
            if c_test:
                t = _row_cell(row, c_test)
                analyte = "" if (t is None or (isinstance(t, float) and pd.isna(t))) else str(t).strip().lower()

            if ENFORCE_ANALYTE_FILTER:
                if analyte and (analyte not in TARGET_ANALYTES):
                    skipped_non_target += 1
                    continue

            existing = db.query(Report).filter(Report.lab_id == _normalize_leading(lab_id)).one_or_none()
            if not existing:
                existing = Report(lab_id=_normalize_leading(lab_id), client=client)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

            if c_test:
                tv = _row_cell(row, c_test)
                existing.test = None if (tv is None or (isinstance(tv, float) and pd.isna(tv))) else str(tv)
            if c_result:
                rv = _row_cell(row, c_result)
                existing.result = None if (rv is None or (isinstance(rv, float) and pd.isna(rv))) else str(rv)
            if c_collected:
                existing.collected_date = parse_date(_row_cell(row, c_collected))
            if c_resulted:
                existing.resulted_date = parse_date(_row_cell(row, c_resulted))
            if c_pdf:
                pv = _row_cell(row, c_pdf)
                existing.pdf_url = "" if (pv is None or (isinstance(pv, float) and pd.isna(pv))) else str(pv)

        db.commit()
        flash(
            f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_non_numeric} non-numeric Lab ID row(s) and "
            f"{skipped_non_target} non-target analyte row(s).",
            "success"
        )
        log_action(
            u["username"], u["role"], "upload_csv",
            f"{filename} -> created {created}, updated {updated}, "
            f"skipped_non_numeric={skipped_non_numeric}, skipped_non_target={skipped_non_target}"
        )
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

    data = [{
        "Lab ID": r.lab_id,
        "Client": r.client,
        "Analyte": r.test or "",
        "Result": r.result or "",
        "Collected / Received Date": r.collected_date.isoformat() if r.collected_date else "",
        "Reported / Resulted Date": r.resulted_date.isoformat() if r.resulted_date else "",
        "PDF URL": r.pdf_url or ""
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

# ----------- Health check -----------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

# ----------- Error handlers -----------
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal Server Error"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
