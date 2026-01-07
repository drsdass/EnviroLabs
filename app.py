import os
import io
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text as sql_text
import pandas as pd
import json
from functools import wraps


# ------------------- Config -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = str(os.getenv("KEEP_UPLOADED_CSVS", "true")).lower() == "true"

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# IMPORTANT:
# PFAS_LIST must be defined before PFAS_SET_UPPER is built.
# If you maintain this list elsewhere, you can import it instead.
PFAS_LIST: List[str] = [
    "PFOA", "PFOS", "PFNA", "FOSAA", "N-MeFOSAA", "N-EtFOSAA",
    "SAmPAP", "PFOSA", "N-MeFOSA", "N-MeFOSE", "N-EtFOSA", "N-EtFOSE",
    "diSAmPAP",
]


# ------------------- App -------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Expose `current_app` to Jinja templates (some templates use it).
@app.context_processor
def _inject_current_app():
    return {"current_app": current_app}


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

    # Sample Results (primary)
    test = Column(String, nullable=True)     # analyte (e.g., "Bisphenol S", "PFAS GROUP", ...)
    result = Column(String, nullable=True)   # numeric-as-text or textual

    collected_date = Column(Date, nullable=True)  # "Received Date"
    resulted_date = Column(Date, nullable=True)   # "Reported Date"
    pdf_url = Column(String, nullable=True)       # <-- Sample Analyte Accumulation

    # ---- Optional metadata fields (strings to keep SQLite simple) ----
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

    # QC Accumulation Repurposed Fields
    acq_datetime = Column(String, nullable=True)  # <-- MB Accumulation
    sheet_name = Column(String, nullable=True)    # <-- MS1 Accumulation

    # QC: Matrix Spike 1
    ms1_analyte = Column(String, nullable=True)
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)  # <-- MSD Accumulation (repurposed)

    # QC: Matrix Spike Duplicate
    msd_analyte = Column(String, nullable=True)
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'admin' or 'client'
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)


class ChainOfCustody(Base):
    __tablename__ = "coc_records"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, unique=True, index=True)
    client_name = Column(String)
    sample_name = Column(String)
    asin = Column(String)
    sample_type = Column(String)
    product_link = Column(String, nullable=True)
    matrix = Column(String, nullable=True)
    anticipated_chemical = Column(String, nullable=True)
    expected_delivery_date = Column(String, nullable=True)
    storage_bin_no = Column(String, nullable=True)
    analyzed = Column(String, nullable=True)
    analysis_date = Column(String, nullable=True)
    results_ng_g = Column(String, nullable=True)
    comments = Column(Text, nullable=True)
    sample_condition = Column(String, nullable=True)
    weight_grams = Column(String, nullable=True)
    carrier_name = Column(String, nullable=True)
    tracking_number = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    project_lead = Column(String, nullable=True)
    address = Column(String, nullable=True)
    # Start as Pending until a user checks the sample in.
    status = Column(String, default="Pending")
    location = Column(String, default="Intake")
    # Time/date stamped when a user checks the sample in.
    received_at = Column(DateTime, nullable=True)
    # Who received it (set when the Total Products file is uploaded)
    received_by = Column(String, nullable=True)
    received_by_role = Column(String, nullable=True)
    received_via_file = Column(String, nullable=True)


Base.metadata.create_all(engine)


# --- one-time add columns if the DB was created earlier without the new fields ---
def _ensure_report_columns():
    needed = {
        # Existing Fields
        "phone", "email", "project_lead", "address", "sample_name", "prepared_by",
        "matrix", "prepared_date", "qualifiers", "asin", "product_weight_g",
        "sample_mrl", "sample_units", "sample_dilution", "sample_analyzed", "sample_qualifier",
        "mb_analyte", "mb_result", "mb_mrl", "mb_units", "mb_dilution",
        "ms1_analyte", "ms1_result", "ms1_mrl", "ms1_units", "ms1_dilution",
        "ms1_fortified_level", "ms1_pct_rec", "ms1_pct_rec_limits",
        "msd_analyte", "msd_result", "msd_units", "msd_dilution",
        "msd_pct_rec", "msd_pct_rec_limits", "msd_pct_rpd", "msd_pct_rpd_limit",
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


def _ensure_coc_columns():
    """One-time add columns for coc_records if DB existed before new fields were added."""
    needed = {
        # intake / metadata fields
        "product_link",
        "matrix",
        "anticipated_chemical",
        "expected_delivery_date",
        "storage_bin_no",
        "analyzed",
        "analysis_date",
        "results_ng_g",
        "comments",
        "weight_grams",
        "carrier_name",
        "tracking_number",
        "phone",
        "email",
        "project_lead",
        "address",

        # receipt/audit fields
        "received_by",
        "received_by_role",
        "received_via_file",
        "sample_condition",
    }
    with engine.begin() as conn:
        cols = set()
        for row in conn.execute(sql_text("PRAGMA table_info(coc_records)")):
            cols.add(row[1])
        missing = needed - cols
        for col in sorted(missing):
            conn.execute(sql_text(f"ALTER TABLE coc_records ADD COLUMN {col} TEXT"))


_ensure_coc_columns()

PFAS_SET_UPPER = {a.upper() for a in PFAS_LIST}

# --- CRITICAL STATIC LIST FOR TEMPLATE STABILITY ---
STATIC_ANALYTES_LIST = [
    "PFOA", "PFOS", "PFNA", "FOSAA", "N-MeFOSAA", "N-EtFOSAA",
    "SAmPAP", "PFOSA", "N-MeFOSA", "N-MeFOSE", "N-EtFOSA", "N-EtFOSE",
    "diSAmPAP", "Bisphenol S"
]


def current_user():
    return {
        "username": session.get("username"),
        "role": session.get("role"),
        "client_name": session.get("client_name"),
    }


def require_login(role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("home"))
            if role and session.get("role") != role:
                flash("Unauthorized", "error")
                return redirect(url_for("dashboard"))
            return fn(*args, **kwargs)
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


def parse_datetime(val):
    """Parse common datetime formats from Total Products (e.g., '8/18/25 15:45')."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    # Common formats seen in spreadsheets
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
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


def _find_exact(cols: List[str], name: str) -> Optional[int]:
    name_l = name.strip().lower()
    for i, c in enumerate(cols):
        if c.strip().lower() == name_l:
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


# --- CRITICAL HELPER: Lab ID Normalization ---
def _normalize_lab_id(lab_id: str) -> str:
    """
    Removes common Lab ID suffixes (like ' 2x_nitrogen blow down', ' 0.5ppb', etc.)
    using regular expressions for robust stripping.
    """
    s = (lab_id or "").strip()
    if not s:
        return s

    # Pattern looks for a space followed by numbers, units, or common suffixes.
    pattern = r"\s+[\-\+]?\d*\.?\d+(?:ppb|ppt|ng\/g|ug\/g|\s\d*|\s.*)?$"
    normalized = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
    if not normalized:
        return s
    return normalized


def _target_analyte_ok(analyte: str) -> bool:
    if analyte is None:
        return False
    a = analyte.strip().upper()
    return (a == "BISPHENOL S") or (a in PFAS_SET_UPPER)


def _is_pfas_analyte(analyte: str) -> bool:
    if analyte is None:
        return False
    return analyte.strip().upper() in PFAS_SET_UPPER


# --- NEW HELPER: Generates the HTML table body content ---
def _generate_report_table_html(reports: List[Report]) -> str:
    """
    Generates the raw <tbody> HTML string for the dashboard table.
    """
    html_rows = []

    for r in reports:
        detail_url = url_for("report_detail", report_id=r.id)

        lab_id = r.lab_id or ""
        client = r.client or ""
        sample_name = r.sample_name or ""
        test = r.test or ""
        result = r.result or ""
        sample_units = r.sample_units or ""
        reported_date = r.resulted_date.isoformat() if r.resulted_date else ""

        row = f"""
        <tr>
            <td><a class="link" href="{detail_url}">{lab_id}</a></td>
            <td>{client}</td>
            <td>{sample_name}</td>
            <td>{test}</td>
            <td>{result}</td>
            <td>{sample_units}</td>
            <td>{reported_date}</td>
        </tr>
        """
        html_rows.append(row)

    return "\n".join(html_rows)


# --- CRITICAL HELPER: Retrieves and structures QC data for the template ---
def _get_structured_qc_data(r: Report) -> List[Dict[str, Any]]:
    """
    Parses accumulation strings and reorganizes data into a final list for template use.
    The output guarantees that all STATIC_ANALYTES_LIST names are present and ordered.
    """
    sample_map: Dict[str, Dict[str, str]] = {}
    mb_map: Dict[str, Dict[str, str]] = {}
    ms1_map: Dict[str, Dict[str, str]] = {}
    msd_map: Dict[str, Dict[str, str]] = {}

    # 1. Main Sample Results Parsing (r.pdf_url: Analyte: ResultUnit | ...)
    if r.pdf_url:
        for item in r.pdf_url.split(" | "):
            if ": " in item:
                analyte, result_unit = item.split(": ", 1)
                analyte = analyte.strip()
                sample_map[analyte] = {"sample_result_units": result_unit.strip()}

    # 2. Method Blank Parsing (r.acq_datetime: Analyte|Result|MRL|Units|Dilution | ...)
    if r.acq_datetime:
        for item in r.acq_datetime.split(" | "):
            parts = item.split("|")
            if len(parts) >= 5:
                analyte = parts[0].strip()
                mb_map[analyte] = {
                    "mb_result": parts[1].strip(),
                    "mb_mrl": parts[2].strip(),
                    "mb_units": parts[3].strip(),
                    "mb_dilution": parts[4].strip(),
                }

    # 3. Matrix Spike 1 Parsing (r.sheet_name: Analyte|Result|MRL|Units|Dilution|FortifiedLevel|%REC | ...)
    if r.sheet_name:
        for item in r.sheet_name.split(" | "):
            parts = item.split("|")
            if len(parts) >= 7:
                analyte = parts[0].strip()
                ms1_map[analyte] = {
                    "ms1_result": parts[1].strip(),
                    "ms1_mrl": parts[2].strip(),
                    "ms1_units": parts[3].strip(),
                    "ms1_dilution": parts[4].strip(),
                    "ms1_fortified_level": parts[5].strip(),
                    "ms1_pct_rec": parts[6].strip(),
                }

    # 4. Matrix Spike Duplicate Parsing (r.ms1_pct_rec_limits: Analyte|Result|Units|Dilution|%REC|%REC Limits|%RPD | ...)
    if r.ms1_pct_rec_limits:
        for item in r.ms1_pct_rec_limits.split(" | "):
            parts = item.split("|")
            if len(parts) >= 7:
                analyte = parts[0].strip()
                msd_map[analyte] = {
                    "msd_result": parts[1].strip(),
                    "msd_units": parts[2].strip(),
                    "msd_dilution": parts[3].strip(),
                    "msd_pct_rec": parts[4].strip(),
                    "msd_pct_rec_limits": parts[5].strip(),
                    "msd_pct_rpd": parts[6].strip(),
                }

    final_list: List[Dict[str, Any]] = []

    for analyte_name in STATIC_ANALYTES_LIST:
        data: Dict[str, str] = {}
        data.update(sample_map.get(analyte_name, {}))
        data.update(mb_map.get(analyte_name, {}))
        data.update(ms1_map.get(analyte_name, {}))
        data.update(msd_map.get(analyte_name, {}))

        final_list.append({
            "analyte": analyte_name,

            # Sample Result Fields
            "sample_result": data.get("sample_result_units", ""),
            "sample_mrl": r.sample_mrl or "",
            "sample_units": r.sample_units or "",
            "sample_dilution": r.sample_dilution or "",
            "sample_analyzed": r.sample_analyzed or "",
            "sample_qualifier": r.sample_qualifier or "",

            # MB Fields
            "mb_result": data.get("mb_result", ""),
            "mb_mrl": data.get("mb_mrl", ""),
            "mb_units": data.get("mb_units", ""),
            "mb_dilution": data.get("mb_dilution", ""),

            # MS1 Fields
            "ms1_result": data.get("ms1_result", ""),
            "ms1_mrl": data.get("ms1_mrl", ""),
            "ms1_units": data.get("ms1_units", ""),
            "ms1_dilution": data.get("ms1_dilution", ""),
            "ms1_fortified_level": data.get("ms1_fortified_level", ""),
            "ms1_pct_rec": data.get("ms1_pct_rec", ""),
            "ms1_pct_rec_limits": data.get("msd_pct_rec_limits", ""),

            # MSD Fields
            "msd_result": data.get("msd_result", ""),
            "msd_units": data.get("msd_units", ""),
            "msd_dilution": data.get("msd_dilution", ""),
            "msd_pct_rec": data.get("msd_pct_rec", ""),
            "msd_pct_rec_limits": data.get("msd_pct_rec_limits", ""),
            "msd_pct_rpd": data.get("msd_pct_rpd", ""),
            "msd_pct_rpd_limit": r.msd_pct_rpd_limit or "",
        })

    return final_list


# ------------------- Routes -------------------
@app.route("/")
def home():
    if session.get("username"):
        return redirect(url_for("portal_choice"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    role = request.form.get("role")
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if role == "admin" and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"], session["role"] = username, "admin"
        session["client_name"] = None
        log_action(username, "admin", "login", "Admin logged in")
        return redirect(url_for("portal_choice"))

    if role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"], session["role"], session["client_name"] = username, "client", CLIENT_NAME
        log_action(username, "client", "login", "Client logged in")
        return redirect(url_for("portal_choice"))

    flash("Invalid credentials", "error")
    return redirect(url_for("home"))


@app.route("/portal")
def portal_choice():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    return render_template("portal_choice.html", user=u)


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
    try:
        q = db.query(Report)
        if u["role"] == "client":
            q = q.filter(Report.client == u["client_name"])

        if lab_id:
            normalized_lab_id = _normalize_lab_id(lab_id)
            q = q.filter(Report.lab_id == normalized_lab_id)

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

    finally:
        db.close()

    reports_html = _generate_report_table_html(reports)
    return render_template("dashboard.html", user=u, reports=reports, reports_html=reports_html)


@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    try:
        # SQLAlchemy 1.4+ supports Session.get(Model, id)
        r = db.get(Report, report_id)
    finally:
        db.close()

    if not r:
        flash("Report not found", "error")
        return redirect(url_for("dashboard"))

    if u["role"] == "client" and r.client != u["client_name"]:
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    def val(x):  # noqa: E306
        return "" if x is None else str(x)

    structured_qc_list = _get_structured_qc_data(r)

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
            "prepared_by": val(r.prepared_by),
            "matrix": val(r.matrix),
            "prepared_date": val(r.prepared_date),
            "qualifiers": val(r.qualifiers),
            "asin": val(r.asin),
            "product_weight_g": val(r.product_weight_g),
        },
        "sample_results": {
            "analyte": val(r.test),
            "result_summary": val(r.pdf_url) or "N/A",
            "mrl": val(r.sample_mrl),
            "units": val(r.sample_units),
            "dilution": val(r.sample_dilution),
            "analyzed": val(r.sample_analyzed),
            "qualifier": val(r.sample_qualifier),
        },
        "analyte_list": structured_qc_list,
        "matrix_spike_dup": {
            "pct_rpd_limit": val(r.msd_pct_rpd_limit),
        },
    }

    return render_template("report_detail.html", user=u, r=r, p=p)


# ----------- CSV/Excel upload with robust header detection -----------
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
    last_err = None
    for loader in (
        lambda: pd.read_csv(saved_path, header=None, dtype=str),
        lambda: pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl"),
    ):
        try:
            raw = loader()
            break
        except Exception as e:
            last_err = e

    if raw is None:
        flash(f"Could not read file: {last_err}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    raw = raw.fillna("")

    # --- FIX for two-row header: Explicitly set header_row_idx to 1 (the second row) ---
    header_row_idx = 1

    if len(raw) <= header_row_idx:
        flash("File is too short to contain the required header (row 2) and data.", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    flash("Using Excel row 2 as the column header (to skip the thematic row).", "info")

    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers

    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")
    flash("Header preview: " + ", ".join(df.columns[:12]), "info")

    msg = _ingest_master_upload(df, u, filename)
    flash(
        msg,
        "success" if not msg.lower().startswith("import failed") else "error"
    )

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))


def _ingest_master_upload(df: pd.DataFrame, u, filename: str) -> str:
    """
    Parse the Data Consolidator File layout with new condensed headers.
    Groups all target analytes for a single (Normalized) Lab ID into one Report.
    Analyte results are accumulated into r.pdf_url.
    QC data is accumulated using a PIPE (|) separator for inner fields into r.acq_datetime and r.sheet_name.
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    def find_any_col(names: List[str], fallback_tokens: List[str]) -> Optional[int]:
        for name in names:
            idx = _find_exact(cols, name)
            if idx is not None:
                return idx
        return _find_token_col(cols, *fallback_tokens)

    # ---- CORE SAMPLE / CLIENT MAPPING (Updated for new condensed headers) ----
    idx_lab = find_any_col(["Sample ID", "SampleID"], ["sample", "id"])
    idx_analyte_name = find_any_col(["Name", "Analyte Name"], ["name", "analyte"])
    idx_final_conc = find_any_col(["Final Conc."], ["final", "conc"])
    idx_dilution = find_any_col(["Dil.", "Dilution"], ["dilution"])
    idx_acq_datetime_orig = find_any_col(["Acq. Date-Time"], ["acq", "date"])

    idx_sample_name = find_any_col(["Product Name"], ["product", "name"])
    idx_matrix = find_any_col(["Matrix"], ["matrix"])
    idx_received_by = find_any_col(["Received By"], ["received", "by"])
    idx_asin = find_any_col(["ASIN (Identifier)", "Amazon ID"], ["asin", "identifier"])
    idx_weight = find_any_col(["Weight (Grams)"], ["weight", "g"])
    idx_client = find_any_col(["Client"], ["client"])
    idx_phone = find_any_col(["Phone"], ["phone"])
    idx_email = find_any_col(["Email"], ["email"])
    idx_project_lead = find_any_col(["Project Lead"], ["project", "lead"])
    idx_address = find_any_col(["Address"], ["address"])
    idx_sheet_name_orig = find_any_col(["SheetName"], ["sheetname"])

    # ---- QUALITY CONTROL BLOCK MAPPING (Using new named headers) ----
    idx_mb_analyte = find_any_col(["Analyte (MB)"], ["analyte", "mb"])
    idx_mb_result = find_any_col(["Result (MB)"], ["result", "mb"])
    idx_mb_mrl = find_any_col(["MRL (MB)"], ["mrl", "mb"])
    idx_mb_dilution = find_any_col(["Dilution (MB)"], ["dilution", "mb"])

    idx_ms1_analyte = find_any_col(["Analyte (MS1)"], ["analyte", "ms1"])
    idx_ms1_result = find_any_col(["Result (MS1)"], ["result", "ms1"])
    idx_ms1_fort_level = find_any_col(["Fortified Level (MS1)"], ["fortified", "level", "ms1"])
    idx_ms1_pct_rec = find_any_col(["%REC (MS1)"], ["%rec", "ms1"])

    idx_msd_result = find_any_col(["Result (MSD)"], ["result", "msd"])
    idx_msd_pct_rec = find_any_col(["%REC (MSD)"], ["%rec", "msd"])
    idx_msd_rpd = find_any_col(["%RPD (MSD)"], ["%rpd", "msd"])

    if idx_lab is None or idx_final_conc is None or idx_client is None:
        return "Import failed: Essential columns (Sample ID, Final Conc., Client) not found."

    created = 0
    updated = 0
    skipped_no_sample_name = 0
    skipped_analyte = 0

    db = SessionLocal()
    report_data: Dict[str, Report] = {}

    try:
        for _, row in df.iterrows():
            original_lab_id = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""

            sample_name_value = str(row.iloc[idx_sample_name]).strip() if idx_sample_name is not None else ""
            if not sample_name_value:
                skipped_no_sample_name += 1
                continue

            lab_id = _normalize_lab_id(original_lab_id)
            client = str(row.iloc[idx_client]).strip() if idx_client is not None else CLIENT_NAME

            sr_analyte = str(row.iloc[idx_analyte_name]).strip() if idx_analyte_name is not None else ""
            if not _target_analyte_ok(sr_analyte):
                skipped_analyte += 1
                continue

            is_pfas = _is_pfas_analyte(sr_analyte)
            db_key = lab_id

            existing = report_data.get(db_key)
            if not existing:
                existing = db.query(Report).filter(Report.lab_id == db_key).one_or_none()
                if not existing:
                    test_name = "PFAS GROUP" if is_pfas else sr_analyte
                    existing = Report(lab_id=db_key, client=client, test=test_name)
                    existing.pdf_url = ""
                    existing.sample_name = ""
                    existing.acq_datetime = ""          # MB
                    existing.sheet_name = ""            # MS1
                    existing.ms1_pct_rec_limits = ""    # MSD Accumulation
                    db.add(existing)
                    created += 1
                else:
                    updated += 1

                report_data[db_key] = existing

            r = existing

            # --- General Info and Sample Summary ---
            r.client = client
            r.sample_name = sample_name_value

            if idx_phone is not None:
                r.phone = str(row.iloc[idx_phone]).strip()
            if idx_email is not None:
                r.email = str(row.iloc[idx_email]).strip()
            if idx_project_lead is not None:
                r.project_lead = str(row.iloc[idx_project_lead]).strip()
            if idx_address is not None:
                r.address = str(row.iloc[idx_address]).strip()

            if idx_acq_datetime_orig is not None:
                r.resulted_date = parse_date(row.iloc[idx_acq_datetime_orig])
            r.collected_date = r.resulted_date

            if idx_matrix is not None:
                r.matrix = str(row.iloc[idx_matrix]).strip()
            if idx_asin is not None:
                r.asin = str(row.iloc[idx_asin]).strip()
            if idx_weight is not None:
                r.product_weight_g = str(row.iloc[idx_weight]).strip()

            # --- Sample Results Accumulation ---
            current_result = str(row.iloc[idx_final_conc]).strip() if idx_final_conc is not None else ""
            r.pdf_url = r.pdf_url or ""

            accumulation_string = f"{sr_analyte}: {current_result} {r.sample_units or ''}".strip()

            if r.pdf_url:
                if accumulation_string not in r.pdf_url:
                    r.pdf_url += f" | {accumulation_string}"
            else:
                r.pdf_url = accumulation_string

            if sr_analyte.upper() == "BISPHENOL S":
                r.test = sr_analyte
                r.result = current_result
            elif is_pfas:
                r.test = "PFAS GROUP"
                r.result = "See Details"

            if idx_dilution is not None:
                r.sample_dilution = str(row.iloc[idx_dilution]).strip()

            # --- QC Blocks Accumulation ---
            # Fill MB
            if idx_mb_analyte is not None:
                try:
                    mb_analyte_val = str(row.iloc[idx_mb_analyte]).strip()
                    mb_result_val = str(row.iloc[idx_mb_result]).strip() if idx_mb_result is not None else ""
                    mb_mrl_val = str(row.iloc[idx_mb_mrl]).strip() if idx_mb_mrl is not None else ""
                    mb_dilution_val = str(row.iloc[idx_mb_dilution]).strip() if idx_mb_dilution is not None else ""

                    mb_accumulation_string = (
                        f"{mb_analyte_val}|{mb_result_val}|{mb_mrl_val}|{r.sample_units or ''}|{mb_dilution_val}"
                    )

                    r.acq_datetime = r.acq_datetime or ""
                    temp_mb_acc = [s.strip() for s in r.acq_datetime.split(" | ") if s.strip()]

                    if mb_analyte_val and not any(mb_analyte_val in s for s in temp_mb_acc):
                        if r.acq_datetime:
                            r.acq_datetime += f" | {mb_accumulation_string}"
                        else:
                            r.acq_datetime = mb_accumulation_string

                    if mb_result_val.upper() in ["#VALUE!", "NAN", "NOT FOUND"]:
                        r.mb_result = ""
                    elif mb_result_val:
                        r.mb_result = mb_result_val
                    else:
                        r.mb_result = ""

                    r.mb_analyte = mb_analyte_val
                    r.mb_mrl = mb_mrl_val
                    r.mb_dilution = mb_dilution_val
                except Exception:
                    pass

            # Fill MS1
            if idx_ms1_analyte is not None:
                try:
                    ms1_analyte_val = str(row.iloc[idx_ms1_analyte]).strip()
                    ms1_result_val = str(row.iloc[idx_ms1_result]).strip() if idx_ms1_result is not None else ""
                    ms1_fortified_level_val = str(row.iloc[idx_ms1_fort_level]).strip() if idx_ms1_fort_level is not None else ""
                    ms1_pct_rec_val = str(row.iloc[idx_ms1_pct_rec]).strip() if idx_ms1_pct_rec is not None else ""

                    ms1_mrl_val = str(row.iloc[idx_mb_mrl]).strip() if idx_mb_mrl is not None else ""
                    ms1_dilution_val = str(row.iloc[idx_mb_dilution]).strip() if idx_mb_dilution is not None else ""

                    ms1_accumulation_string = (
                        f"{ms1_analyte_val}|{ms1_result_val}|{ms1_mrl_val}|{r.sample_units or ''}|"
                        f"{ms1_dilution_val}|{ms1_fortified_level_val}|{ms1_pct_rec_val}"
                    )

                    r.sheet_name = r.sheet_name or ""
                    temp_ms1_acc = [s.strip() for s in r.sheet_name.split(" | ") if s.strip()]

                    if ms1_analyte_val and not any(ms1_analyte_val in s for s in temp_ms1_acc):
                        if r.sheet_name:
                            r.sheet_name += f" | {ms1_accumulation_string}"
                        else:
                            r.sheet_name = ms1_accumulation_string

                    r.ms1_analyte = ms1_analyte_val
                    r.ms1_result = ms1_result_val
                    r.ms1_mrl = ms1_mrl_val
                    r.ms1_units = r.sample_units
                    r.ms1_dilution = ms1_dilution_val
                    r.ms1_fortified_level = ms1_fortified_level_val
                    r.ms1_pct_rec = ms1_pct_rec_val
                except Exception:
                    pass

            # Fill MSD (Matrix Spike Duplicate)
            if idx_msd_result is not None:
                try:
                    msd_analyte_val = sr_analyte
                    msd_result_val = str(row.iloc[idx_msd_result]).strip()
                    msd_pct_rec_val = str(row.iloc[idx_msd_pct_rec]).strip() if idx_msd_pct_rec is not None else ""
                    msd_rpd_val = str(row.iloc[idx_msd_rpd]).strip() if idx_msd_rpd is not None else ""

                    msd_units_val = r.ms1_units
                    msd_dilution_val = r.ms1_dilution
                    msd_pct_rec_limits_val = r.ms1_pct_rec_limits

                    msd_accumulation_string = (
                        f"{msd_analyte_val}|{msd_result_val}|{msd_units_val}|{msd_dilution_val}|"
                        f"{msd_pct_rec_val}|{msd_pct_rec_limits_val}|{msd_rpd_val}"
                    )

                    r.ms1_pct_rec_limits = r.ms1_pct_rec_limits or ""
                    temp_msd_acc = [s.strip() for s in r.ms1_pct_rec_limits.split(" | ") if s.strip()]

                    if msd_analyte_val and not any(msd_analyte_val in s for s in temp_msd_acc):
                        if r.ms1_pct_rec_limits:
                            r.ms1_pct_rec_limits += f" | {msd_accumulation_string}"
                        else:
                            r.ms1_pct_rec_limits = msd_accumulation_string

                    r.msd_analyte = msd_analyte_val
                    r.msd_result = msd_result_val
                    r.msd_pct_rec = msd_pct_rec_val
                    r.msd_pct_rpd = msd_rpd_val
                    r.msd_units = msd_units_val
                    r.msd_dilution = msd_dilution_val
                    r.msd_pct_rec_limits = msd_pct_rec_limits_val
                except Exception:
                    pass

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: Critical database error. Details: {e}"
    finally:
        db.close()

    return (
        f"Imported {created} new and updated {updated} report(s). "
        f"Skipped {skipped_no_sample_name} row(s) with missing Sample Name and "
        f"{skipped_analyte} non-target analyte row(s)."
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
    try:
        rows = db.query(AuditLog).order_by(AuditLog.at.desc()).limit(500).all()
    finally:
        db.close()

    return render_template("audit.html", user=u, rows=rows)


@app.route("/export_csv")
def export_csv():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    try:
        q = db.query(Report)
        if u["role"] == "client":
            q = q.filter(Report.client == u["client_name"])
        rows = q.all()
    finally:
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
        "Analyte Details": r.pdf_url or "",
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


# ----------- Chain of Custody Routes (MOVED ABOVE app.run) -----------

def _detect_header_row(raw: pd.DataFrame, required_tokens: List[str], max_rows: int = 6) -> int:
    """Given a dataframe read with header=None, guess which row is the header."""
    req = [t.lower() for t in required_tokens]
    max_check = min(len(raw), max_rows)
    best_idx = 0
    best_score = -1
    for i in range(max_check):
        row_vals = [str(x) for x in raw.iloc[i].values]
        normed = _norm(" ".join(row_vals))
        score = sum(1 for t in req if t in normed)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _ingest_total_products_for_coc(df: pd.DataFrame, u: Dict[str, Any], filename: str) -> str:
    """Create/Update coc_records from a Total Products file.

    IMPORTANT: This import does **not** automatically mark samples as Received.
    Receiving is handled by user selection on the COC page (bulk Check In).
    """
    df = df.fillna("").copy()
    cols = list(df.columns)

    def find_any_col(names: List[str], fallback_tokens: List[str]) -> Optional[int]:
        for name in names:
            idx = _find_exact(cols, name)
            if idx is not None:
                return idx
        return _find_token_col(cols, *fallback_tokens)

    idx_lab = find_any_col(["Laboratory ID", "Lab ID", "LabID", "Sample ID", "SampleID"], ["laboratory", "id"])
    idx_client = find_any_col(["Client", "Client Name"], ["client"])
    idx_sample_name = find_any_col(["Product Name", "Sample Name", "Name"], ["product", "name"])
    idx_product_link = find_any_col(["Link to Product"], ["link", "product"])
    idx_matrix = find_any_col(["Matrix"], ["matrix"])
    idx_anticipated = find_any_col(["Anticipated Chemical"], ["anticipated", "chemical"])
    idx_expected_delivery = find_any_col(["Expected Delivery Date"], ["expected", "delivery"])
    idx_received_on = find_any_col(["Received On"], ["received", "on"])
    idx_received_by_file = find_any_col(["Received By"], ["received", "by"])
    idx_storage_bin = find_any_col(["Storage Bin No", "Storage Bin"], ["storage", "bin"])
    idx_analyzed = find_any_col(["Analyzed?", "Analyzed"], ["analyzed"])
    idx_analysis_date = find_any_col(["Analysis Date"], ["analysis", "date"])
    idx_results = find_any_col(["Results ng/g", "Results"], ["results"])
    idx_comments = find_any_col(["Comments"], ["comments"])
    idx_sample_condition = find_any_col(["Sample Condition", "SAMPLE CONDITION", "Condition"], ["sample", "condition"])
    idx_asin = find_any_col(["ASIN (Identifier)", "ASIN", "Amazon ID"], ["asin"])
    idx_weight = find_any_col(["Weight (Grams)"], ["weight", "grams"])
    idx_carrier = find_any_col(["Carrier Name", "Carrier"], ["carrier", "name"])
    idx_tracking = find_any_col(["Tracking Number", "Tracking Number"], ["tracking", "number"])
    idx_phone = find_any_col(["Phone"], ["phone"])
    idx_email = find_any_col(["Email"], ["email"])
    idx_project_lead = find_any_col(["Project Lead"], ["project", "lead"])
    idx_address = find_any_col(["Address"], ["address"])

    if idx_lab is None:
        return "COC import failed: Could not find a Sample ID / Lab ID column in the Total Products file."

    created = 0
    updated = 0
    skipped = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            raw_lab = str(row.iloc[idx_lab]).strip() if idx_lab is not None else ""
            if not raw_lab:
                skipped += 1
                continue

            lab_id = _normalize_lab_id(raw_lab)

            # Skip placeholder / invalid Lab IDs that would violate UNIQUE constraints
            if (not lab_id) or (lab_id.strip().upper() in {"X", "NA", "N/A", "TBD"}) or (len(lab_id.strip()) < 3):
                skipped += 1
                continue

            def get(idx: Optional[int]) -> str:
                if idx is None:
                    return ""
                return str(row.iloc[idx]).strip()

            client_name = get(idx_client) or (u.get("client_name") if u.get("role") == "client" else CLIENT_NAME)
            sample_name = get(idx_sample_name)
            asin = get(idx_asin)
            matrix = get(idx_matrix)
            anticipated_chemical = get(idx_anticipated)
            expected_delivery_date = get(idx_expected_delivery)
            product_link = get(idx_product_link)
            storage_bin_no = get(idx_storage_bin)
            analyzed = get(idx_analyzed)
            analysis_date = get(idx_analysis_date)
            results_ng_g = get(idx_results)
            comments = get(idx_comments)
            sample_condition = get(idx_sample_condition)
            weight_grams = get(idx_weight)
            carrier_name = get(idx_carrier)
            tracking_number = get(idx_tracking)
            phone = get(idx_phone)
            email = get(idx_email)
            project_lead = get(idx_project_lead)
            address = get(idx_address)

            received_on_raw = get(idx_received_on)
            received_by_raw = get(idx_received_by_file)

            rec = db.query(ChainOfCustody).filter(ChainOfCustody.lab_id == lab_id).one_or_none()
            is_new = False
            if not rec:
                rec = ChainOfCustody(lab_id=lab_id)
                db.add(rec)
                created += 1
                is_new = True
            else:
                updated += 1

            # Always update the latest metadata from the Total Products file
            rec.client_name = client_name
            rec.sample_name = sample_name
            rec.asin = asin
            rec.matrix = matrix
            rec.sample_type = rec.sample_type or matrix
            rec.sample_condition = sample_condition
            rec.product_link = product_link
            rec.anticipated_chemical = anticipated_chemical
            rec.expected_delivery_date = expected_delivery_date
            rec.storage_bin_no = storage_bin_no
            rec.analyzed = analyzed
            rec.analysis_date = analysis_date
            rec.results_ng_g = results_ng_g
            rec.comments = comments
            rec.weight_grams = weight_grams
            rec.carrier_name = carrier_name
            rec.tracking_number = tracking_number
            rec.phone = phone
            rec.email = email
            rec.project_lead = project_lead
            rec.address = address

            # If the file already contains receiving info, keep it if we don't already have a check-in.
            if (not rec.received_at) and received_on_raw:
                parsed = parse_datetime(received_on_raw)
                if parsed:
                    rec.received_at = parsed
            if (not rec.received_by) and received_by_raw:
                rec.received_by = received_by_raw

            # If this is a brand-new record, initialize it as NOT received yet.
            if is_new:
                rec.status = rec.status or "Pending"
                rec.location = rec.location or "Intake"
                rec.received_at = None
                rec.received_by = None
                rec.received_by_role = None

            # Track the last file used to import/refresh metadata (useful for traceability)
            rec.received_via_file = filename

        db.commit()

        log_action(
            u.get("username"),
            u.get("role"),
            "COC_TOTAL_PRODUCTS_UPLOAD",
            f"Uploaded Total Products file '{filename}'. Upserted {created} created / {updated} updated (skipped {skipped}) samples.",
        )
    except Exception as e:
        db.rollback()
        return f"COC import failed: {e}"
    finally:
        db.close()

    return f"COC Intake complete: {created} created, {updated} updated, {skipped} skipped (missing Lab ID)."


@app.route("/coc/upload", methods=["GET", "POST"])
def coc_upload():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    # The upload widget may be embedded directly on the /coc page. If a user
    # navigates to /coc/upload directly and you don't have a dedicated template,
    # just send them back to the COC list.
    if request.method == "GET":
        try:
            return render_template("coc_upload.html", user=u)
        except Exception:
            return redirect(url_for("coc_list"))

    # Support either input name: "total_products_file" (preferred) or "file".
    f = request.files.get("total_products_file") or request.files.get("file")
    if not f or f.filename.strip() == "":
        flash("No file uploaded", "error")
        return redirect(url_for("coc_list"))

    filename = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(saved_path)

    raw = None
    last_err = None
    for loader in (
        lambda: pd.read_csv(saved_path, header=None, dtype=str),
        lambda: pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl"),
    ):
        try:
            raw = loader()
            break
        except Exception as e:
            last_err = e

    if raw is None:
        flash(f"Could not read file: {last_err}", "error")
        if os.path.exists(saved_path) and not KEEP_UPLOADED_CSVS:
            os.remove(saved_path)
        return redirect(url_for("coc_list"))

    raw = raw.fillna("")
    header_row_idx = _detect_header_row(raw, required_tokens=["sample", "id"], max_rows=8)

    if len(raw) <= header_row_idx:
        flash("File is too short to contain a header row.", "error")
        if os.path.exists(saved_path) and not KEEP_UPLOADED_CSVS:
            os.remove(saved_path)
        return redirect(url_for("coc_list"))

    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")

    msg = _ingest_total_products_for_coc(df, u, filename)
    flash(msg, "success" if not msg.lower().startswith("coc import failed") else "error")

    if os.path.exists(saved_path) and not KEEP_UPLOADED_CSVS:
        os.remove(saved_path)

    return redirect(url_for("coc_list"))


@app.route("/coc/import", methods=["POST"])
def coc_import_total_products():
    """Import/refresh COC rows from a Total Products file (does NOT mark Received)."""
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    if u["role"] != "admin":
        flash("Only admins can import Total Products into Chain of Custody.", "error")
        return redirect(url_for("coc_list"))

    # Support either input name: `total_products_file` (recommended) or `file` (legacy)
    f = request.files.get("total_products_file") or request.files.get("file")
    if not f or f.filename.strip() == "":
        flash("No file uploaded", "error")
        return redirect(url_for("coc_list"))

    filename = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(saved_path)

    raw = None
    last_err = None
    for loader in (
        lambda: pd.read_csv(saved_path, header=None, dtype=str),
        lambda: pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl"),
    ):
        try:
            raw = loader()
            break
        except Exception as e:
            last_err = e

    if raw is None:
        flash(f"Could not read file: {last_err}", "error")
        if os.path.exists(saved_path) and not KEEP_UPLOADED_CSVS:
            os.remove(saved_path)
        return redirect(url_for("coc_list"))

    raw = raw.fillna("")
    header_row_idx = _detect_header_row(raw, required_tokens=["sample", "id"], max_rows=8)

    if len(raw) <= header_row_idx:
        flash("File is too short to contain a header row.", "error")
        if os.path.exists(saved_path) and not KEEP_UPLOADED_CSVS:
            os.remove(saved_path)
        return redirect(url_for("coc_list"))

    headers = [str(x).strip() for x in raw.iloc[header_row_idx].values]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")

    msg = _ingest_total_products_for_coc(df, u, filename)
    flash(msg, "success" if not msg.lower().startswith("coc import failed") else "error")

    if os.path.exists(saved_path) and not KEEP_UPLOADED_CSVS:
        os.remove(saved_path)

    return redirect(url_for("coc_list"))


@app.route("/coc/receive", methods=["POST"])
def coc_receive_selected():
    """Bulk check-in/receive selected samples."""
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    selected_ids = request.form.getlist("selected_ids")
    if not selected_ids:
        flash("No samples selected.", "error")
        return redirect(url_for("coc_list"))

    # Admin can bulk-set status/location; clients can only mark Received.
    if u["role"] == "admin":
        bulk_status = (request.form.get("bulk_status") or "Received").strip()
        bulk_location = (request.form.get("bulk_location") or "Intake").strip()
    else:
        bulk_status = "Received"
        bulk_location = "Intake"

    now = datetime.utcnow()
    updated = 0
    already_received = 0
    not_found = 0

    db = SessionLocal()
    try:
        for rid in selected_ids:
            try:
                rec_id = int(rid)
            except Exception:
                not_found += 1
                continue

            rec = db.get(ChainOfCustody, rec_id)
            if not rec:
                not_found += 1
                continue

            # Enforce client visibility
            if u["role"] != "admin" and rec.client_name != u.get("client_name"):
                continue

            # Client users: don't overwrite an existing receive stamp
            if u["role"] != "admin" and rec.received_at is not None:
                already_received += 1
                continue

            rec.status = bulk_status
            rec.location = bulk_location or rec.location

            if bulk_status.lower() == "received":
                rec.received_at = rec.received_at or now
                rec.received_by = rec.received_by or u.get("username")
                rec.received_by_role = rec.received_by_role or u.get("role")

            updated += 1

        db.commit()

        log_action(
            u.get("username"),
            u.get("role"),
            "COC_BULK_UPDATE",
            f"Bulk update: status='{bulk_status}', location='{bulk_location}'. Updated={updated}, already_received={already_received}, not_found={not_found}",
        )
    except Exception as e:
        db.rollback()
        flash(f"Bulk check-in failed: {e}", "error")
        return redirect(url_for("coc_list"))
    finally:
        db.close()

    if u["role"] == "admin":
        flash(f"Updated {updated} sample(s). Not found: {not_found}.", "success")
    else:
        flash(f"Received {updated} sample(s). Already received (skipped): {already_received}.", "success")

    return redirect(url_for("coc_list"))


def _build_coc_pdf(records: List[ChainOfCustody], title: str = "Chain of Custody") -> io.BytesIO:
    """Create a simple PDF for download/printing."""
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=18,
        rightMargin=18,
        topMargin=18,
        bottomMargin=18,
    )
    styles = getSampleStyleSheet()

    data = [[
        "SAMPLE I.D. / NAME",
        "LAB ID#",
        "ASIN (Identifier)",
        "SAMPLE TYPE",
        "SHIPPED BY",
        "SAMPLE CONDITION",
        "TEST(S) REQUESTED",
        "Status",
    ]]

    for r in records:
        data.append([
            r.sample_name or "",
            r.lab_id or "",
            r.asin or "",
            (r.sample_type or r.matrix or ""),
            r.carrier_name or "",
            r.sample_condition or "",
            r.anticipated_chemical or "",
            r.status or "",
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 12),
        table,
    ]
    doc.build(story)
    buf.seek(0)
    return buf




# ----------- Individual Chain of Custody (per-sample) Print/PDF -----------
@app.route("/coc/<int:record_id>/print")
def coc_record_print(record_id):
    u = current_user()
    if not u.get("username"):
        return redirect(url_for("home"))

    db = SessionLocal()
    try:
        rec = db.get(ChainOfCustody, record_id)
    finally:
        db.close()

    if not rec:
        flash("COC record not found", "error")
        return redirect(url_for("coc_list"))

    if u.get("role") != "admin" and rec.client_name != u.get("client_name"):
        flash("Unauthorized", "error")
        return redirect(url_for("coc_list"))

    # A simple, print-friendly single-record page
    return render_template(
        "coc_record_print.html",
        record=rec,
        user=u,
        now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/coc/<int:record_id>/pdf")
def coc_record_pdf(record_id):
    u = current_user()
    if not u.get("username"):
        return redirect(url_for("home"))

    try:
        import reportlab  # noqa: F401
    except Exception:
        flash("PDF export requires reportlab. Add: reportlab==4.2.5 to requirements.txt and redeploy.", "error")
        return redirect(url_for("coc_list"))

    db = SessionLocal()
    try:
        rec = db.get(ChainOfCustody, record_id)
    finally:
        db.close()

    if not rec:
        flash("COC record not found", "error")
        return redirect(url_for("coc_list"))

    if u.get("role") != "admin" and rec.client_name != u.get("client_name"):
        flash("Unauthorized", "error")
        return redirect(url_for("coc_list"))

    title = f"Chain of Custody - {rec.lab_id or rec.id}"
    pdf = _build_coc_pdf([rec], title=title)
    log_action(u.get("username"), u.get("role"), "COC_RECORD_PDF", f"Exported COC record {rec.id} ({rec.lab_id})")

    safe_lab = (rec.lab_id or str(rec.id)).replace("/", "-").replace("", "-")
    return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name=f"coc_{safe_lab}.pdf")

@app.route("/coc/export_pdf")
def coc_export_pdf():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    try:
        q = db.query(ChainOfCustody)
        if u["role"] != "admin":
            q = q.filter_by(client_name=u.get("client_name"))
        records = q.order_by(ChainOfCustody.received_at.desc().nullslast(), ChainOfCustody.id.desc()).all()
    except Exception:
        # old sqlite fallback
        records = q.order_by(ChainOfCustody.received_at.desc(), ChainOfCustody.id.desc()).all()
    finally:
        db.close()

    pdf = _build_coc_pdf(records, title="Chain of Custody")
    log_action(u.get("username"), u.get("role"), "COC_EXPORT_PDF", f"Exported {len(records)} COC record(s) to PDF")
    return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name="chain_of_custody.pdf")


@app.route("/coc/<int:record_id>/print")
def coc_print_single(record_id: int):
    """
    Print view for a SINGLE Chain of Custody record.
    """
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    db = SessionLocal()
    try:
        rec = db.get(ChainOfCustody, record_id)
    finally:
        db.close()

    if not rec:
        flash("COC record not found.", "error")
        return redirect(url_for("coc_list"))

    if u.get("role") != "admin" and rec.client_name != u.get("client_name"):
        flash("Unauthorized", "error")
        return redirect(url_for("coc_list"))

    return render_template(
        "coc_print.html",
        records=[rec],
        user=u,
        now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        single=True,
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
