import os
import io
from typing import Optional, List, Dict
from datetime import datetime, date

from flask import (
    Flask, render_template, request, redirect, url_for,
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

    # optional legacy field
    patient_name = Column(String, nullable=True)

    # primary fields
    test = Column(String, nullable=True)    # analyte (e.g., "Bisphenol S", "PFOA", ...)
    result = Column(String, nullable=True)

    collected_date = Column(Date, nullable=True)  # "Received Date"
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
        have = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(reports)"))}
        for col in sorted(needed - have):
            conn.execute(sql_text(f"ALTER TABLE reports ADD COLUMN {col} TEXT"))
_ensure_report_columns()

# =========================
# Analyte helpers
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

def _first_nonempty(*vals) -> str:
    for v in vals:
        if v is None: continue
        s = str(v).strip()
        if s and s.lower() not in {"nan","none"}:
            return s
    return ""

# =========================
# Auth & basic pages
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
        return redirect(url_for("dashboard"))
    if role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"] = username
        session["role"] = "client"
        session["client_name"] = CLIENT_NAME
        return redirect(url_for("dashboard"))
    flash("Invalid credentials", "error")
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# =========================
# Dashboard
# =========================
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("home"))
    urole = session.get("role")
    cname = session.get("client_name")

    lab_id = request.args.get("lab_id","").strip()
    start  = request.args.get("start","").strip()
    end    = request.args.get("end","").strip()

    db = SessionLocal()
    q = db.query(Report)
    if urole == "client":
        q = q.filter(Report.client == cname)
    if lab_id:
        q = q.filter(Report.lab_id == lab_id)
    if start:
        sd = parse_date(start);  q = q.filter(Report.resulted_date >= sd) if sd else q
    if end:
        ed = parse_date(end);    q = q.filter(Report.resulted_date <= ed) if ed else q

    try:
        rows = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).limit(1000).all()
    except Exception:
        rows = q.order_by(Report.resulted_date.desc(), Report.id.desc()).limit(1000).all()
    db.close()
    return render_template("dashboard.html", user={"role":urole, "client_name":cname}, reports=rows)

# =========================
# Report detail – bundle PFAS
# =========================
@app.route("/report/<int:report_id>")
def report_detail(report_id):
    if "username" not in session:
        return redirect(url_for("home"))
    urole = session.get("role"); cname = session.get("client_name")

    db = SessionLocal()
    base = db.query(Report).get(report_id)
    if not base:
        db.close(); flash("Report not found", "error"); return redirect(url_for("dashboard"))
    if urole == "client" and base.client != cname:
        db.close(); flash("Unauthorized", "error"); return redirect(url_for("dashboard"))

    rows = db.query(Report).filter(Report.lab_id == base.lab_id).all()
    db.close()

    has_bps = any(_is_bps(r.test) for r in rows)

    if has_bps:
        mode = "BPS"
        cand = [r for r in rows if _is_bps(r.test)]
        cand.sort(key=lambda r: (str(r.result or "") == "", r.id))
        r = cand[0]

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

        base.sample_name = _first_nonempty(base.sample_name, base.lab_id)
    else:
        mode = "PFAS"
        groups: Dict[str, List[Report]] = {}
        for r in rows:
            if _is_internal_standard(r.test):
                continue
            k = _pfas_key(r.test)
            if not k:  # ignore non-PFAS analytes on PFAS panel
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

        # borrow summary fields from any PFAS row
        if chosen:
            pick = list(chosen.values())[0]
        else:
            pick = rows[0]
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
        user={"role": urole, "client_name": cname}, r=base,
        sample_rows=sample_rows, mb_rows=mb_rows, ms1_rows=ms1_rows, msd_rows=msd_rows,
        mode=mode
    )

# =========================
# Upload (very tolerant)
# =========================
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "username" not in session:
        return redirect(url_for("home"))
    if session.get("role") != "admin":
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

    # Always start headerless, then detect header row
    raw = None; last_err = None
    readers = [
        lambda: pd.read_csv(saved_path, header=None, dtype=str, encoding_errors="ignore"),
        lambda: pd.read_excel(saved_path, header=None, dtype=str, engine="openpyxl"),
    ]
    for r in readers:
        try:
            raw = r().fillna("")
            break
        except Exception as e:
            last_err = e
    if raw is None:
        flash(f"Could not read file: {last_err}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # --- Find header row robustly ---
    def find_header_row(data: pd.DataFrame) -> Optional[int]:
        limit = min(25, len(data))
        for i in range(limit):
            vals = [str(x) for x in data.iloc[i].values]
            line = " | ".join(vals)
            norm = _norm(line)
            if "sample" in norm and "id" in norm:
                return i
        return None

    header_row = find_header_row(raw)
    if header_row is None:
        # Your file: banner row then the real headers at row 1 (0-indexed)
        header_row = 1 if len(raw) > 1 else 0
        flash(f"Header row not found by scan; forcing row {header_row} as header.", "info")
    else:
        flash(f"Detected header at row {header_row}.", "info")

    headers = [str(x).strip() for x in raw.iloc[header_row].values]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].fillna("")

    msg = _ingest_master_upload(df, session.get("client_name") or CLIENT_NAME, filename, debug=True)
    flash(msg, "success")

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)
    return redirect(url_for("dashboard"))

def _ingest_master_upload(df: pd.DataFrame, default_client: str, filename: str, debug: bool=False) -> str:
    cols = list(df.columns)

    # exact captions you shared + token fallback
    EXACT = {
        "lab": "Sample ID (Lab ID, Laboratory ID)",
        "client": "Client",
        "phone": "Phone",
        "email": "Email",
        "plead": "Project Lead",
        "addr": "Address",
        "reported": "Reported",
        "received": "Received Date",
        "sname": "Sample Name",
        "prepby": "Prepared By",
        "matrix": "Matrix",
        "prepdate": "Prepared Date",
        "quals": "Qualifiers",
        "asin": "ASIN (Identifier)",
        "pweight": "Product Weight (Grams)",
        "acq": "Acq. Date-Time",
        "sheet": "SheetName",
    }

    def pick(name_key, *tokens):
        exact = EXACT.get(name_key)
        if exact in cols:
            return exact
        # token fallback
        want = [t.lower() for t in tokens]
        for c in cols:
            if all(t in _norm(c) for t in want):
                return c
        return None

    col_lab   = pick("lab", "sample","id")
    col_client= pick("client","client")
    col_phone = pick("phone","phone")
    col_email = pick("email","email")
    col_pj    = pick("plead","project","lead")
    col_addr  = pick("addr","address")
    col_rep   = pick("reported","reported")
    col_recv  = pick("received","received")
    col_sname = pick("sname","sample","name")
    col_pby   = pick("prepby","prepared","by")
    col_matrix= pick("matrix","matrix")
    col_pdt   = pick("prepdate","prepared","date")
    col_qual  = pick("quals","qualifiers")
    col_asin  = pick("asin","asin","identifier")
    col_wt    = pick("pweight","product","weight")
    col_acq   = pick("acq","acq","date")
    col_sheet = pick("sheet","sheet","name")

    # locate the four analyte blocks by name
    ncols = [_norm(c) for c in cols]
    idxs = [i for i, n in enumerate(ncols) if n == "analyte"]

    def looks_like(i, names):
        sl = ncols[i:i+len(names)]
        return len(sl) == len(names) and all(a == b for a,b in zip(sl, names))

    SR  = MB = MS1 = MSD = None
    sr_names  = ["analyte","result","mrl","units","dilution","analyzed","qualifier"]
    mb_names  = ["analyte","result","mrl","units","dilution"]
    ms1_names = ["analyte","result","mrl","units","dilution","fortified level","%rec","%rec limits"]
    msd_names = ["analyte","result","units","dilution","%rec","%rec limits","%rpd","%rpd limit"]

    for i in idxs:
        if SR is None and looks_like(i, sr_names):   SR  = cols[i:i+7];  continue
        if SR is not None and MB is None and looks_like(i, mb_names): MB = cols[i:i+5]; continue
        if MB is not None and MS1 is None and looks_like(i, ms1_names): MS1 = cols[i:i+8]; continue
        if MS1 is not None and MSD is None and looks_like(i, msd_names): MSD = cols[i:i+8]; continue

    # safe fallbacks if not perfectly found
    if SR is None:
        # try to find *any* "Analyte" and take the next six columns
        if idxs:
            s = idxs[0]; SR = cols[s:s+7]
    if MB is None and SR:
        s = cols.index(SR[0]) + 7
        MB = cols[s:s+5]
    if MS1 is None and MB:
        s = cols.index(MB[0]) + 5
        MS1 = cols[s:s+8]
    if MSD is None and MS1:
        s = cols.index(MS1[0]) + 8
        MSD = cols[s:s+8]

    flash(f"Blocks -> SR:{SR and SR[0]} MB:{MB and MB[0]} MS1:{MS1 and MS1[0]} MSD:{MSD and MSD[0]}", "info")

    created = updated = skipped = 0
    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id = str(row.get(col_lab, "")).strip()
            if not _lab_id_is_numericish(lab_id):
                skipped += 1
                continue

            client = str(row.get(col_client, default_client)).strip() or default_client

            # Sample Results analyte
            sr_analyte = str(row.get(SR[0], "")).strip() if SR else ""
            if _is_internal_standard(sr_analyte):
                continue

            # Only keep Bisphenol S & PFAS analytes
            is_bps = _is_bps(sr_analyte)
            is_pfas = _pfas_key(sr_analyte) is not None
            if not (is_bps or is_pfas):
                continue

            # upsert per (lab_id, analyte)
            rec = db.query(Report).filter(
                Report.lab_id == lab_id,
                Report.test == (sr_analyte or "")
            ).one_or_none()
            if not rec:
                rec = Report(lab_id=lab_id, client=client, test=sr_analyte)
                db.add(rec); created += 1
            else:
                rec.client = client; updated += 1

            # meta
            rec.sample_name      = str(row.get(col_sname, lab_id)).strip()
            rec.phone            = str(row.get(col_phone, "")).strip() if col_phone else ""
            rec.email            = str(row.get(col_email, "")).strip() if col_email else ""
            rec.project_lead     = str(row.get(col_pj, "")).strip() if col_pj else ""
            rec.address          = str(row.get(col_addr, "")).strip() if col_addr else ""
            rec.resulted_date    = parse_date(row.get(col_rep)) if col_rep else None
            rec.collected_date   = parse_date(row.get(col_recv)) if col_recv else None
            rec.prepared_by      = str(row.get(col_pby, "")).strip() if col_pby else ""
            rec.matrix           = str(row.get(col_matrix, "")).strip() if col_matrix else ""
            rec.prepared_date    = str(row.get(col_pdt, "")).strip() if col_pdt else ""
            rec.qualifiers       = str(row.get(col_qual, "")).strip() if col_qual else ""
            rec.asin             = str(row.get(col_asin, "")).strip() if col_asin else ""
            rec.product_weight_g = str(row.get(col_wt, "")).strip() if col_wt else ""
            rec.acq_datetime     = str(row.get(col_acq, "")).strip() if col_acq else ""
            rec.sheet_name       = str(row.get(col_sheet, "")).strip() if col_sheet else ""

            # SR
            if SR:
                rec.result            = str(row.get(SR[1], "")).strip()
                rec.sample_mrl        = str(row.get(SR[2], "")).strip()
                rec.sample_units      = str(row.get(SR[3], "")).strip()
                rec.sample_dilution   = str(row.get(SR[4], "")).strip()
                rec.sample_analyzed   = str(row.get(SR[5], "")).strip()
                rec.sample_qualifier  = str(row.get(SR[6], "")).strip()

            # MB
            if MB:
                rec.mb_analyte  = str(row.get(MB[0], "")).strip()
                rec.mb_result   = str(row.get(MB[1], "")).strip()
                rec.mb_mrl      = str(row.get(MB[2], "")).strip()
                rec.mb_units    = str(row.get(MB[3], "")).strip()
                rec.mb_dilution = str(row.get(MB[4], "")).strip()

            # MS1
            if MS1:
                rec.ms1_analyte         = str(row.get(MS1[0], "")).strip()
                rec.ms1_result          = str(row.get(MS1[1], "")).strip()
                rec.ms1_mrl             = str(row.get(MS1[2], "")).strip()
                rec.ms1_units           = str(row.get(MS1[3], "")).strip()
                rec.ms1_dilution        = str(row.get(MS1[4], "")).strip()
                rec.ms1_fortified_level = str(row.get(MS1[5], "")).strip()
                rec.ms1_pct_rec         = str(row.get(MS1[6], "")).strip()
                rec.ms1_pct_rec_limits  = str(row.get(MS1[7], "")).strip()

            # MSD
            if MSD:
                rec.msd_analyte         = str(row.get(MSD[0], "")).strip()
                rec.msd_result          = str(row.get(MSD[1], "")).strip()
                rec.msd_units           = str(row.get(MSD[2], "")).strip()
                rec.msd_dilution        = str(row.get(MSD[3], "")).strip()
                rec.msd_pct_rec         = str(row.get(MSD[4], "")).strip()
                rec.msd_pct_rec_limits  = str(row.get(MSD[5], "")).strip()
                rec.msd_pct_rpd         = str(row.get(MSD[6], "")).strip()
                rec.msd_pct_rpd_limit   = str(row.get(MSD[7], "")).strip()

        db.commit()
    except Exception as e:
        db.rollback()
        return f"Import failed: {e}"
    finally:
        db.close()

    return (f"Imported {created} new, updated {updated}. "
            f"Skipped {skipped} rows without numeric Lab ID. From file: {filename}")

# =========================
# Audit / Export / Health
# =========================
@app.route("/audit")
def audit():
    if "username" not in session:
        return redirect(url_for("home"))
    if session.get("role") != "admin":
        flash("Admins only.", "error")
        return redirect(url_for("dashboard"))
    db = SessionLocal()
    rows = db.query(AuditLog).order_by(AuditLog.at.desc()).limit(500).all()
    db.close()
    return render_template("audit.html", user={"role":"admin"}, rows=rows)

@app.route("/export_csv")
def export_csv():
    if "username" not in session:
        return redirect(url_for("home"))
    db = SessionLocal()
    q = db.query(Report)
    if session.get("role") == "client":
        q = q.filter(Report.client == (session.get("client_name") or CLIENT_NAME))
    rows = q.all()
    db.close()

    data = [{
        "Lab ID": r.lab_id, "Client": r.client, "Analyte": r.test or "",
        "Result": r.result or "", "MRL": r.sample_mrl or "", "Units": r.sample_units or "",
        "Dilution": r.sample_dilution or "", "Analyzed": r.sample_analyzed or "",
        "Qualifier": r.sample_qualifier or "",
        "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
        "Received": r.collected_date.isoformat() if r.collected_date else "",
    } for r in rows]

    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv", as_attachment=True,
                     download_name="reports_export.csv")

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
