import os, io, json, re, sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

# ------------------- Config -------------------
SECRET_KEY      = os.getenv("SECRET_KEY", "dev-secret-change-me")
ADMIN_USERNAME  = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD  = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME     = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = os.getenv("KEEP_UPLOADED_CSVS", "true").lower() == "true"

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
    id            = Column(Integer, primary_key=True)
    lab_id        = Column(String, nullable=False, index=True)      # Sample ID / Laboratory ID
    client        = Column(String, nullable=False, index=True)
    patient_name  = Column(String, nullable=True)                    # unused in environmental; keep for compat
    test          = Column(String, nullable=True)                    # analyte name (from Master file)
    result        = Column(String, nullable=True)                    # human-friendly summary (Analyte + Result)
    collected_date= Column(Date, nullable=True)                      # “Received Date”
    resulted_date = Column(Date, nullable=True)                      # “Reported”
    pdf_url       = Column(String, nullable=True)
    payload       = Column(Text, nullable=True)                      # JSON: full report sections
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id       = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role     = Column(String, nullable=False)  # 'admin' or 'client'
    action   = Column(String, nullable=False)
    details  = Column(Text, nullable=True)
    at       = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# --- add payload column if missing (lightweight auto-migration) ---
def ensure_payload_column():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(reports)")
        cols = [r[1].lower() for r in cur.fetchall()]
        if "payload".lower() not in cols:
            cur.execute("ALTER TABLE reports ADD COLUMN payload TEXT")
            con.commit()

ensure_payload_column()

# ------------------- Helpers -------------------
def current_user():
    return {
        "username": session.get("username"),
        "role": session.get("role"),
        "client_name": session.get("client_name"),
    }

def log_action(username, role, action, details=""):
    db = SessionLocal()
    try:
        db.add(AuditLog(username=username, role=role, action=action, details=details))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

def is_blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""

DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%Y/%m/%d", "%m-%d-%Y", "%m-%d-%y")
def parse_date(val):
    if is_blank(val):
        return None
    s = str(val).strip()
    # strip times if present
    s = re.sub(r"[T ]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", "", s)
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # fallback
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None

def normalize_header(h):
    s = re.sub(r"\(.*?\)", "", str(h)).lower()      # remove parentheticals
    s = re.sub(r"[^a-z0-9]+", " ", s)               # non-alnum -> space
    return re.sub(r"\s+", " ", s).strip()

# Accepts exact OR close match (contains either way) against synonyms
def find_col(df, synonyms):
    cols = list(df.columns)
    norm_map = {c: normalize_header(c) for c in cols}
    syn_norm = [normalize_header(s) for s in synonyms]

    # exact norm match first
    for c, n in norm_map.items():
        if n in syn_norm:
            return c
    # contains (either direction)
    for c, n in norm_map.items():
        if any(n in sn or sn in n for sn in syn_norm):
            return c
    return None

# Master Upload File header map
ALIASES = {
    "lab_id": [
        "Sample ID", "Lab ID", "Laboratory ID",
        "Sample ID (Lab ID, Laboratory ID)", "Sample"
    ],
    "client": ["Client"],
    "phone": ["Phone"],
    "email": ["Email"],
    "project_lead": ["Project Lead"],
    "address": ["Address"],
    "reported": ["Reported", "Report Date", "Reported Date"],
    "received": ["Received Date", "Received"],
    "sample_name": ["Sample Name"],
    "prepared_by": ["Prepared By"],
    "matrix": ["Matrix"],
    "prepared_date": ["Prepared Date"],
    "qualifiers_summary": ["Qualifiers"],  # the summary in Sample Summary
    "asin": ["ASIN (Identifier)", "ASIN", "Identifier"],
    "weight_g": ["Product Weight (Grams)", "Weight (Grams)"],

    # SAMPLE RESULTS
    "sr_analyte": ["Analyte"],
    "sr_result": ["Result"],
    "sr_mrl": ["MRL"],
    "sr_units": ["Units"],
    "sr_dilution": ["Dilution"],
    "sr_analyzed": ["Analyzed"],
    "sr_qualifier": ["Qualifier"],

    # METHOD BLANK
    "mb_analyte": ["METHOD BLANK Analyte", "Method Blank Analyte", "MB Analyte"],
    "mb_result":  ["METHOD BLANK Result",  "Method Blank Result",  "MB Result"],
    "mb_units":   ["METHOD BLANK Units",   "Method Blank Units",   "MB Units"],
    "mb_dilution":["METHOD BLANK Dilution","Method Blank Dilution","MB Dilution"],

    # MATRIX SPIKE 1
    "ms1_analyte": ["MATRIX SPIKE 1 Analyte", "MS1 Analyte"],
    "ms1_result":  ["MATRIX SPIKE 1 Result",  "MS1 Result"],
    "ms1_mrl":     ["MATRIX SPIKE 1 MRL",     "MS1 MRL"],
    "ms1_units":   ["MATRIX SPIKE 1 Units",   "MS1 Units"],
    "ms1_dilution":["MATRIX SPIKE 1 Dilution","MS1 Dilution"],
    "ms1_fort":    ["MATRIX SPIKE 1 Fortified Level", "Fortified Level"],
    "ms1_rec":     ["MATRIX SPIKE 1 %REC", "%REC"],
    "ms1_rec_lim": ["MATRIX SPIKE 1 %REC Limits", "%REC Limits"],

    # MATRIX SPIKE DUPLICATE
    "msd_analyte": ["MATRIX SPIKE DUPLICATE Analyte", "MSD Analyte"],
    "msd_result":  ["MATRIX SPIKE DUPLICATE Result",  "MSD Result"],
    "msd_units":   ["MATRIX SPIKE DUPLICATE Units",   "MSD Units"],
    "msd_dilution":["MATRIX SPIKE DUPLICATE Dilution","MSD Dilution"],
    "msd_rec":     ["MATRIX SPIKE DUPLICATE %REC",    "MSD %REC"],
    "msd_rec_lim": ["MATRIX SPIKE DUPLICATE %REC Limits", "MSD %REC Limits"],
    "msd_rpd":     ["MATRIX SPIKE DUPLICATE %RPD",    "%RPD"],
    "msd_rpd_lim": ["MATRIX SPIKE DUPLICATE %RPD Limit", "%RPD Limit"],

    "acq_dt": ["Acq. Date-Time", "Acquisition Date-Time", "Acq Date Time"],
    "sheetname": ["SheetName", "Sheet Name"],
    "pdf_url": ["PDF URL", "Report Link", "pdf", "link"]
}

def build_result_summary(row, c):
    # A friendly "Analyte: Result Units (MRL...)" style line
    parts = []
    analyte = row.get(c["sr_analyte"])
    result  = row.get(c["sr_result"])
    units   = row.get(c["sr_units"])
    mrl     = row.get(c["sr_mrl"])
    q       = row.get(c["sr_qualifier"])
    if analyte and not is_blank(analyte):
        parts.append(str(analyte).strip() + ":")
    if not is_blank(result):
        parts.append(str(result).strip())
    if not is_blank(units):
        parts.append(str(units).strip())
    if not is_blank(mrl):
        parts.append(f"(MRL {mrl})")
    if not is_blank(q):
        parts.append(f"[{q}]")
    return " ".join(parts) if parts else None

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

    # Filters
    lab_id = (request.args.get("lab_id") or "").strip()
    start  = (request.args.get("start")  or "").strip()
    end    = (request.args.get("end")    or "").strip()

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

    reports = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).limit(500).all()
    db.close()

    # Attach parsed payload for templates
    for r in reports:
        try:
            r.p = json.loads(r.payload) if r.payload else {}
        except Exception:
            r.p = {}

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
    try:
        r.p = json.loads(r.payload) if r.payload else {}
    except Exception:
        r.p = {}
    return render_template("report_detail.html", user=u, r=r)

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    if u["role"] != "admin":
        flash("Only admins can upload CSV files", "error")
        return redirect(url_for("dashboard"))

    f = request.files.get("csv_file")
    if not f:
        flash("No file uploaded", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(f.filename or "upload.csv")
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(saved_path)

    keep = request.form.get("keep_original", "on") == "on"
    parse_path = saved_path

    try:
        df = pd.read_csv(parse_path, dtype=str).fillna("")  # read everything as text; avoid pandas NA pitfalls
    except Exception as e:
        flash(f"Could not read file: {e}", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # map columns
    col = {}
    for key, syns in ALIASES.items():
        col[key] = find_col(df, syns)

    # Validate required
    if not col["lab_id"] or not col["client"]:
        found = ", ".join(df.columns)
        flash(
            "CSV must include Lab ID (aka 'Sample ID') and Client columns. "
            f"Could not find them. Found columns: {found}",
            "error"
        )
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    created, updated = 0, 0
    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            # Pull basic identifiers
            lab_id_val = row[col["lab_id"]].strip() if col["lab_id"] else ""
            client_val = row[col["client"]].strip() if col["client"] else CLIENT_NAME
            if lab_id_val == "":
                continue

            existing = db.query(Report).filter(Report.lab_id == lab_id_val).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id_val, client=client_val)
                db.add(existing)
                created += 1
            else:
                updated += 1
                existing.client = client_val  # keep client in sync

            # Dates
            reported = parse_date(row[col["reported"]]) if col["reported"] else None
            received = parse_date(row[col["received"]]) if col["received"] else None
            if reported: existing.resulted_date  = reported
            if received: existing.collected_date = received

            # Sample results summary into top-level fields
            result_summary = build_result_summary(row, col)
            if result_summary:
                existing.result = result_summary

            if col["sr_analyte"]:
                existing.test = row[col["sr_analyte"]].strip() or existing.test

            # Build payload JSON (all sections)
            payload = {
                "client_info": {
                    "client": row[col["client"]].strip() if col["client"] else "",
                    "phone": row[col["phone"]].strip() if col["phone"] else "",
                    "email": row[col["email"]].strip() if col["email"] else "",
                    "project_lead": row[col["project_lead"]].strip() if col["project_lead"] else "",
                    "address": row[col["address"]].strip() if col["address"] else "",
                    "reported": row[col["reported"]].strip() if col["reported"] else "",
                    "received_date": row[col["received"]].strip() if col["received"] else "",
                },
                "sample_summary": {
                    "sample_name": row[col["sample_name"]].strip() if col["sample_name"] else "",
                    "prepared_by": row[col["prepared_by"]].strip() if col["prepared_by"] else "",
                    "matrix": row[col["matrix"]].strip() if col["matrix"] else "",
                    "prepared_date": row[col["prepared_date"]].strip() if col["prepared_date"] else "",
                    "qualifiers": row[col["qualifiers_summary"]].strip() if col["qualifiers_summary"] else "",
                    "asin": row[col["asin"]].strip() if col["asin"] else "",
                    "product_weight_g": row[col["weight_g"]].strip() if col["weight_g"] else "",
                },
                "sample_results": {
                    "analyte": row[col["sr_analyte"]].strip() if col["sr_analyte"] else "",
                    "result": row[col["sr_result"]].strip() if col["sr_result"] else "",
                    "mrl": row[col["sr_mrl"]].strip() if col["sr_mrl"] else "",
                    "units": row[col["sr_units"]].strip() if col["sr_units"] else "",
                    "dilution": row[col["sr_dilution"]].strip() if col["sr_dilution"] else "",
                    "analyzed": row[col["sr_analyzed"]].strip() if col["sr_analyzed"] else "",
                    "qualifier": row[col["sr_qualifier"]].strip() if col["sr_qualifier"] else "",
                },
                "method_blank": {
                    "analyte": row[col["mb_analyte"]].strip() if col["mb_analyte"] else "",
                    "result": row[col["mb_result"]].strip() if col["mb_result"] else "",
                    "units": row[col["mb_units"]].strip() if col["mb_units"] else "",
                    "dilution": row[col["mb_dilution"]].strip() if col["mb_dilution"] else "",
                },
                "matrix_spike_1": {
                    "analyte": row[col["ms1_analyte"]].strip() if col["ms1_analyte"] else "",
                    "result": row[col["ms1_result"]].strip() if col["ms1_result"] else "",
                    "mrl": row[col["ms1_mrl"]].strip() if col["ms1_mrl"] else "",
                    "units": row[col["ms1_units"]].strip() if col["ms1_units"] else "",
                    "dilution": row[col["ms1_dilution"]].strip() if col["ms1_dilution"] else "",
                    "fortified_level": row[col["ms1_fort"]].strip() if col["ms1_fort"] else "",
                    "pct_rec": row[col["ms1_rec"]].strip() if col["ms1_rec"] else "",
                    "pct_rec_limits": row[col["ms1_rec_lim"]].strip() if col["ms1_rec_lim"] else "",
                },
                "matrix_spike_dup": {
                    "analyte": row[col["msd_analyte"]].strip() if col["msd_analyte"] else "",
                    "result": row[col["msd_result"]].strip() if col["msd_result"] else "",
                    "units": row[col["msd_units"]].strip() if col["msd_units"] else "",
                    "dilution": row[col["msd_dilution"]].strip() if col["msd_dilution"] else "",
                    "pct_rec": row[col["msd_rec"]].strip() if col["msd_rec"] else "",
                    "pct_rec_limits": row[col["msd_rec_lim"]].strip() if col["msd_rec_lim"] else "",
                    "pct_rpd": row[col["msd_rpd"]].strip() if col["msd_rpd"] else "",
                    "pct_rpd_limit": row[col["msd_rpd_lim"]].strip() if col["msd_rpd_lim"] else "",
                },
                "metadata": {
                    "acq_datetime": row[col["acq_dt"]].strip() if col["acq_dt"] else "",
                    "sheetname": row[col["sheetname"]].strip() if col["sheetname"] else "",
                    "pdf_url": row[col["pdf_url"]].strip() if col["pdf_url"] else "",
                }
            }

            existing.payload = json.dumps(payload)

        db.commit()
        flash(f"Imported {created} new and updated {updated} report(s).", "success")
        log_action(u["username"], u["role"], "upload_csv", f"{filename} -> created {created}, updated {updated}")

    except Exception as e:
        db.rollback()
        flash(f"Import failed: {e}", "error")
    finally:
        db.close()

    if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
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
        try:
            p = json.loads(r.payload) if r.payload else {}
        except Exception:
            p = {}
        data.append({
            "Lab ID": r.lab_id,
            "Client": r.client,
            "Analyte": p.get("sample_results", {}).get("analyte", ""),
            "Result": p.get("sample_results", {}).get("result", ""),
            "Units":  p.get("sample_results", {}).get("units", ""),
            "MRL":    p.get("sample_results", {}).get("mrl", ""),
            "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "Received": r.collected_date.isoformat() if r.collected_date else "",
        })
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(data)} records")
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")), mimetype="text/csv",
                     as_attachment=True, download_name="reports_export.csv")

# ----------- Minimal health check for Render -----------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
