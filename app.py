import os
import io
import json
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

# ------------------- Config -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME     = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = os.getenv("KEEP_UPLOADED_CSVS", "true").lower() == "true"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- App -------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ------------------- DB -------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)

    # list/filter fields
    lab_id = Column(String, nullable=False, index=True)  # Sample / Lab / Laboratory ID
    client = Column(String, nullable=False, index=True)

    sample_name = Column(String, nullable=True)
    test = Column(String, nullable=True)         # primary analyte (Sample Results)
    result = Column(String, nullable=True)       # primary result  (Sample Results)
    units = Column(String, nullable=True)        # primary units   (Sample Results)

    collected_date = Column(Date, nullable=True)  # Received Date
    resulted_date  = Column(Date, nullable=True)  # Reported

    pdf_url = Column(String, nullable=True)

    # full row payload as JSON (everything from Master Upload row)
    payload = Column(Text, nullable=True)

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

# tiny, safe migrations
insp = inspect(engine)
cols = {c['name'] for c in insp.get_columns('reports')}
with engine.begin() as conn:
    if 'units' not in cols:
        conn.exec_driver_sql("ALTER TABLE reports ADD COLUMN units VARCHAR")
    if 'sample_name' not in cols:
        conn.exec_driver_sql("ALTER TABLE reports ADD COLUMN sample_name VARCHAR")
    if 'payload' not in cols:
        conn.exec_driver_sql("ALTER TABLE reports ADD COLUMN payload TEXT")

# ------------------- Helpers -------------------
MASTER_LAB_HEADERS = [
    "Sample ID (Lab ID, Laboratory ID)",
    "Laboratory ID",
    "Lab ID",
    "Sample ID",
]

def parse_date(val):
    if val is None:
        return None
    sval = str(val).strip()
    if sval == "" or sval.lower() in {"na", "n/a", "#n/a", "nan", "none"}:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(sval, fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(sval).date()
    except Exception:
        return None

def clean(val):
    if val is None:
        return None
    sval = str(val).strip()
    return None if sval.lower() in {"", "na", "n/a", "#n/a", "nan", "none"} else sval

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

def read_master_csv_smart(path):
    """
    Auto-skips the banner row (CLIENT INFORMATION / SAMPLE SUMMARY / ...) if present.
    Returns DataFrame of strings with duplicate headers preserved (Analyte, Analyte.1, ...).
    """
    df0 = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    if any(str(h).strip() in MASTER_LAB_HEADERS for h in df0.columns):
        return df0
    # try second row as header
    df1 = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", header=1)
    if any(str(h).strip() in MASTER_LAB_HEADERS for h in df1.columns):
        return df1
    return df0

def pick_first_exact(df, names):
    for n in names:
        for c in df.columns:
            if str(c).strip() == n:
                return c
    return None

def get_nth(df, base_name, nth):
    """Return nth occurrence of a duplicated header (Analyte, Analyte.1, ...)."""
    matches = [c for c in df.columns if c.split('.', 1)[0].strip() == base_name]
    return matches[nth] if nth < len(matches) else None

def map_master_row(df, row):
    # identifiers
    lab_col   = pick_first_exact(df, MASTER_LAB_HEADERS)
    client_c  = pick_first_exact(df, ["Client"])
    phone_c   = pick_first_exact(df, ["Phone"])
    email_c   = pick_first_exact(df, ["Email"])
    p_lead_c  = pick_first_exact(df, ["Project Lead"])
    addr_c    = pick_first_exact(df, ["Address"])
    reported_c= pick_first_exact(df, ["Reported"])
    recv_c    = pick_first_exact(df, ["Received Date"])
    sname_c   = pick_first_exact(df, ["Sample Name"])
    prep_by_c = pick_first_exact(df, ["Prepared By"])
    matrix_c  = pick_first_exact(df, ["Matrix"])
    prep_dt_c = pick_first_exact(df, ["Prepared Date"])
    qual_c    = pick_first_exact(df, ["Qualifiers"])
    asin_c    = pick_first_exact(df, ["ASIN (Identifier)"])
    weight_c  = pick_first_exact(df, ["Product Weight (Grams)"])

    # SAMPLE RESULTS (first set)
    sr_analyte_c   = get_nth(df, "Analyte", 0)
    sr_result_c    = get_nth(df, "Result", 0)
    sr_mrl_c       = get_nth(df, "MRL", 0)
    sr_units_c     = get_nth(df, "Units", 0)
    sr_dilution_c  = get_nth(df, "Dilution", 0)
    sr_analyzed_c  = get_nth(df, "Analyzed", 0)
    sr_qual_c      = get_nth(df, "Qualifier", 0)

    # METHOD BLANK (second set)
    mb_analyte_c   = get_nth(df, "Analyte", 1)
    mb_result_c    = get_nth(df, "Result", 1)
    mb_mrl_c       = get_nth(df, "MRL", 1)
    mb_units_c     = get_nth(df, "Units", 1)
    mb_dilution_c  = get_nth(df, "Dilution", 1)

    # MATRIX SPIKE 1 (third set)
    ms1_analyte_c  = get_nth(df, "Analyte", 2)
    ms1_result_c   = get_nth(df, "Result", 2)
    ms1_mrl_c      = get_nth(df, "MRL", 2)
    ms1_units_c    = get_nth(df, "Units", 2)
    ms1_dilution_c = get_nth(df, "Dilution", 2)
    ms1_fort_c     = pick_first_exact(df, ["Fortified Level"])
    ms1_prec_c     = pick_first_exact(df, ["%REC"])
    ms1_prec_lim_c = pick_first_exact(df, ["%REC Limits"])

    # MATRIX SPIKE DUPLICATE (fourth set)
    msd_analyte_c  = get_nth(df, "Analyte", 3)
    msd_result_c   = get_nth(df, "Result", 3)
    msd_units_c    = get_nth(df, "Units", 3)
    msd_dilution_c = get_nth(df, "Dilution", 3)
    msd_prec_c     = get_nth(df, "%REC", 1)         # second %REC
    msd_prec_lim_c = get_nth(df, "%REC Limits", 1)  # second %REC Limits
    msd_rpd_c      = pick_first_exact(df, ["%RPD"])
    msd_rpd_lim_c  = pick_first_exact(df, ["%RPD Limit"])

    acq_c   = pick_first_exact(df, ["Acq. Date-Time"])
    sheet_c = pick_first_exact(df, ["SheetName"])

    def g(col):
        return clean(row.get(col)) if col else None

    mapped = {
        "lab_id": g(lab_col),
        "client": g(client_c),

        "client_info": {
            "phone": g(phone_c),
            "email": g(email_c),
            "project_lead": g(p_lead_c),
            "address": g(addr_c),
        },
        "sample_summary": {
            "reported": g(reported_c),
            "received_date": g(recv_c),
            "sample_name": g(sname_c),
            "prepared_by": g(prep_by_c),
            "matrix": g(matrix_c),
            "prepared_date": g(prep_dt_c),
            "qualifiers": g(qual_c),
            "asin": g(asin_c),
            "product_weight_g": g(weight_c),
        },
        "sample_results": {
            "analyte": g(sr_analyte_c),
            "result": g(sr_result_c),
            "mrl": g(sr_mrl_c),
            "units": g(sr_units_c),
            "dilution": g(sr_dilution_c),
            "analyzed": g(sr_analyzed_c),
            "qualifier": g(sr_qual_c),
        },
        "method_blank": {
            "analyte": g(mb_analyte_c),
            "result": g(mb_result_c),
            "mrl": g(mb_mrl_c),
            "units": g(mb_units_c),
            "dilution": g(mb_dilution_c),
        },
        "matrix_spike_1": {
            "analyte": g(ms1_analyte_c),
            "result": g(ms1_result_c),
            "mrl": g(ms1_mrl_c),
            "units": g(ms1_units_c),
            "dilution": g(ms1_dilution_c),
            "fortified_level": g(ms1_fort_c),
            "pct_rec": g(ms1_prec_c),
            "pct_rec_limits": g(ms1_prec_lim_c),
        },
        "matrix_spike_dup": {
            "analyte": g(msd_analyte_c),
            "result": g(msd_result_c),
            "units": g(msd_units_c),
            "dilution": g(msd_dilution_c),
            "pct_rec": g(msd_prec_c),
            "pct_rec_limits": g(msd_prec_lim_c),
            "pct_rpd": g(msd_rpd_c),
            "pct_rpd_limit": g(msd_rpd_lim_c),
        },
        "acq_datetime": g(acq_c),
        "sheet_name": g(sheet_c),
    }

    # top-level quick fields for list
    mapped["list_test"]   = mapped["sample_results"]["analyte"]
    mapped["list_result"] = mapped["sample_results"]["result"]
    mapped["list_units"]  = mapped["sample_results"]["units"]

    return mapped

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

    rows = q.all()
    # sort: newest reported first, None at bottom
    rows.sort(key=lambda r: (r.resulted_date is None, r.resulted_date or date.min), reverse=True)
    db.close()

    return render_template("dashboard.html", user=u, reports=rows)

@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    db = SessionLocal()
    r = db.query(Report).get(report_id)  # OK for SQLAlchemy 2.x, though deprecated
    db.close()
    if not r:
        flash("Report not found", "error")
        return redirect(url_for("dashboard"))
    if u["role"] == "client" and r.client != u["client_name"]:
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    payload = {}
    if r.payload:
        try:
            payload = json.loads(r.payload)
        except Exception:
            payload = {}
    return render_template("report_detail.html", user=u, r=r, p=payload)

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
    parse_path = saved_path

    try:
        df = read_master_csv_smart(parse_path)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        flash(f"Could not read file: {e}", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    lab_header = pick_first_exact(df, MASTER_LAB_HEADERS)
    client_header = pick_first_exact(df, ["Client"])
    if not lab_header or not client_header:
        found = ", ".join(df.columns)
        flash(
            "Could not find a Lab ID column. Include a header like: "
            "'Sample ID (Lab ID, Laboratory ID)' / 'Laboratory ID' / 'Lab ID' / 'Sample ID'. "
            f"Found columns: {found}",
            "error",
        )
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    db = SessionLocal()
    created, updated = 0, 0
    try:
        for _, row in df.iterrows():
            mapped = map_master_row(df, row)
            lab_id = mapped["lab_id"]
            client = mapped["client"] or CLIENT_NAME
            if not lab_id or not client:
                continue

            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing)
                created += 1
            else:
                updated += 1

            # update list fields
            existing.client = client
            existing.sample_name    = mapped["sample_summary"]["sample_name"]
            existing.test           = mapped["list_test"]
            existing.result         = mapped["list_result"]
            existing.units          = mapped["list_units"]
            existing.collected_date = parse_date(mapped["sample_summary"]["received_date"])
            existing.resulted_date  = parse_date(mapped["sample_summary"]["reported"])
            existing.payload        = json.dumps(mapped)

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

    data = [{
        "Lab ID": r.lab_id,
        "Client": r.client,
        "Sample Name": r.sample_name or "",
        "Analyte": r.test or "",
        "Result": r.result or "",
        "Units": r.units or "",
        "Received Date": r.collected_date.isoformat() if r.collected_date else "",
        "Reported": r.resulted_date.isoformat() if r.resulted_date else "",
    } for r in rows]
    df = pd.DataFrame(data)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    log_action(u["username"], u["role"], "export_csv", f"Exported {len(data)} records")
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")), mimetype="text/csv",
                     as_attachment=True, download_name="reports_export.csv")

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
