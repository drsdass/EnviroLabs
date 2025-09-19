\
import os
import io
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from flask import jsonify
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
    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)
    patient_name = Column(String, nullable=True)
    test = Column(String, nullable=True)
    result = Column(String, nullable=True)
    collected_date = Column(Date, nullable=True)
    resulted_date = Column(Date, nullable=True)
    pdf_url = Column(String, nullable=True)  # optional link to actual PDF file
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
    except Exception as e:
        db.rollback()
    finally:
        db.close()

def parse_date(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(val), fmt).date()
        except Exception:
            continue
    # Try pandas to_datetime
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None

# Infer common variant column names
COLUMN_ALIASES = {
    "lab_id": ["lab_id", "lab id", "id", "labid", "accession", "accession_id"],
    "client": ["client", "client_name", "account", "facility"],
    "patient_name": ["patient", "patient_name", "name"],
    "test": ["test", "panel", "assay"],
    "result": ["result", "final_result", "outcome"],
    "collected_date": ["collected_date", "collection_date", "collected"],
    "resulted_date": ["resulted_date", "reported_date", "finalized", "result_date"],
    "pdf_url": ["pdf", "pdf_url", "report_link"],
}

def get_col(df, logical_name):
    for candidate in COLUMN_ALIASES[logical_name]:
        matches = [c for c in df.columns if c.strip().lower() == candidate]
        if matches:
            return matches[0]
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

    # Filters
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
        try:
            sd = parse_date(start)
            if sd:
                q = q.filter(Report.resulted_date >= sd)
        except Exception:
            pass
    if end:
        try:
            ed = parse_date(end)
            if ed:
                q = q.filter(Report.resulted_date <= ed)
        except Exception:
            pass

    reports = q.order_by(Report.resulted_date.desc().nullslast(), Report.id.desc()).limit(500).all()
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

    filename = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(saved_path)

    keep = request.form.get("keep_original", "on") == "on"
    if not KEEP_UPLOADED_CSVS or not keep:
        # We'll parse from disk first, then delete file
        parse_path = saved_path
    else:
        parse_path = saved_path

    # Parse CSV with pandas; create/update reports by Lab ID
    try:
        df = pd.read_csv(parse_path)
    except Exception:
        try:
            df = pd.read_excel(parse_path)
        except Exception as e:
            flash(f"Could not read file: {e}", "error")
            if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
                os.remove(saved_path)
            return redirect(url_for("dashboard"))

    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    c_lab_id = get_col(df, "lab_id")
    c_client = get_col(df, "client")
    if not c_lab_id or not c_client:
        flash("CSV must include Lab ID and Client columns (various names accepted).", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    c_patient = get_col(df, "patient_name")
    c_test = get_col(df, "test")
    c_result = get_col(df, "result")
    c_collected = get_col(df, "collected_date")
    c_resulted = get_col(df, "resulted_date")
    c_pdf = get_col(df, "pdf_url")

    db = SessionLocal()
    created, updated = 0, 0
    try:
        for _, row in df.iterrows():
            lab_id = str(row[c_lab_id]).strip()
            if lab_id == "" or lab_id.lower() == "nan":
                continue
            client = str(row[c_client]).strip() if c_client else CLIENT_NAME

            # find existing
            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(
                    lab_id = lab_id,
                    client = client
                )
                db.add(existing)
                created += 1
            else:
                updated += 1

            if c_patient: existing.patient_name = None if pd.isna(row[c_patient]) else str(row[c_patient])
            if c_test: existing.test = None if pd.isna(row[c_test]) else str(row[c_test])
            if c_result: existing.result = None if pd.isna(row[c_result]) else str(row[c_result])
            if c_collected: existing.collected_date = parse_date(row[c_collected])
            if c_resulted: existing.resulted_date = parse_date(row[c_resulted])
            if c_pdf: existing.pdf_url = None if pd.isna(row[c_pdf]) else str(row[c_pdf])

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

    # Build dataframe
    data = [{
        "Lab ID": r.lab_id,
        "Client": r.client,
        "Patient": r.patient_name,
        "Test": r.test,
        "Result": r.result,
        "Collected Date": r.collected_date.isoformat() if r.collected_date else "",
        "Resulted Date": r.resulted_date.isoformat() if r.resulted_date else "",
        "PDF URL": r.pdf_url or ""
    } for r in rows]
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
