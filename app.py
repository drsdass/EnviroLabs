import os
import io
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

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MASTER_CACHE_PATH = os.path.join(UPLOAD_FOLDER, "_master_latest.csv")

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
    patient_name = Column(String, nullable=True)  # used for "Sample Name"
    test = Column(String, nullable=True)          # e.g., "Bisphenol S", "PFAS"
    result = Column(String, nullable=True)
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
        d = pd.to_datetime(val, errors="coerce")
        return d.date() if pd.notna(d) else None
    except Exception:
        return None

def norm(s: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(s))
    return cleaned.split()

def header_contains(col: str, tokens: list[str]) -> bool:
    words = norm(col)
    return all(tok in words for tok in tokens)

ALIASES = {
    "lab_id": [
        ["lab", "id"], ["laboratory", "id"], ["sample", "id"],
        ["sample", "name"], ["sample"], ["accession", "id"]
    ],
    "client": [["client"], ["client", "name"], ["account"], ["facility"]],
    "test":   [["test"], ["panel"], ["assay"], ["analyte"]],
    "result": [["result"], ["final", "result"], ["outcome"]],
    "collected_date": [["collected", "date"], ["collection", "date"], ["collected"], ["received", "date"]],
    "resulted_date":  [["resulted", "date"], ["reported", "date"], ["finalized"], ["result", "date"]],
    "pdf_url": [["pdf"], ["pdf", "url"], ["report", "link"]],
    "sample_name": [["sample", "name"]],
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

def _maybe(df: pd.DataFrame, words_like):
    for c in df.columns:
        if header_contains(c, [w.lower() for w in words_like]):
            return c
    return None

# ---------- Master Upload cache helpers ----------
def _detect_and_reheader(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the real header row is within the first few rows (e.g., banner first line),
    move it to columns and drop rows above it.
    """
    # If we already have a Sample ID-ish header, keep as is.
    if any("sample id" in " ".join(norm(c)) for c in df.columns):
        return df

    # Look for a row that contains 'Sample ID' text
    for i in range(min(8, len(df))):
        row = [str(x) for x in df.iloc[i].tolist()]
        if any("sample id" in x.lower() for x in row):
            df = df.copy()
            df.columns = row
            df = df.iloc[i + 1 :].reset_index(drop=True)
            return df
    return df

def _write_master_cache_if_possible(df: pd.DataFrame):
    """
    If df contains the Master columns, write a normalized cache to MASTER_CACHE_PATH.
    """
    df = _detect_and_reheader(df)

    # Column selectors (prefer exacts, else by position A..F)
    def pick(header_name: str, fallback_index: int | None = None):
        target = header_name.lower()
        for c in df.columns:
            if str(c).strip().lower() == target:
                return c
        for c in df.columns:
            if target in str(c).strip().lower():
                return c
        if fallback_index is not None and fallback_index < len(df.columns):
            return df.columns[fallback_index]
        return None

    c_lab     = pick("sample id", 0) or pick("lab id", 0) or pick("sample name", 0)
    c_client  = pick("client", 1)
    c_phone   = pick("phone", 2)
    c_email   = pick("email", 3)
    c_lead    = pick("project lead", 4)
    c_address = pick("address", 5)

    # Only write cache if we have at least Sample ID + Email/Lead/Address columns present
    needed = [c_lab, c_client, c_phone, c_email, c_lead, c_address]
    if not c_lab or all(x is None for x in needed[1:]):
        return  # Not a master-style file; skip silently

    cache_cols = {}
    if c_lab:     cache_cols["Sample ID"]   = df[c_lab].astype(str).str.strip()
    if c_client:  cache_cols["Client"]      = df[c_client].astype(str).fillna("").str.strip()
    if c_phone:   cache_cols["Phone"]       = df[c_phone].astype(str).fillna("").str.strip()
    if c_email:   cache_cols["Email"]       = df[c_email].astype(str).fillna("").str.strip()
    if c_lead:    cache_cols["Project Lead"]= df[c_lead].astype(str).fillna("").str.strip()
    if c_address: cache_cols["Address"]     = df[c_address].astype(str).fillna("").str.strip()

    cache_df = pd.DataFrame(cache_cols)
    try:
        cache_df.to_csv(MASTER_CACHE_PATH, index=False)
    except Exception:
        pass

def _lookup_master_cached(lab_id: str) -> dict:
    """
    Read master cache and return client info for a specific Sample ID/Lab ID.
    """
    if not os.path.exists(MASTER_CACHE_PATH):
        return {}
    try:
        df = pd.read_csv(MASTER_CACHE_PATH, dtype=str).fillna("")
    except Exception:
        return {}
    if "Sample ID" not in df.columns:
        return {}

    mask = df["Sample ID"].astype(str).str.strip() == str(lab_id).strip()
    hit = df.loc[mask]
    if hit.empty:
        return {}
    row = hit.iloc[0]
    return {
        "client":       row.get("Client", "").strip(),
        "phone":        row.get("Phone", "").strip(),
        "email":        row.get("Email", "").strip(),
        "project_lead": row.get("Project Lead", "").strip(),
        "address":      row.get("Address", "").strip(),
    }

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

    # Base payload
    p = {
        "client_info": {
            "client": r.client or "",
            "phone": "",
            "email": "support@envirolabsusa.com",
            "project_lead": "",
            "address": "",
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": r.patient_name or r.lab_id or "",
            "prepared_by": "", "matrix": "", "prepared_date": "",
            "qualifiers": "", "asin": "", "product_weight_g": ""
        },
        "sample_results": {
            "analyte": r.test or "", "result": r.result or "", "mrl": "", "units": "",
            "dilution": "", "analyzed": "", "qualifier": ""
        },
        "method_blank":   {"analyte":"", "result":"", "mrl":"", "units":"", "dilution":""},
        "matrix_spike_1": {"analyte":"", "result":"", "mrl":"", "units":"", "dilution":"", "fortified_level":"", "pct_rec":"", "pct_rec_limits":""},
        "matrix_spike_dup":{"analyte":"", "result":"", "units":"", "dilution":"", "pct_rec":"", "pct_rec_limits":"", "pct_rpd":"", "pct_rpd_limit":""},
        "acq_datetime": "", "sheet_name": ""
    }

    # Overlay from cached Master Upload (if available)
    master = _lookup_master_cached(r.lab_id)
    if master:
        if master.get("client"):       p["client_info"]["client"] = master["client"]
        if master.get("phone"):        p["client_info"]["phone"] = master["phone"]
        if master.get("email"):        p["client_info"]["email"] = master["email"]
        if master.get("project_lead"): p["client_info"]["project_lead"] = master["project_lead"]
        if master.get("address"):      p["client_info"]["address"] = master["address"]

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

    # Parse file
    df = None
    try:
        ext = os.path.splitext(saved_path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(saved_path, engine="openpyxl")
        else:
            df = pd.read_csv(saved_path)
    except Exception:
        try:
            df = pd.read_csv(saved_path)
        except Exception:
            try:
                df = pd.read_excel(saved_path, engine="openpyxl")
            except Exception as e:
                flash(f"Could not read file: {e}", "error")
                if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
                    os.remove(saved_path)
                return redirect(url_for("dashboard"))

    # Try to write/update the Master cache if this looks like the master file
    try:
        _write_master_cache_if_possible(df.copy())
    except Exception:
        pass

    # Normalize headers (strip originals)
    df.columns = [str(c).strip() for c in df.columns]

    # Standard import into Report table
    c_lab_id = find_col(df, "lab_id")
    c_client = find_col(df, "client")
    if not c_lab_id:
        for c in df.columns:
            if "sample" in " ".join(norm(c)) and "id" in " ".join(norm(c)):
                c_lab_id = c
                break
    if not c_client:
        for c in df.columns:
            if "client" in " ".join(norm(c)):
                c_client = c
                break

    if not c_lab_id or not c_client:
        preview = ", ".join(df.columns[:20])
        flash("CSV must include Lab ID (aka 'Sample ID') and Client columns "
              f"(various names accepted). Found columns: {preview}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    c_test        = find_col(df, "test") or _maybe(df, ["analyte"])
    c_result      = find_col(df, "result")
    c_collected   = find_col(df, "collected_date")  # "Received Date"
    c_resulted    = find_col(df, "resulted_date")   # "Reported"
    c_pdf         = find_col(df, "pdf_url")
    c_sample_name = find_col(df, "sample_name")

    db = SessionLocal()
    created, updated = 0, 0
    try:
        for _, row in df.iterrows():
            lab_id_val = row.get(c_lab_id, "")
            lab_id = "" if pd.isna(lab_id_val) else str(lab_id_val).strip()
            if lab_id == "" or lab_id.lower() == "nan":
                continue

            client_val = row.get(c_client, CLIENT_NAME)
            client = "" if pd.isna(client_val) else str(client_val).strip() or CLIENT_NAME

            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

            if c_sample_name:
                existing.patient_name = None if pd.isna(row.get(c_sample_name)) else str(row.get(c_sample_name))
            if c_test:
                existing.test = None if pd.isna(row.get(c_test)) else str(row.get(c_test))
            if c_result:
                existing.result = None if pd.isna(row.get(c_result)) else str(row.get(c_result))
            if c_collected:
                existing.collected_date = parse_date(row.get(c_collected))
            if c_resulted:
                existing.resulted_date = parse_date(row.get(c_resulted))
            if c_pdf:
                val = row.get(c_pdf)
                existing.pdf_url = "" if pd.isna(val) else str(val)

        db.commit()
        flash(f"Imported {created} new and updated {updated} report(s).", "success")
        log_action(u["username"], u["role"], "upload_csv", f"{filename} -> created {created}, updated {updated}")
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
        "Sample Name": r.patient_name or "",
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
