import os
import io
import secrets
from datetime import datetime, date
from flask import (
    Flask, render_template, render_template_string, request, redirect, url_for,
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

# Reviewer defaults – hardcoded, but can be overridden by env if you want later
REVIEWER_EMAIL = os.getenv("REVIEWER_EMAIL", "satishsdass@gmail.com")
REVIEWER_NAME  = os.getenv("REVIEWER_NAME", "Reviewer")

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
    patient_name = Column(String, nullable=True)  # unused in your domain, left for compatibility
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

# New table to avoid altering the existing Report schema
class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, index=True, nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, default="pending")  # pending/approved/rejected
    reviewer_email = Column(String, nullable=False)
    reviewer_name = Column(String, nullable=True)
    requested_by = Column(String, nullable=True)
    requested_at = Column(DateTime, default=datetime.utcnow)
    acted_by = Column(String, nullable=True)
    acted_at = Column(DateTime, nullable=True)
    note = Column(Text, nullable=True)

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
        dt = pd.to_datetime(val, errors="coerce")
        return dt.date() if pd.notna(dt) else None
    except Exception:
        return None

def norm(s: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(s)).split()

def header_contains(col: str, tokens: list[str]) -> bool:
    words = norm(col)
    return all(tok in words for tok in tokens)

ALIASES = {
    "lab_id": [
        ["lab", "id"],
        ["laboratory", "id"],
        ["sample", "id"],
        ["sample"],                  # catch "Sample Name" when key identifier
        ["sample", "name"],
        ["accession", "id"]
    ],
    "client": [
        ["client"], ["client", "name"], ["account"], ["facility"]
    ],
    "test": [["test"], ["panel"], ["assay"], ["analyte"]],
    "result": [["result"], ["final", "result"], ["outcome"]],
    "collected_date": [["collected", "date"], ["collection", "date"], ["collected"], ["received", "date"]],
    "resulted_date": [["resulted", "date"], ["reported", "date"], ["finalized"], ["result", "date"]],
    "pdf_url": [["pdf"], ["pdf", "url"], ["report", "link"]],
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

def is_numeric_leading(s: str) -> bool:
    s = (s or "").strip()
    return len(s) > 0 and s[0].isdigit()

def is_target_analyte(val: str) -> bool:
    s = (val or "").strip().lower()
    return ("bisphenol s" in s) or ("pfas" in s)

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

    # Safe default payload for the template ('p')
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
            "method_blank": {
                "analyte": "", "result": "", "mrl": "", "units": "", "dilution": ""
            },
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

    df = None
    ext = os.path.splitext(saved_path)[1].lower()
    try:
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

    # Normalize columns (strip only)
    df.columns = [str(c).strip() for c in df.columns]

    # If this is the "Master Upload File" shape with a banner row,
    # we auto-detect the header row by looking for the Sample ID col in row 1/2.
    # If your file already loads with correct headers, this has no effect.
    if "Sample ID (Lab ID, Laboratory ID)" in df.columns:
        pass  # already good
    else:
        # Try to find the header row (where col A equals 'Sample ID...' etc.)
        header_row = None
        for i in range(min(5, len(df))):
            row_vals = [str(x).strip() for x in df.iloc[i].tolist()]
            if any("sample id" in " ".join(norm(v)) for v in row_vals):
                header_row = i
                break
        if header_row is not None:
            df = pd.read_csv(saved_path, header=header_row)

    # Re-normalize
    df.columns = [str(c).strip() for c in df.columns]

    # Core columns
    c_lab_id = find_col(df, "lab_id")
    c_client = find_col(df, "client")

    # Second chance for "Sample ID (Lab ID, Laboratory ID)"
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

    # Optional/related columns
    c_test        = find_col(df, "test") or _maybe(df, ["analyte"])
    c_result      = find_col(df, "result")
    c_collected   = find_col(df, "collected_date")  # "Received Date"
    c_resulted    = find_col(df, "resulted_date")   # "Reported"
    c_pdf         = find_col(df, "pdf_url")

    # Count filters
    skipped_non_numeric = 0
    skipped_non_target  = 0
    created, updated    = 0, 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            lab_id_val = row.get(c_lab_id, "")
            lab_id = "" if pd.isna(lab_id_val) else str(lab_id_val).strip()
            if lab_id == "" or lab_id.lower() == "nan":
                continue

            # Only create report rows when Lab ID starts with a number AND analyte is BPS/PFAS (if available)
            if not is_numeric_leading(lab_id):
                skipped_non_numeric += 1
                continue

            client_val = row.get(c_client, CLIENT_NAME)
            client = "" if pd.isna(client_val) else str(client_val).strip() or CLIENT_NAME

            # If we have analyte column, enforce target filter
            if c_test:
                analyte_val = row.get(c_test, "")
                if not is_target_analyte(analyte_val):
                    skipped_non_target += 1
                    continue

            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing)
                created += 1
            else:
                existing.client = client
                updated += 1

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
        flash(
            f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_non_numeric} non-numeric Lab ID row(s) and {skipped_non_target} non-target analyte row(s).",
            "success"
        )
        log_action(u["username"], u["role"], "upload_csv",
                   f"{filename} -> created {created}, updated {updated}, skipped_non_numeric={skipped_non_numeric}, skipped_non_target={skipped_non_target}")
    except Exception as e:
        db.rollback()
        flash(f"Import failed: {e}", "error")
    finally:
        db.close()

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

# ----------- Review flow -----------
def make_review_token() -> str:
    return secrets.token_urlsafe(24)

@app.route("/request_review/<int:report_id>", methods=["POST"])
@require_login(role="admin")
def request_review(report_id):
    u = current_user()
    db = SessionLocal()
    try:
        r = db.query(Report).get(report_id)
        if not r:
            flash("Report not found", "error")
            return redirect(url_for("dashboard"))

        token = make_review_token()
        rr = Review(
            report_id=report_id,
            token=token,
            status="pending",
            reviewer_email=REVIEWER_EMAIL,
            reviewer_name=REVIEWER_NAME,
            requested_by=u["username"] or "admin",
        )
        db.add(rr)
        db.commit()

        link = url_for("review_decide", token=token, _external=True)
        flash(f"Review link created and (simulated) emailed to {REVIEWER_EMAIL}. Link: {link}", "success")
        log_action(u["username"], "admin", "request_review", f"report_id={report_id}, link={link}")
        # (Optional) You could send a real email here if SMTP env is configured.
        return redirect(url_for("report_detail", report_id=report_id))
    except Exception as e:
        db.rollback()
        flash(f"Could not create review: {e}", "error")
        return redirect(url_for("report_detail", report_id=report_id))
    finally:
        db.close()

@app.route("/review/<token>", methods=["GET", "POST"])
def review_decide(token):
    db = SessionLocal()
    rr = db.query(Review).filter(Review.token == token).one_or_none()
    if not rr:
        db.close()
        return render_template_string("<h3>Invalid or expired review link.</h3>"), 404

    if request.method == "POST":
        decision = request.form.get("decision")
        note = request.form.get("note", "").strip()
        if decision not in {"approve", "reject"}:
            db.close()
            return render_template_string("<h3>Invalid decision.</h3>"), 400

        rr.status = "approved" if decision == "approve" else "rejected"
        rr.acted_by = rr.reviewer_email
        rr.acted_at = datetime.utcnow()
        rr.note = note
        db.commit()
        db.close()
        return render_template_string(
            "<h3>Thanks! You have {{status}} this report.</h3>",
        , status=rr.status)

    # GET: show a tiny inline form (no extra template file needed)
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Review Report</title>
        <style>
          body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial;margin:40px;color:#0f2547}
          .box{max-width:560px;margin:0 auto;border:1px solid #e6eef9;border-radius:12px;padding:18px;background:#fff}
          h1{margin:0 0 8px;font-size:1.4rem}
          p.muted{color:#5f7a99;margin:6px 0 16px}
          textarea{width:100%;min-height:100px;padding:10px;border:1px solid #e6eef9;border-radius:8px}
          .row{display:flex;gap:10px;margin-top:12px}
          .button{appearance:none;border:0;border-radius:10px;padding:10px 14px;font-weight:700;cursor:pointer}
          .ok{background:#1f6feb;color:#fff}
          .no{background:#d12b2b;color:#fff}
        </style>
      </head>
      <body>
        <div class="box">
          <h1>Review Report</h1>
          <p class="muted">Reviewer: {{email}} &middot; Status: <strong>{{status}}</strong></p>
          <form method="post">
            <label>Add a note (optional)</label>
            <textarea name="note" placeholder="Short note…"></textarea>
            <div class="row">
              <button class="button ok" name="decision" value="approve" type="submit">Approve</button>
              <button class="button no" name="decision" value="reject" type="submit">Reject</button>
            </div>
          </form>
        </div>
      </body>
    </html>
    """
    out = render_template_string(html, email=rr.reviewer_email, status=rr.status)
    db.close()
    return out

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
