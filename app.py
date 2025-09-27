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
import re

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
    patient_name = Column(String, nullable=True)  # (used as Sample Name slot)
    test = Column(String, nullable=True)          # e.g., "Bisphenol S", "PFAS"
    result = Column(String, nullable=True)        # text or numeric-as-text
    collected_date = Column(Date, nullable=True)  # Received Date
    resulted_date = Column(Date, nullable=True)   # Reported Date
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
    # Try common formats first
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # Fallback to pandas
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        # pandas can return Timestamp; pull date
        return ts.date()
    except Exception:
        return None

def _norm_words(colname: str):
    # Normalize to a list of "words"
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(colname))
    return [w for w in cleaned.split() if w]

def _header_contains(col: str, tokens: list[str]) -> bool:
    words = _norm_words(col)
    return all(tok in words for tok in tokens)

def _find_col_tokenized(df: pd.DataFrame, aliases: list[list[str]], fallback_idx: int|None=None):
    # First pass: exact token inclusion match
    for tokens in aliases:
        for c in df.columns:
            if _header_contains(c, tokens):
                return c
    # Second pass: loose substring of token-joined
    for tokens in aliases:
        needle = " ".join(tokens)
        for c in df.columns:
            if needle in " ".join(_norm_words(c)):
                return c
    # Fallback by index if provided and in range
    if fallback_idx is not None:
        cols = list(df.columns)
        if 0 <= fallback_idx < len(cols):
            return cols[fallback_idx]
    return None

def _reheader_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master Upload File often has:
      Row 1: section banner like 'CLIENT INFORMATION' / 'SAMPLE SUMMARY' / 'SAMPLE RESULTS'
      Row 2: actual headers
      Row 3+: data
    If we detect a banner in the original columns, promote row 2 to header and drop the first two rows.
    """
    original_cols = [str(c).strip() for c in df.columns]
    # If any original column mentions the section names, it's probably the banner-row format
    banner_hit = any(
        any(tag in c.lower() for tag in ("client information", "sample summary", "sample results"))
        for c in original_cols
    )
    if not banner_hit:
        # Some CSV writers put unnamed columns like "Unnamed: 0" AND the first data row contains those banners.
        # If the first row includes the banners, also treat as banner format.
        first_row_str = " ".join(str(x) for x in df.iloc[0].tolist()) if len(df) else ""
        if any(tag in first_row_str.lower() for tag in ("client information", "sample summary", "sample results")):
            banner_hit = True

    if banner_hit and len(df) >= 2:
        # Use second row (index=1) as header, then keep from row index=2 onward
        new_header = [str(x).strip() for x in df.iloc[1].tolist()]
        df2 = df.iloc[2:].copy()
        df2.columns = new_header
        return df2.reset_index(drop=True)

    # Otherwise, just strip current headers
    df.columns = [str(c).strip() for c in df.columns]
    return df

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

    p = build_payload_from_master_cache(r.lab_id, r)
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

    # Decide parser by extension, but also try both
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

    # Normalize headers or promote row 2 to headers if needed
    df = _reheader_if_needed(df)

    # Detect Master Upload File signature (has these critical fields after reheader)
    looks_like_master = any("sample id" in " ".join(_norm_words(c)) for c in df.columns) and \
                        any("client" in " ".join(_norm_words(c)) for c in df.columns)

    if looks_like_master:
        # Write a slim cache for the Client Info (ensures Phone/Email/Lead/Address map correctly)
        _write_master_cache_if_possible(df)

    # ---- DB upsert (same behavior you had; we keep it simple) ----
    # Identify Lab ID / Client for DB records
    c_lab = _find_col_tokenized(
        df,
        aliases=[["lab","id"], ["laboratory","id"], ["sample","id"], ["sample"], ["accession","id"]]
    )
    if not c_lab:
        # try very literal "Sample ID (Lab ID, Laboratory ID)" phrasing
        for c in df.columns:
            if "sample id" in " ".join(_norm_words(c)):
                c_lab = c
                break

    c_client = _find_col_tokenized(df, aliases=[["client"], ["client","name"], ["facility"], ["account"]])

    if not c_lab or not c_client:
        preview = ", ".join(df.columns[:20])
        flash("CSV must include Lab ID (aka 'Sample ID') and Client columns "
              f"(various names accepted). Found columns: {preview}", "error")
        if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Optional fields that sometimes exist in Master file
    c_sample_name = _find_col_tokenized(df, aliases=[["sample","name"], ["product","name"]])
    c_analyte     = _find_col_tokenized(df, aliases=[["analyte"]])
    c_result      = _find_col_tokenized(df, aliases=[["result"]])
    c_received    = _find_col_tokenized(df, aliases=[["received","date"], ["collected","date"]])
    c_reported    = _find_col_tokenized(df, aliases=[["reported"], ["resulted","date"], ["reported","date"]])

    created, updated = 0, 0
    skipped_non_numeric_lab = 0

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            raw_lab = row.get(c_lab, "")
            lab_id = "" if pd.isna(raw_lab) else str(raw_lab).strip()
            if not lab_id:
                continue

            # Only create/update DB rows for Lab IDs that start with a digit (skip QC names)
            if not re.match(r"^\d", lab_id):
                skipped_non_numeric_lab += 1
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

            # Optional fields
            if c_sample_name:
                val = row.get(c_sample_name)
                existing.patient_name = None if pd.isna(val) else str(val)

            if c_analyte:
                val = row.get(c_analyte)
                existing.test = None if pd.isna(val) else str(val)

            if c_result:
                val = row.get(c_result)
                existing.result = None if pd.isna(val) else str(val)

            if c_received:
                existing.collected_date = parse_date(row.get(c_received))
            if c_reported:
                existing.resulted_date = parse_date(row.get(c_reported))

        db.commit()
        flash(
            f"Imported {created} new and updated {updated} report(s). "
            f"Skipped {skipped_non_numeric_lab} non-numeric Lab ID row(s).",
            "success"
        )
        log_action(u["username"], u["role"], "upload_csv",
                   f"{filename} -> created {created}, updated {updated}, skipped_non_numeric={skipped_non_numeric_lab}")
    except Exception as e:
        db.rollback()
        flash(f"Import failed: {e}", "error")
    finally:
        db.close()

    if os.path.exists(saved_path) and (not KEEP_UPLOADED_CSVS or not keep):
        os.remove(saved_path)

    return redirect(url_for("dashboard"))

# ----------- Master cache helpers -----------
def _write_master_cache_if_possible(df: pd.DataFrame):
    """
    Recognize the 'Master Upload File' and write a compact cache with:
    Sample ID, Client, Phone (C), Email (D), Project Lead (E), Address (F)
    """
    if df is None or df.empty:
        return

    df = _reheader_if_needed(df).fillna("")

    # Identify key headers by tokens
    c_lab = _find_col_tokenized(
        df,
        aliases=[["sample","id"], ["lab","id"], ["laboratory","id"], ["sample"]],
        fallback_idx=0
    )
    c_client = _find_col_tokenized(df, aliases=[["client"], ["client","name"], ["facility"]])

    # These 4 are sometimes hard to detect by tokens in your file; use token first, then position fallback
    c_phone = _find_col_tokenized(df, aliases=[["phone"], ["phone","number"]])
    c_email = _find_col_tokenized(df, aliases=[["email"], ["e","mail"]])
    c_lead  = _find_col_tokenized(df, aliases=[["project","lead"], ["project","manager"], ["project"]])
    c_addr  = _find_col_tokenized(df, aliases=[["address"], ["addr"]])

    # Position fallback AFTER reheader: A=Sample ID, B=Client, C=Phone, D=Email, E=Project Lead, F=Address
    cols = list(df.columns)
    if not c_phone and len(cols) > 2: c_phone = cols[2]   # Col C
    if not c_email and len(cols) > 3: c_email = cols[3]   # Col D
    if not c_lead  and len(cols) > 4: c_lead  = cols[4]   # Col E
    if not c_addr  and len(cols) > 5: c_addr  = cols[5]   # Col F

    # Must have at least Sample ID and Client
    if not c_lab or not c_client:
        return

    out = pd.DataFrame({
        "Sample ID": df[c_lab].astype(str).str.strip(),
        "Client":    df[c_client].astype(str).str.strip(),
        "Phone":     (df[c_phone].astype(str).str.strip() if c_phone else ""),
        "Email":     (df[c_email].astype(str).str.strip() if c_email else ""),
        "Project Lead": (df[c_lead].astype(str).str.strip() if c_lead else ""),
        "Address":   (df[c_addr].astype(str).str.strip() if c_addr else ""),
    })
    out = out[out["Sample ID"] != ""]
    if not out.empty:
        out.to_csv(MASTER_CACHE_PATH, index=False)

def build_payload_from_master_cache(lab_id: str, r: Report):
    """
    Build the payload for the report page.
    We load client info (Phone, Email, Project Lead, Address) from the cached Master csv when available.
    Other fields keep your previous behavior (use DB values or remain blank).
    """
    # Default skeleton
    p = {
        "client_info": {
            "client": r.client or "",
            "phone": "",
            "email": "support@envirolabsusa.com",  # fallback to company email
            "project_lead": "",
            "address": ""
        },
        "sample_summary": {
            "reported": r.resulted_date.isoformat() if r.resulted_date else "",
            "received_date": r.collected_date.isoformat() if r.collected_date else "",
            "sample_name": r.patient_name or r.lab_id or "",
            "prepared_by": "",
            "matrix": "",
            "prepared_date": "",
            "qualifiers": "",
            "asin": "",
            "product_weight_g": ""
        },
        "sample_results": {
            "analyte": r.test or "",
            "result": r.result or "",
            "mrl": "",
            "units": "",
            "dilution": "",
            "analyzed": "",
            "qualifier": ""
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

    # Try to enrich from master cache
    try:
        if os.path.exists(MASTER_CACHE_PATH):
            m = pd.read_csv(MASTER_CACHE_PATH).fillna("")
            # normalize headers for safety
            m.columns = [str(c).strip() for c in m.columns]
            # Prefer exact match on Sample ID
            row = m.loc[m["Sample ID"].astype(str).str.strip() == str(lab_id).strip()]
            if not row.empty:
                row = row.iloc[0]
                # Override client if present
                if str(row.get("Client", "")).strip():
                    p["client_info"]["client"] = str(row.get("Client")).strip()
                # Phone/Email/Lead/Address from cache
                p["client_info"]["phone"] = str(row.get("Phone", "")).strip()
                # If the CSV email is empty, keep company email fallback
                csv_email = str(row.get("Email", "")).strip()
                if csv_email:
                    p["client_info"]["email"] = csv_email
                p["client_info"]["project_lead"] = str(row.get("Project Lead", "")).strip()
                p["client_info"]["address"] = str(row.get("Address", "")).strip()
    except Exception:
        # If anything goes wrong, silently keep defaults (DB + fallbacks).
        pass

    return p

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

# ----------- Health & Errors -----------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Not found"), 404

@app.errorhandler(500)
def server_error(e):
    # Keep generic so we don't leak details to users
    return render_template("error.html", code=500, message="Internal Server Error"), 500

# ------------------- Main -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
