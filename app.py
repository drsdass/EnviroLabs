import os
import io
import csv
import json
import hashlib
from datetime import datetime, date
from typing import Optional, Dict, Any, List

import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_file, abort
)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, current_user, login_required
)
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# ---- Optional formula hooks (keep app running if file is missing) ----
try:
    from formula_hooks import compute_fields  # type: ignore
except Exception:  # pragma: no cover
    def compute_fields(_row: dict) -> dict:
        # Safe fallback if hooks are not present
        return {}

# ------------------ Config ------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")

CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = os.getenv("KEEP_UPLOADED_CSVS", "true").lower() in {"1", "true", "yes", "y"}

# ------------------ Flask app ------------------
app = Flask(__name__)
app.config.update(SECRET_KEY=SECRET_KEY, MAX_CONTENT_LENGTH=25 * 1024 * 1024)

# ------------------ Database ------------------
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, index=True, nullable=False)
    client = Column(String, index=True, nullable=False)
    patient_name = Column(String, nullable=True)
    test = Column(String, nullable=True)
    result = Column(String, nullable=True)
    collected_date = Column(Date, nullable=True)
    resulted_date = Column(Date, nullable=True)
    pdf_url = Column(String, nullable=True)  # optional link to PDF
    computed = Column(Text, nullable=True)   # JSON of derived fields

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Audit(Base):
    __tablename__ = "audits"
    id = Column(Integer, primary_key=True)
    user = Column(String, index=True)
    action = Column(String)
    detail = Column(Text)
    ip = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)

# Lightweight migration: ensure columns exist if upgrading from older app
with engine.begin() as conn:
    try:
        cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(reports)").fetchall()]
        if "computed" not in cols:
            conn.exec_driver_sql("ALTER TABLE reports ADD COLUMN computed TEXT")
    except Exception:
        pass

# ------------------ Auth ------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, username: str, role: str):
        self.username = username
        self.role = role

    def get_id(self) -> str:  # type: ignore[override]
        return f"{self.role}:{self.username}"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_client(self) -> bool:
        return self.role == "client"


# Static users (you can swap to a real user table later)
USERS = {
    "admin": {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
    "client": {"username": CLIENT_USERNAME, "password": CLIENT_PASSWORD},
}


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    try:
        role, username = user_id.split(":", 1)
    except ValueError:
        return None
    if role in USERS and USERS[role]["username"] == username:
        return User(username, role)
    return None


# ------------------ Helpers ------------------

def audit(action: str, detail: str = "") -> None:
    try:
        sess = SessionLocal()
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?")
        who = current_user.get_id() if current_user.is_authenticated else "anon"
        rec = Audit(user=who, action=action, detail=detail[:4000], ip=ip)
        sess.add(rec)
        sess.commit()
    except Exception:
        sess.rollback()
    finally:
        sess.close()


def parse_date(val: Any) -> Optional[date]:
    if val is None:
        return None
    # pandas Timestamp or datetime/date
    if isinstance(val, (pd.Timestamp, datetime, date)):
        try:
            return val.date() if isinstance(val, (pd.Timestamp, datetime)) else val
        except Exception:
            return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # let pandas try
    try:
        x = pd.to_datetime(s, errors="coerce")
        if pd.isna(x):
            return None
        return x.date()
    except Exception:
        return None


def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        k = cand.lower()
        if k in lower:
            return lower[k]
    # fuzzy: strip spaces and punctuation
    simp = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in cols}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in simp:
            return simp[key]
    return None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv", "xlsx", "xls"}


# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("index.html", user=current_user)


@app.route("/login", methods=["GET", "POST"])
def login():
    role = request.args.get("role", "client")
    if request.method == "POST":
        role = request.form.get("role", role)
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if role not in USERS:
            flash("Invalid role.", "error")
            return redirect(url_for("login", role=role))
        good = (username == USERS[role]["username"]) and (password == USERS[role]["password"])
        if good:
            login_user(User(username, role))
            audit("login", f"role={role}")
            if role == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("client_dashboard"))
        flash("Invalid credentials.", "error")
    return render_template("login.html", role=role)


@app.route("/logout")
@login_required
def logout():
    audit("logout")
    logout_user()
    return redirect(url_for("index"))


# ------------- Admin -------------
@app.route("/admin")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)
    sess = SessionLocal()
    q = sess.query(Report)

    # filters
    lab_id = request.args.get("lab_id", "").strip()
    client = request.args.get("client", "").strip()
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")

    if lab_id:
        q = q.filter(Report.lab_id.like(f"%{lab_id}%"))
    if client:
        q = q.filter(Report.client.like(f"%{client}%"))

    if date_from:
        dfrom = parse_date(date_from)
        if dfrom:
            q = q.filter(Report.resulted_date >= dfrom)
    if date_to:
        dto = parse_date(date_to)
        if dto:
            q = q.filter(Report.resulted_date <= dto)

    q = q.order_by(Report.resulted_date.desc().nullslast(), Report.created_at.desc())
    rows = q.limit(500).all()
    sess.close()
    return render_template("admin_dashboard.html", user=current_user, rows=rows)


@app.route("/admin/upload", methods=["POST"]) 
@login_required
def upload_csv():
    if not current_user.is_admin:
        abort(403)
    if "file" not in request.files:
        flash("No file part.", "error")
        return redirect(url_for("admin_dashboard"))
    f = request.files["file"]
    if not f or f.filename == "":
        flash("No selected file.", "error")
        return redirect(url_for("admin_dashboard"))
    if not allowed_file(f.filename):
        flash("Unsupported file type. Use CSV/XLSX.", "error")
        return redirect(url_for("admin_dashboard"))

    # Persist upload if requested
    saved_path = None
    if KEEP_UPLOADED_CSVS:
        saved_path = os.path.join(UPLOAD_DIR, f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}_{f.filename}")
        f.save(saved_path)
    else:
        # read into memory
        data = f.read()

    # Read into pandas
    try:
        if KEEP_UPLOADED_CSVS:
            if f.filename.lower().endswith(".csv"):
                df = pd.read_csv(saved_path, dtype=str, keep_default_na=True)
            else:
                df = pd.read_excel(saved_path, dtype=str)
        else:
            if f.filename.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=True)
            else:
                df = pd.read_excel(io.BytesIO(data), dtype=str)
    except Exception as e:
        flash(f"Failed to parse file: {e}", "error")
        return redirect(url_for("admin_dashboard"))

    df.columns = [str(c).strip() for c in df.columns]

    # Map common headers (case-insensitive)
    c_lab_id = find_col(df.columns.tolist(), [
        "Lab ID", "LabID", "Sample ID", "Document #", "Document Id", "Sample", "ID"
    ])
    if not c_lab_id:
        flash("CSV must include a Lab ID column.", "error")
        return redirect(url_for("admin_dashboard"))

    c_client = find_col(df.columns.tolist(), ["Client", "Client Name", "Account", "Customer"]) 
    c_patient = find_col(df.columns.tolist(), ["Patient", "Patient Name", "Name"]) 
    c_test = find_col(df.columns.tolist(), ["Test", "Analyte", "Panel", "Assay"]) 
    c_result = find_col(df.columns.tolist(), ["Result", "Results", "Value", "Outcome", "Interpretation"]) 
    c_collected = find_col(df.columns.tolist(), ["Collected Date", "Collection Date", "Collected", "Sample Date"]) 
    c_resulted = find_col(df.columns.tolist(), ["Resulted Date", "Reported Date", "Result Date", "Reported"]) 
    c_pdf = find_col(df.columns.tolist(), ["PDF URL", "PDF", "Report URL", "Link"]) 

    sess = SessionLocal()
    created = 0
    updated = 0

    try:
        for _, row in df.iterrows():
            lab_id = str(row[c_lab_id]).strip() if pd.notna(row[c_lab_id]) else ""
            if not lab_id:
                continue
            client_val = str(row[c_client]).strip() if (c_client and pd.notna(row[c_client])) else CLIENT_NAME

            # Raw row dict for formula hooks (preserve original column names)
            raw_row = {str(k): (None if (pd.isna(row[k])) else row[k]) for k in df.columns}
            try:
                computed_fields = compute_fields(raw_row) or {}
            except Exception as e:
                computed_fields = {"_formula_error": str(e)[:300]}

            existing = sess.query(Report).filter(Report.lab_id == lab_id).first()
            if existing:
                if c_patient: existing.patient_name = None if pd.isna(row[c_patient]) else str(row[c_patient])
                if c_test: existing.test = None if pd.isna(row[c_test]) else str(row[c_test])
                if c_result: existing.result = None if pd.isna(row[c_result]) else str(row[c_result])
                if c_collected: existing.collected_date = parse_date(row[c_collected])
                if c_resulted: existing.resulted_date = parse_date(row[c_resulted])
                if c_pdf: existing.pdf_url = None if (pd.isna(row[c_pdf])) else str(row[c_pdf])
                existing.client = client_val
                try:
                    existing.computed = json.dumps(computed_fields, ensure_ascii=False)
                except Exception:
                    existing.computed = json.dumps({"_formula_error": "json-encode-failed"})
                updated += 1
            else:
                rec = Report(
                    lab_id=lab_id,
                    client=client_val,
                    patient_name=(None if not c_patient or pd.isna(row[c_patient]) else str(row[c_patient])),
                    test=(None if not c_test or pd.isna(row[c_test]) else str(row[c_test])),
                    result=(None if not c_result or pd.isna(row[c_result]) else str(row[c_result])),
                    collected_date=(parse_date(row[c_collected]) if c_collected else None),
                    resulted_date=(parse_date(row[c_resulted]) if c_resulted else None),
                    pdf_url=(None if not c_pdf or pd.isna(row[c_pdf]) else str(row[c_pdf])),
                    computed=json.dumps(computed_fields, ensure_ascii=False),
                )
                sess.add(rec)
                created += 1
        sess.commit()
        audit("csv_import", f"created={created}, updated={updated}, file='{f.filename}'")
        flash(f"Import complete. Created {created}, updated {updated} reports.", "success")
    except Exception as e:
        sess.rollback()
        flash(f"Import failed: {e}", "error")
    finally:
        sess.close()

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/audit")
@login_required
def admin_audit():
    if not current_user.is_admin:
        abort(403)
    sess = SessionLocal()
    rows = sess.query(Audit).order_by(Audit.created_at.desc()).limit(500).all()
    sess.close()
    return render_template("admin_audit.html", user=current_user, rows=rows)


@app.route("/admin/export")
@login_required
def admin_export():
    if not current_user.is_admin:
        abort(403)
    sess = SessionLocal()
    q = sess.query(Report)

    # same filters as dashboard
    lab_id = request.args.get("lab_id", "").strip()
    client = request.args.get("client", "").strip()
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")

    if lab_id:
        q = q.filter(Report.lab_id.like(f"%{lab_id}%"))
    if client:
        q = q.filter(Report.client.like(f"%{client}%"))
    if date_from:
        dfrom = parse_date(date_from)
        if dfrom:
            q = q.filter(Report.resulted_date >= dfrom)
    if date_to:
        dto = parse_date(date_to)
        if dto:
            q = q.filter(Report.resulted_date <= dto)

    rows = q.order_by(Report.resulted_date.desc().nullslast(), Report.created_at.desc()).all()
    sess.close()

    # Build CSV in memory, flattening computed fields
    output = io.StringIO()
    writer = csv.writer(output)

    # Collect all computed keys for header
    computed_keys = []
    for r in rows:
        try:
            comp = json.loads(r.computed) if r.computed else {}
            for k in comp.keys():
                if k not in computed_keys:
                    computed_keys.append(k)
        except Exception:
            if "_parse_error" not in computed_keys:
                computed_keys.append("_parse_error")

    header = [
        "Lab ID", "Client", "Patient", "Test", "Result",
        "Collected Date", "Resulted Date", "PDF URL"
    ] + [f"Computed: {k}" for k in computed_keys]
    writer.writerow(header)

    for r in rows:
        base = [
            r.lab_id, r.client, r.patient_name or "", r.test or "", r.result or "",
            r.collected_date.isoformat() if r.collected_date else "",
            r.resulted_date.isoformat() if r.resulted_date else "",
            r.pdf_url or "",
        ]
        comp_vals = []
        try:
            comp = json.loads(r.computed) if r.computed else {}
            for k in computed_keys:
                comp_vals.append(comp.get(k, ""))
        except Exception:
            comp_vals = ["parse_error" if k == "_parse_error" else "" for k in computed_keys]
        writer.writerow(base + comp_vals)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    filename = f"reports_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    audit("export_csv", f"rows={len(rows)}")
    return send_file(mem, as_attachment=True, download_name=filename, mimetype="text/csv")


# ------------- Client -------------
@app.route("/client")
@login_required
def client_dashboard():
    if not current_user.is_client:
        abort(403)
    sess = SessionLocal()
    q = sess.query(Report).filter(Report.client == CLIENT_NAME)

    lab_id = request.args.get("lab_id", "").strip()
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")

    if lab_id:
        q = q.filter(Report.lab_id.like(f"%{lab_id}%"))
    if date_from:
        dfrom = parse_date(date_from)
        if dfrom:
            q = q.filter(Report.resulted_date >= dfrom)
    if date_to:
        dto = parse_date(date_to)
        if dto:
            q = q.filter(Report.resulted_date <= dto)

    rows = q.order_by(Report.resulted_date.desc().nullslast(), Report.created_at.desc()).limit(500).all()
    sess.close()
    return render_template("client_dashboard.html", user=current_user, rows=rows, client_name=CLIENT_NAME)


# ------------- Reports -------------
@app.route("/report/<int:rid>")
@login_required
def report_detail(rid: int):
    sess = SessionLocal()
    r = sess.query(Report).get(rid)
    sess.close()
    if not r:
        abort(404)
    # Clients may only view their own records
    if current_user.is_client and r.client != CLIENT_NAME:
        abort(403)

    computed = {}
    try:
        if r.computed:
            computed = json.loads(r.computed)
    except Exception:
        computed = {"_parse_error": True}

    return render_template("report_detail.html", user=current_user, r=r, computed=computed)


# ------------------ Error handlers ------------------
@app.errorhandler(403)
def forbidden(_e):
    return render_template("error.html", code=403, message="Forbidden"), 403


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, message="Not found"), 404


# ------------------ CLI convenience ------------------
@app.cli.command("seed")
def seed():  # pragma: no cover
    """Optional: quick seed for local testing."""
    sess = SessionLocal()
    try:
        r = Report(
            lab_id="LAB-1001",
            client=CLIENT_NAME,
            patient_name="Jane Doe",
            test="PFAS Panel",
            result="Not Detected",
            collected_date=date(2024, 8, 1),
            resulted_date=date(2024, 8, 5),
            pdf_url="",
            computed=json.dumps({"Flag": "Not Detected", "PFAS_Sum": 0.0}),
        )
        sess.add(r)
        sess.commit()
        print("Seeded one report.")
    finally:
        sess.close()


if __name__ == "__main__":  # Local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
