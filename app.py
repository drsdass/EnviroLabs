import os
import io
import re
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

# Optional writer for XLSX output
import xlsxwriter

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
    except Exception:
        db.rollback()
    finally:
        db.close()

def parse_date(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    if pd.isna(val):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(val), fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None

# ------- Fuzzy header normalization -------
def _norm(s: str) -> str:
    """normalize a header: lowercase, strip, remove non-alphanumerics"""
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())

# Infer common variant column names (expanded)
COLUMN_ALIASES = {
    # Treat "Sample ID" and "Lab ID" as the same
    "lab_id": [
        "lab_id","lab id","id","labid","accession","accession_id","accession id",
        "sample","sampleid","sample id","sample_no","sampleno","sample number","sample#",
        "report id","report_id","case id","caseid","job number","job","identifier"
    ],
    "client": [
        "client","client_name","client name","account","account name","facility",
        "facility name","customer","customer name","company","submitter"
    ],
    "patient_name": ["patient","patient_name","patient name","name"],
    "test": ["test","panel","assay"],
    "result": ["result","final_result","final result","outcome"],
    "collected_date": ["collected_date","collection_date","collected","collection date"],
    "resulted_date": ["resulted_date","reported_date","reported date","finalized","result_date","result date"],
    "pdf_url": ["pdf","pdf_url","report_link","report link"],
}

def get_col(df, logical_name):
    """Fuzzy column resolver using aliases."""
    targets = {_norm(x) for x in COLUMN_ALIASES.get(logical_name, [])}
    # exact normalized match
    for c in df.columns:
        if _norm(c) in targets:
            return c
    # loose: target substring inside normalized header
    for c in df.columns:
        nc = _norm(c)
        if any(t in nc for t in targets):
            return c
    return None

def load_tabular_file(path: str, filename: str) -> pd.DataFrame:
    """
    Robustly load CSV or Excel into a DataFrame.
    - CSV/TSV/TXT: sniff delimiter, handle UTF-8 BOM, try a couple encodings.
    - XLSX/XLSM/XLT*: use openpyxl via pandas.
    - XLS: not supported by default (xlrd no longer reads xls by default).
    """
    ext = (os.path.splitext(filename or "")[1] or "").lstrip(".").lower()

    # CSV-like
    if ext in {"csv", "tsv", "txt"}:
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(
                    path,
                    sep=None,        # sniff delimiter
                    engine="python", # needed for sep=None
                    encoding=enc,
                    dtype=str,       # keep strings; we parse when needed
                )
            except Exception as e:
                last_err = e
                continue
        raise last_err or ValueError("Unable to parse CSV/TSV/TXT file.")

    # Modern Excel
    if ext in {"xlsx", "xlsm", "xltx", "xltm"}:
        return pd.read_excel(path, engine="openpyxl", dtype=str)

    # Legacy Excel
    if ext == "xls":
        raise ValueError("Legacy .xls is not supported. Please upload .xlsx or .csv.")

    # Unknown extension: try CSV first, then Excel explicitly
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    except Exception:
        return pd.read_excel(path, engine="openpyxl", dtype=str)

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
    parse_path = saved_path

    # Parse uploaded file (CSV or Excel) using a robust loader
    try:
        df = load_tabular_file(parse_path, filename)
    except Exception as e:
        flash(f"Could not read file: {e}", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    c_lab_id = get_col(df, "lab_id")
    c_client = get_col(df, "client")

    if not c_lab_id:
        flash("CSV must include a Lab ID column (e.g., Lab ID / Sample ID / Accession).", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))
    if not c_client:
        flash(f"No client column found; using default client '{CLIENT_NAME}'.", "info")

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
            lab_id_val = str(row[c_lab_id]).strip()
            if lab_id_val == "" or lab_id_val.lower() == "nan":
                continue
            client_val = str(row[c_client]).strip() if c_client else CLIENT_NAME

            # find existing
            existing = db.query(Report).filter(Report.lab_id == lab_id_val).one_or_none()
            if not existing:
                existing = Report(
                    lab_id = lab_id_val,
                    client = client_val
                )
                db.add(existing)
                created += 1
            else:
                updated += 1

            if c_patient:   existing.patient_name  = None if pd.isna(row[c_patient])  else str(row[c_patient])
            if c_test:      existing.test          = None if pd.isna(row[c_test])     else str(row[c_test])
            if c_result:    existing.result        = None if pd.isna(row[c_result])   else str(row[c_result])
            if c_collected: existing.collected_date = parse_date(row[c_collected])
            if c_resulted:  existing.resulted_date  = parse_date(row[c_resulted])
            if c_pdf:       existing.pdf_url       = None if pd.isna(row[c_pdf])      else str(row[c_pdf])

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

# --------------- BUILD REPORT (Admin) ---------------
def _detect_sample_id_col(df: pd.DataFrame) -> str | None:
    return get_col(df, "lab_id")

def _row_by_sample_id(df: pd.DataFrame, sample_id: str) -> pd.Series | None:
    col = _detect_sample_id_col(df)
    if not col:
        return None
    mask = df[col].astype(str).str.strip().str.casefold() == str(sample_id).strip().casefold()
    if not mask.any():
        return None
    return df[mask].iloc[0]

# Heuristic getter for a field from TotalProducts row by common header name patterns
def _get_by_alias(row: pd.Series, aliases: list[str]) -> str | None:
    if row is None:
        return None
    columns = row.index.tolist()
    for c in columns:
        if _norm(c) in {_norm(a) for a in aliases}:
            val = row[c]
            return None if pd.isna(val) else str(val)
    # loose contains
    for c in columns:
        nc = _norm(c)
        if any(_norm(a) in nc for a in aliases):
            val = row[c]
            return None if pd.isna(val) else str(val)
    return None

# Try to populate A17/D17/G17/H17 from TotalProducts row if headers are recognizable
TP_ALIASES = {
    "A17": ["product","product name","item","sample name","name"],
    "D17": ["brand","manufacturer","company","producer"],
    "G17": ["matrix","material","substrate","category","sample type"],
    "H17": ["analyte","target","compound","analyte name"],
    "D28": ["m","value m","limit","lims","regulatory limit","reporting limit","threshold"]
}

def _to_number_or_none(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return float(m.group(0)) if m else None

def build_report_xlsx(sample_id: str, total_products: pd.DataFrame, data_consolidator: pd.DataFrame) -> bytes:
    """
    Create an XLSX with:
      - Sheet 'Report': E17=sample_id, D12=today, and best-effort values pulled from TotalProducts
      - Sheet 'TotalProducts': full data
      - Sheet 'Data_Consolidator': full data
    """
    # Normalize columns
    total_products = total_products.copy()
    data_consolidator = data_consolidator.copy()
    total_products.columns = [str(c).strip() for c in total_products.columns]
    data_consolidator.columns = [str(c).strip() for c in data_consolidator.columns]

    # Find the row in TotalProducts for sample_id (E17)
    tp_row = _row_by_sample_id(total_products, sample_id)

    # Prepare workbook in memory
    out = io.BytesIO()
    wb = xlsxwriter.Workbook(out, {'in_memory': True})
    ws_report = wb.add_worksheet("Report")

    # Basic formats
    fmt_bold = wb.add_format({'bold': True})
    fmt_date = wb.add_format({'num_format': 'yyyy-mm-dd'})
    fmt_text = wb.add_format({'text_wrap': True})

    # Put some headers for clarity (non-critical)
    ws_report.write("B10", "Generated Report", fmt_bold)
    ws_report.write("B11", "Note: Cells mirror your Google Sheet layout where feasible.")

    # D12 = today
    ws_report.write_datetime("D12", datetime.utcnow(), fmt_date)
    # E17 = Sample ID
    ws_report.write("E17", sample_id)

    # Try to fill A17/D17/G17/H17 from TotalProducts row using header aliases
    if tp_row is not None:
        a17 = _get_by_alias(tp_row, TP_ALIASES["A17"])
        d17 = _get_by_alias(tp_row, TP_ALIASES["D17"])
        g17 = _get_by_alias(tp_row, TP_ALIASES["G17"])
        h17 = _get_by_alias(tp_row, TP_ALIASES["H17"])
        d28 = _get_by_alias(tp_row, TP_ALIASES["D28"])

        if a17: ws_report.write("A17", a17)
        if d17: ws_report.write("D17", d17)
        if g17: ws_report.write("G17", g17)
        if h17: ws_report.write("H17", h17)
        if d28:
            # If numeric-ish, write number; else text
            num = _to_number_or_none(d28)
            if num is not None:
                ws_report.write_number("D28", num)
            else:
                ws_report.write("D28", d28)

    # You can extend here to compute more cells (Method Blank / Matrix Spike, etc.)
    # using data_consolidator with the same approach as your Google Sheet filters.

    # Dump source sheets for transparency / downstream formulas
    def write_df(sheet_name: str, df: pd.DataFrame):
        ws = wb.add_worksheet(sheet_name)
        # headers
        for j, col in enumerate(df.columns):
            ws.write(0, j, str(col), fmt_bold)
        # rows
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            for j, col in enumerate(df.columns):
                val = row[col]
                if pd.isna(val):
                    ws.write(i, j, "")
                else:
                    # keep as text so nothing gets unintentionally coerced
                    ws.write(i, j, str(val))

    write_df("TotalProducts", total_products)
    write_df("Data_Consolidator", data_consolidator)

    wb.close()
    out.seek(0)
    return out.getvalue()

@app.route("/build_report", methods=["GET", "POST"])
@require_login(role="admin")
def build_report():
    if request.method == "GET":
        # inline minimal form to avoid template dependency
        return """
        <div style="max-width:720px;margin:40px auto;font-family:Inter,system-ui,Arial;">
          <h2>Build Report (Admin)</h2>
          <form method="post" enctype="multipart/form-data" style="display:grid;gap:12px;">
            <label>Sample ID (E17):
              <input name="sample_id" required style="width:100%;padding:8px;" />
            </label>
            <label>TotalProducts (CSV/XLSX):
              <input type="file" name="total_products" accept=".csv,.xlsx,.xlsm,.xltx,.xltm" required />
            </label>
            <label>Data_Consolidator (CSV/XLSX):
              <input type="file" name="data_consolidator" accept=".csv,.xlsx,.xlsm,.xltx,.xltm" required />
            </label>
            <label>Report Template (CSV - optional):
              <input type="file" name="report_template" accept=".csv" />
            </label>
            <div>
              <button type="submit" style="padding:10px 16px;">Build Workbook</button>
              <a href="/dashboard" style="margin-left:12px;">Back to Dashboard</a>
            </div>
          </form>
          <p style="margin-top:12px;color:#666">
            Tip: You can upload just <b>TotalProducts</b> and <b>Data_Consolidator</b>.
            The builder will place Sample ID in E17, today in D12, and try to fill A17/D17/G17/H17/D28 from TotalProducts.
          </p>
        </div>
        """

    # POST: build the workbook
    sample_id = request.form.get("sample_id", "").strip()
    if not sample_id:
        flash("Please provide a Sample ID.", "error")
        return redirect(url_for("build_report"))

    tp_file = request.files.get("total_products")
    dc_file = request.files.get("data_consolidator")

    if not tp_file or tp_file.filename == "":
        flash("TotalProducts file is required.", "error")
        return redirect(url_for("build_report"))
    if not dc_file or dc_file.filename == "":
        flash("Data_Consolidator file is required.", "error")
        return redirect(url_for("build_report"))

    # Save to disk temporarily
    tp_name = secure_filename(tp_file.filename)
    dc_name = secure_filename(dc_file.filename)
    tp_path = os.path.join(UPLOAD_FOLDER, tp_name)
    dc_path = os.path.join(UPLOAD_FOLDER, dc_name)
    tp_file.save(tp_path)
    dc_file.save(dc_path)

    try:
        tp_df = load_tabular_file(tp_path, tp_name)
        dc_df = load_tabular_file(dc_path, dc_name)
    except Exception as e:
        flash(f"Could not read uploaded files: {e}", "error")
        # clean up
        for p in (tp_path, dc_path):
            if os.path.exists(p) and not KEEP_UPLOADED_CSVS:
                os.remove(p)
        return redirect(url_for("build_report"))

    # Optional: template CSV (currently ignored in logic; future: cell-level overrides)
    rt_df = None
    rt_file = request.files.get("report_template")
    rt_name = None
    rt_path = None
    if rt_file and rt_file.filename:
        rt_name = secure_filename(rt_file.filename)
        rt_path = os.path.join(UPLOAD_FOLDER, rt_name)
        rt_file.save(rt_path)
        try:
            rt_df = load_tabular_file(rt_path, rt_name)  # will read CSV as a table (not used yet)
        except Exception:
            # Not fatal
            flash("Report Template CSV could not be parsed; continuing without it.", "info")

    # Build report workbook
    try:
        xlsx_bytes = build_report_xlsx(sample_id, tp_df, dc_df)
    except Exception as e:
        flash(f"Report build failed: {e}", "error")
        xlsx_bytes = None

    # Clean up temp files if desired
    for p in (tp_path, dc_path, rt_path):
        if p and os.path.exists(p) and not KEEP_UPLOADED_CSVS:
            try:
                os.remove(p)
            except Exception:
                pass

    if not xlsx_bytes:
        return redirect(url_for("build_report"))

    log_action(session.get("username","admin"), "admin", "build_report", f"Built report for {sample_id}")
    return send_file(
        io.BytesIO(xlsx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"EnviroLabs_Report_{sample_id}.xlsx"
    )

# ----------- Minimal health check for Render -----------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
