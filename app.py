import os
import io
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
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

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())

# Aliases for fuzzy header resolution
COLUMN_ALIASES = {
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
    targets = {_norm(x) for x in COLUMN_ALIASES.get(logical_name, [])}
    for c in df.columns:
        if _norm(c) in targets:
            return c
    for c in df.columns:
        nc = _norm(c)
        if any(t in nc for t in targets):
            return c
    return None

def load_tabular_file(path: str, filename: str) -> pd.DataFrame:
    ext = (os.path.splitext(filename or "")[1] or "").lstrip(".").lower()
    if ext in {"csv","tsv","txt"}:
        last_err = None
        for enc in ("utf-8-sig","utf-8","latin-1"):
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
            except Exception as e:
                last_err = e
                continue
        raise last_err or ValueError("Unable to parse CSV-like file.")
    if ext in {"xlsx","xlsm","xltx","xltm"}:
        return pd.read_excel(path, engine="openpyxl", dtype=str)
    if ext == "xls":
        raise ValueError("Legacy .xls not supported. Please upload .xlsx or .csv.")
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

    try:
        df = load_tabular_file(parse_path, filename)
    except Exception as e:
        flash(f"Could not read file: {e}", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    df.columns = [str(c).strip() for c in df.columns]

    c_lab_id = get_col(df, "lab_id")
    c_client = get_col(df, "client")
    if not c_lab_id:
        flash("CSV must include a Lab ID / Sample ID column.", "error")
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

            existing = db.query(Report).filter(Report.lab_id == lab_id_val).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id_val, client=client_val)
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

# ---------------- Build Report (Admin) ----------------

# Aliases for fields we need from TotalProducts
TP_ALIASES = {
    "product": ["product","product name","item","sample name","name"],
    "brand":   ["brand","manufacturer","company","producer"],
    "matrix":  ["matrix","material","substrate","category","sample type"],
    "analyte": ["analyte","target","compound","analyte name"],
    "limitm":  ["m","value m","limit","lims","regulatory limit","reporting limit","threshold"]
}

# Aliases for Data_Consolidator
DC_ALIASES = {
    "record_name": ["name","record","sample name","run name","id string","descriptor","a"],
    "sample_id":   ["sample id","lab id","accession","sample","id","identifier"],
    "analyte":     ["analyte","compound","target","analyte name","b"],
    "result":      ["result","value","meas","concentration","c"],
    "units":       ["unit","units","uom","e"],
    "sheet":       ["sheet","sheet name","batch","run","g"],
}

def dc_get_col(df, logical):
    aliases = DC_ALIASES.get(logical, [])
    # try exact normalized match
    for c in df.columns:
        if _norm(c) in {_norm(a) for a in aliases}:
            return c
    # loose contains
    for c in df.columns:
        nc = _norm(c)
        if any(_norm(a) in nc for a in aliases):
            return c
    return None

def tp_get_by_alias(row: pd.Series, aliases: list[str]):
    if row is None:
        return None
    for c in row.index:
        if _norm(c) in {_norm(a) for a in aliases}:
            v = row[c]
            return None if pd.isna(v) else str(v)
    for c in row.index:
        nc = _norm(c)
        if any(_norm(a) in nc for a in aliases):
            v = row[c]
            return None if pd.isna(v) else str(v)
    return None

def find_tp_row_by_sample_id(tp_df: pd.DataFrame, sample_id: str) -> pd.Series | None:
    col = get_col(tp_df, "lab_id")
    if not col:
        return None
    mask = tp_df[col].astype(str).str.strip().str.casefold() == str(sample_id).strip().casefold()
    if not mask.any():
        return None
    return tp_df[mask].iloc[0]

def _pick_primary_analyte():
    # default preference order if both exist
    return ["Bisphenol S", "PFAS"]

def dc_select_rows(dc_df: pd.DataFrame, sample_id: str):
    """Return dict of interesting rows from Data_Consolidator for this sample_id."""
    # Normalize
    df = dc_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    c_rec = dc_get_col(df, "record_name")
    c_sid = dc_get_col(df, "sample_id")
    c_ana = dc_get_col(df, "analyte")
    c_res = dc_get_col(df, "result")
    c_uom = dc_get_col(df, "units")
    c_sht = dc_get_col(df, "sheet")

    # helper extractors
    def _val(row, col):
        if not col:
            return None
        v = row.get(col, None)
        return None if (v is None or pd.isna(v)) else str(v)

    # sample match: either explicit Sample ID column equals,
    # or record_name begins with Sample ID (prefix match like LEFT(...)=E17)
    def _is_sample_row(row) -> bool:
        sid = _val(row, c_sid)
        if sid and sid.strip().casefold() == sample_id.strip().casefold():
            return True
        rec = _val(row, c_rec) or ""
        return rec.strip().startswith(str(sample_id).strip())

    # filter to only rows for this sample_id
    f = df[df.apply(_is_sample_row, axis=1)].copy()

    # exclude obvious non-sample QC in "primary"
    def _is_blank_like(nm: str):
        s = (nm or "").lower()
        return ("blank" in s) or ("calibr" in s)

    def _is_spike_like(nm: str):
        s = (nm or "").lower()
        return ("spike" in s)

    # choose primary analyte preference
    target_order = _pick_primary_analyte()

    primary = None
    for analyte_name in target_order:
        g = f[f[c_ana].astype(str).str.strip().str.casefold() == analyte_name.strip().casefold()] if c_ana else f
        if c_rec:
            g = g[~g[c_rec].astype(str).str.lower().apply(_is_blank_like)]
            g = g[~g[c_rec].astype(str).str.lower().apply(_is_spike_like)]
        if len(g) > 0:
            primary = g.iloc[0]
            break

    # Method Blank / Matrix Spike 1 / Matrix Spike Duplicate
    def _first_contains(text):
        return (f[c_rec].astype(str).str.contains(text, case=False, na=False)).idxmax() if c_rec and f[c_rec].astype(str).str.contains(text, case=False, na=False).any() else None

    mb_idx = _first_contains("Method Blank")
    ms1_idx = _first_contains("Matrix Spike 1")
    msd_idx = _first_contains("Matrix Spike Duplicate")

    method_blank = f.loc[mb_idx] if mb_idx is not None else None
    ms1 = f.loc[ms1_idx] if ms1_idx is not None else None
    msd = f.loc[msd_idx] if msd_idx is not None else None

    return {
        "primary": primary,
        "method_blank": method_blank,
        "ms1": ms1,
        "msd": msd,
        "columns": {"rec": c_rec, "sid": c_sid, "ana": c_ana, "res": c_res, "uom": c_uom, "sht": c_sht}
    }

def _num_or_text(ws, cell, value, numfmt=None):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        ws.write(cell, "")
        return
    s = str(value)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if m:
        try:
            f = float(m.group(0))
            if numfmt:
                ws.write(cell, f, numfmt)
            else:
                ws.write_number(cell, f)
            return
        except Exception:
            pass
    ws.write(cell, s)

def draw_report_layout_and_fill(sample_id: str, tp_df: pd.DataFrame, dc_df: pd.DataFrame) -> bytes:
    # prepare data
    tp_row = find_tp_row_by_sample_id(tp_df, sample_id)
    sel = dc_select_rows(dc_df, sample_id)

    # workbook in memory
    out = io.BytesIO()
    wb = xlsxwriter.Workbook(out, {"in_memory": True})
    ws = wb.add_worksheet("Report")

    # --- Styles ---
    color_primary = "#133E7C"  # deep blue header
    color_accent  = "#E9F0FB"  # light blue fill
    color_border  = "#9FB3D1"

    f_title = wb.add_format({"bold": True, "font_size": 16, "font_color": "white", "align": "center", "valign": "vcenter", "bg_color": color_primary})
    f_subttl = wb.add_format({"bold": True, "font_size": 11, "font_color": color_primary})
    f_label = wb.add_format({"bold": True, "font_size": 10, "bg_color": color_accent, "border": 1, "border_color": color_border})
    f_value = wb.add_format({"font_size": 10, "border": 1, "border_color": color_border})
    f_small = wb.add_format({"font_size": 9})
    f_date  = wb.add_format({"font_size": 10, "border": 1, "border_color": color_border, "num_format": "yyyy-mm-dd"})
    f_th    = wb.add_format({"bold": True, "font_size": 10, "bg_color": color_accent, "border": 1, "border_color": color_border, "align": "center"})
    f_td    = wb.add_format({"font_size": 10, "border": 1, "border_color": color_border})
    f_center= wb.add_format({"font_size": 10, "border": 1, "border_color": color_border, "align": "center"})

    # column widths and row heights (A..I)
    ws.set_column("A:A", 18)
    ws.set_column("B:B", 16)
    ws.set_column("C:C", 16)
    ws.set_column("D:D", 16)
    ws.set_column("E:E", 20)
    ws.set_column("F:F", 16)
    ws.set_column("G:G", 16)
    ws.set_column("H:H", 18)
    ws.set_column("I:I", 16)
    ws.set_row(0, 6)

    # Top title band
    ws.merge_range("A1:I3", "Enviro Labs – Analytical Report", f_title)

    # Date (report date) & Sample ID (to mimic D12/E17 concept)
    ws.write("C5", "Report Date", f_label)
    ws.write_datetime("D5", datetime.utcnow(), f_date)
    ws.write("F5", "Sample ID", f_label)
    ws.write("G5", sample_id, f_value)

    # Client/Product/Brand/Matrix/Analyte section (mimic A17/D17/G17/H17)
    ws.write("A7", "Product", f_label)
    ws.write("B7", tp_get_by_alias(tp_row, TP_ALIASES["product"]) or "", f_value)
    ws.write("D7", "Brand", f_label)
    ws.write("E7", tp_get_by_alias(tp_row, TP_ALIASES["brand"]) or "", f_value)
    ws.write("G7", "Matrix", f_label)
    ws.write("H7", tp_get_by_alias(tp_row, TP_ALIASES["matrix"]) or "", f_value)

    ws.write("A9", "Analyte", f_label)
    analyte = tp_get_by_alias(tp_row, TP_ALIASES["analyte"]) or (sel["primary"][sel["columns"]["ana"]] if sel["primary"] is not None and sel["columns"]["ana"] else "")
    ws.write("B9", analyte, f_value)
    ws.write("D9", "Regulatory Limit (m)", f_label)
    _num_or_text(ws, "E9", tp_get_by_alias(tp_row, TP_ALIASES["limitm"]), f_value)

    # Results table (primary + QC)
    ws.write("A12", "Results", f_subttl)
    headers = ["Type", "Record", "Analyte", "Result", "Units"]
    for j, h in enumerate(headers):
        ws.write(12, j, h, f_th)  # row 13 (0-index => row 12)

    # helper to add a row
    def add_row(r, typ: str):
        nonlocal ws
        if r is None:
            return ["", "", "", "", ""]
        c = sel["columns"]
        rec = str(r.get(c["rec"])) if c["rec"] else ""
        ana = str(r.get(c["ana"])) if c["ana"] else ""
        res = r.get(c["res"])
        uom = str(r.get(c["uom"])) if c["uom"] else ""
        return [typ, rec or "", ana or "", "" if res is None or pd.isna(res) else str(res), uom]

    data_rows = []
    data_rows.append(add_row(sel["primary"], "Primary"))
    data_rows.append(add_row(sel["method_blank"], "Method Blank"))
    data_rows.append(add_row(sel["ms1"], "Matrix Spike 1"))
    data_rows.append(add_row(sel["msd"], "Matrix Spike Duplicate"))

    base_r = 13  # start writing after header
    for i, rowvals in enumerate(data_rows):
        for j, v in enumerate(rowvals):
            fmt = f_center if j in (0,) else f_td
            ws.write(base_r + i, j, v, fmt)

    # Footnote
    ws.write(base_r + len(data_rows) + 2, 0,
             "Notes: Values are computed from Data_Consolidator and TotalProducts uploads. "
             "Report layout styled to match Enviro Labs branded template.",
             f_small)

    # dump source sheets for traceability
    def write_df(sheet_name: str, df: pd.DataFrame):
        s = wb.add_worksheet(sheet_name[:31])
        bold = wb.add_format({"bold": True, "bg_color": "#F3F6FA"})
        for j, col in enumerate(df.columns):
            s.write(0, j, str(col), bold)
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            for j, col in enumerate(df.columns):
                v = r[col]
                s.write(i, j, "" if pd.isna(v) else str(v))

    write_df("TotalProducts", tp_df)
    write_df("Data_Consolidator", dc_df)

    wb.close()
    out.seek(0)
    return out.getvalue()

@app.route("/build_report", methods=["GET", "POST"])
@require_login(role="admin")
def build_report():
    if request.method == "GET":
        # quick inline form (keeps your existing templates untouched)
        return """
        <div style="max-width:760px;margin:40px auto;font-family:Inter,system-ui,Arial;line-height:1.4;">
          <h2 style="margin:0 0 16px;">Build Report (Admin)</h2>
          <form method="post" enctype="multipart/form-data" style="display:grid;gap:12px;">
            <label>Sample ID
              <input name="sample_id" required style="width:100%;padding:8px;margin-top:4px;" />
            </label>
            <label>TotalProducts (CSV/XLSX)
              <input type="file" name="total_products" accept=".csv,.xlsx,.xlsm,.xltx,.xltm" required />
            </label>
            <label>Data_Consolidator (CSV/XLSX)
              <input type="file" name="data_consolidator" accept=".csv,.xlsx,.xlsm,.xltx,.xltm" required />
            </label>
            <div style="margin-top:6px;">
              <button type="submit" style="padding:10px 16px;background:#133E7C;color:white;border:none;border-radius:6px;">Build Workbook</button>
              <a href="/dashboard" style="margin-left:12px;">Back to Dashboard</a>
            </div>
            <p style="color:#666;margin-top:8px;">
              The report sheet will be styled (header, sections, borders) and filled from <b>TotalProducts</b> and <b>Data_Consolidator</b>.
              If you’d like exact cell-by-cell placement (e.g., “put Method Blank result in D65”), tell me those coordinates and I’ll wire them up.
            </p>
          </form>
        </div>
        """

    sample_id = request.form.get("sample_id", "").strip()
    if not sample_id:
        flash("Please provide a Sample ID.", "error")
        return redirect(url_for("build_report"))

    tp_file = request.files.get("total_products")
    dc_file = request.files.get("data_consolidator")
    if not tp_file or not tp_file.filename:
        flash("TotalProducts file is required.", "error")
        return redirect(url_for("build_report"))
    if not dc_file or not dc_file.filename:
        flash("Data_Consolidator file is required.", "error")
        return redirect(url_for("build_report"))

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
        if os.path.exists(tp_path) and not KEEP_UPLOADED_CSVS: os.remove(tp_path)
        if os.path.exists(dc_path) and not KEEP_UPLOADED_CSVS: os.remove(dc_path)
        return redirect(url_for("build_report"))

    try:
        xlsx_bytes = draw_report_layout_and_fill(sample_id, tp_df, dc_df)
    except Exception as e:
        flash(f"Report build failed: {e}", "error")
        xlsx_bytes = None

    if os.path.exists(tp_path) and not KEEP_UPLOADED_CSVS: os.remove(tp_path)
    if os.path.exists(dc_path) and not KEEP_UPLOADED_CSVS: os.remove(dc_path)

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
