import os
import io
import re
from datetime import datetime, date
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

def load_tabular_file(path: str, filename: str) -> pd.DataFrame:
    # Tries CSV, then Excel, with sensible defaults for Google Sheets exports.
    ext = (os.path.splitext(filename or "")[1] or "").lstrip(".").lower()
    if ext in {"csv", "tsv", "txt"}:
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
            except Exception as e:
                last_err = e
                continue
        raise last_err or ValueError("Unable to parse CSV-like file.")
    if ext in {"xlsx", "xlsm", "xltx", "xltm"}:
        return pd.read_excel(path, engine="openpyxl", dtype=str)
    # Fallback guess
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    except Exception:
        return pd.read_excel(path, engine="openpyxl", dtype=str)

# ---- generic column helpers for your two files ----
def get_col_by_letter(df: pd.DataFrame, letter: str) -> str | None:
    idx = ord(letter.upper()) - ord("A")
    if 0 <= idx < len(df.columns):
        return df.columns[idx]
    return None

def extract_number(s) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))
    return float(m.group(0)) if m else None

# --- lookup in TotalProducts by the Sample ID key in column I (or by header name) ---
TP_SAMPLE_ID_HEADERS = {"sample id", "sampleid", "lab id", "labid", "identifier", "id"}

def tp_find_row_by_sample_id(tp_df: pd.DataFrame, sample_id: str) -> pd.Series | None:
    # Try header-based first
    for c in tp_df.columns:
        if _norm(c) in TP_SAMPLE_ID_HEADERS:
            hits = tp_df[c].astype(str).str.strip().str.casefold() == sample_id.strip().casefold()
            if hits.any():
                return tp_df[hits].iloc[0]
    # Fallback to column I (9th col) per your formula
    col_I = get_col_by_letter(tp_df, "I")
    if col_I:
        hits = tp_df[col_I].astype(str).str.strip().str.casefold() == sample_id.strip().casefold()
        if hits.any():
            return tp_df[hits].iloc[0]
    return None

def tp_get_col(row: pd.Series, letter: str):
    if row is None:
        return None
    c = get_col_by_letter(pd.DataFrame([row]).reset_index(drop=True), letter)
    return None if c is None else row.get(c)

# ---- filters for Data_Consolidator ----
def dc_cols(df: pd.DataFrame):
    # Expected: A=name/record, B=Analyte, C=Result (text), D=Numeric value, E=Units or spike text, G=Sheet
    cols = {i: df.columns[i] for i in range(len(df.columns))}
    return {
        "A": cols.get(0),
        "B": cols.get(1),
        "C": cols.get(2),
        "D": cols.get(3),
        "E": cols.get(4),
        "G": cols.get(6) if len(df.columns) > 6 else None,
    }

def dc_subset_for_sample(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    c = dc_cols(df)
    A = c["A"]
    if A:
        return df[df[A].astype(str).str.startswith(str(sample_id), na=False)].copy()
    return df.head(0).copy()

def dc_prefer_analyte(df: pd.DataFrame) -> pd.Series | None:
    c = dc_cols(df)
    B = c["B"]
    if not B or df.empty:
        return None
    for target in ("Bisphenol S", "PFAS"):
        hit = df[df[B].astype(str).str.strip().str.casefold() == target.strip().casefold()]
        if len(hit) > 0:
            return hit.iloc[0]
    return df.iloc[0] if len(df) > 0 else None

def dc_first_contains(df: pd.DataFrame, text: str) -> pd.Series | None:
    c = dc_cols(df); A = c["A"]
    if not A or df.empty:
        return None
    hit = df[df[A].astype(str).str.contains(text, case=False, na=False)]
    return hit.iloc[0] if len(hit) > 0 else None

def percent_recovery(ms_row: pd.Series, base_df: pd.DataFrame, sample_id: str) -> str:
    """
    % = (NumericValue * 100) / (OriginalResult + SpikeAmount)
    - NumericValue from D of MS row
    - SpikeAmount = first number parsed from E of MS row
    - ParentID = text after ':' in the MS row's A; parent's original result from C
    """
    if ms_row is None or base_df is None or base_df.empty:
        return "Calculation Error"
    c = dc_cols(base_df)
    A, B, Cc, D, E = c["A"], c["B"], c["C"], c["D"], c["E"]
    name = str(ms_row.get(A)) if A else ""
    m = re.search(r":\s*(.*)$", name)
    parent = m.group(1).strip() if m else sample_id
    parent_rows = base_df[base_df[A].astype(str).str.startswith(parent, na=False)] if A else base_df.head(0)
    if len(parent_rows) == 0:
        return "Calculation Error"
    ana = str(ms_row.get(B)) if B else None
    if ana:
        cand = parent_rows[parent_rows[B].astype(str).str.strip().str.casefold() == ana.strip().casefold()]
        if len(cand) > 0:
            parent_rows = cand
    parent_row = parent_rows.iloc[0]
    orig = extract_number(parent_row.get(Cc) if Cc else None)
    spike_amt = extract_number(ms_row.get(E) if E else None)
    ms_val = ms_row.get(D) if D else None
    try:
        ms_num = float(ms_val) if ms_val is not None and str(ms_val).strip() != "" else None
    except Exception:
        ms_num = extract_number(ms_val)
    if orig is None or ms_num is None:
        return "Calculation Error"
    denom = orig + (spike_amt or 0.0)
    if denom == 0:
        return "Calculation Error"
    return f"{(ms_num * 100.0) / denom:.1f}"

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

# -------- Upload to index reports quickly (optional) --------
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

    try:
        df = load_tabular_file(saved_path, filename)
    except Exception as e:
        flash(f"Could not read file: {e}", "error")
        return redirect(url_for("dashboard"))

    df.columns = [str(c).strip() for c in df.columns]

    # Lab ID: accept Sample ID/Lab ID variants
    labid_header = None
    for c in df.columns:
        if _norm(c) in {"labid","lab id","sample id","sampleid","identifier","accession"}:
            labid_header = c
            break
    if labid_header is None:
        flash("CSV must include Lab ID / Sample ID column.", "error")
        return redirect(url_for("dashboard"))

    client_header = None
    for c in df.columns:
        if _norm(c) in {"client","clientname","client name","facility","account","company"}:
            client_header = c
            break

    db = SessionLocal()
    created, updated = 0, 0
    try:
        for _, row in df.iterrows():
            lab_id_val = str(row[labid_header]).strip()
            if not lab_id_val:
                continue
            client_val = str(row[client_header]).strip() if client_header else CLIENT_NAME
            existing = db.query(Report).filter(Report.lab_id == lab_id_val).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id_val, client=client_val)
                db.add(existing); created += 1
            else:
                updated += 1
        db.commit()
        flash(f"Imported {created} new and updated {updated} report(s).", "success")
    except Exception as e:
        db.rollback()
        flash(f"Import failed: {e}", "error")
        app.logger.exception("Import failed")
    finally:
        db.close()

    return redirect(url_for("dashboard"))

# ---------------- Build Report (Admin) ----------------
def write_cell(ws, addr: str, value, fmt=None, dt_fmt=None):
    # Use ws.write for A1-style cell addresses (works with datetimes too)
    if isinstance(value, (datetime, date)):
        ws.write(addr, value, dt_fmt or fmt)
        return
    n = None
    if isinstance(value, (int, float)):
        n = float(value)
    else:
        n = extract_number(value)
    if n is not None and str(value).strip() not in ("", "Not Found"):
        ws.write_number(addr, n, fmt)
    else:
        ws.write(addr, "" if value is None else str(value), fmt)

def draw_report_by_cellmap(sample_id: str, tp_df: pd.DataFrame, dc_df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    wb = xlsxwriter.Workbook(out, {"in_memory": True})
    ws = wb.add_worksheet("Report")

    # Formats
    f_date = wb.add_format({"num_format": "yyyy-mm-dd"})
    f_num  = wb.add_format({"border": 1})

    ws.set_column("A:I", 16)

    # ---- Load source rows ----
    tp_row = tp_find_row_by_sample_id(tp_df, sample_id)

    # Data_Consolidator rows subset for this sample
    dc_for_sample = dc_subset_for_sample(dc_df, sample_id)
    c = dc_cols(dc_for_sample)
    A, B, Cc, D, E, G = c["A"], c["B"], c["C"], c["D"], c["E"], c["G"]

    primary = dc_prefer_analyte(dc_for_sample)
    method_blank = dc_first_contains(dc_for_sample, "Method Blank")
    ms1          = dc_first_contains(dc_for_sample, "Matrix Spike 1")
    msd          = dc_first_contains(dc_for_sample, "Matrix Spike Duplicate")

    # ------------- Cell mappings -------------
    # IDs/dates
    ws.write("E17", sample_id)
    ws.write("H12", sample_id)
    ws.write("I3",  sample_id)
    ws.write("D12", datetime.utcnow(), f_date)

    # TotalProducts lookups
    a17 = tp_get_col(tp_row, "B")
    d17 = tp_get_col(tp_row, "H")
    g17 = tp_get_col(tp_row, "D")
    h17 = tp_get_col(tp_row, "G")
    ws.write("A17", "" if a17 is None else str(a17))
    ws.write("D17", "" if d17 is None else str(d17))
    ws.write("G17", "" if g17 is None else str(g17))
    ws.write("H17", "" if h17 is None else str(h17))

    # f12 = h17
    ws.write("F12", "" if h17 is None else str(h17))

    # Row 28 block
    a28 = (primary.get(B) if (primary is not None and B) else None) or h17 or "Not Found"
    ws.write("A28", str(a28))

    d28 = tp_get_col(tp_row, "M")  # VLOOKUP to M
    ws.write("D28", "" if d28 is None else str(d28))

    g28 = primary.get(D) if (primary is not None and D) else None
    write_cell(ws, "G28", g28, f_num)
    write_cell(ws, "E28", g28, f_num)  # 1*G28
    h28 = primary.get(E) if (primary is not None and E) else None
    ws.write("H28", "" if h28 is None else str(h28))

    # Misc copies
    ws.write("I54", sample_id)
    ws.write("D59", datetime.utcnow(), f_date)
    ws.write("F59", "" if h17 is None else str(h17))
    ws.write("H59", sample_id)
    ws.write("A57", "")
    ws.write("F57", "")

    # Method Blank (row 65)
    a65 = method_blank.get(B) if (method_blank is not None and B) else "Not Found"
    d65 = method_blank.get(Cc) if (method_blank is not None and Cc) else "Not Found"
    g65 = method_blank.get(D) if (method_blank is not None and D) else None
    h65 = method_blank.get(E) if (method_blank is not None and E) else ""
    ws.write("A65", str(a65))
    ws.write("D65", "" if d65 is None else str(d65))
    write_cell(ws, "G65", g65, f_num)
    ws.write("H65", "" if h65 is None else str(h65))
    write_cell(ws, "E65", g65, f_num)  # 1*G65
    ws.write("F65", "" if d28 is None else str(d28))

    # Matrix Spike 1 (row 68)
    a68 = ms1.get(B) if (ms1 is not None and B) else "Not Found"
    d68 = ms1.get(Cc) if (ms1 is not None and Cc) else "Not Found"
    g68 = ms1.get(D) if (ms1 is not None and D) else None
    h68 = ms1.get(E) if (ms1 is not None and E) else ""
    ws.write("A68", str(a68))
    ws.write("D68", "" if d68 is None else str(d68))
    write_cell(ws, "G68", g68, f_num)
    ws.write("H68", "" if h68 is None else str(h68))
    write_cell(ws, "E68", g68, f_num)  # 1*G68
    ws.write("F68", "" if d28 is None else str(d28))
    i68 = percent_recovery(ms1, dc_for_sample, sample_id)
    ws.write("I68", i68 if isinstance(i68, str) else f"{i68}")

    # Matrix Spike Duplicate (row 71)
    a71 = msd.get(B) if (msd is not None and B) else "Not Found"
    d71 = msd.get(Cc) if (msd is not None and Cc) else "Not Found"
    f71 = msd.get(D) if (msd is not None and D) else None
    ws.write("A71", str(a71))
    ws.write("D71", "" if d71 is None else str(d71))
    write_cell(ws, "F71", f71, f_num)
    g71 = percent_recovery(msd, dc_for_sample, sample_id)
    ws.write("G71", g71 if isinstance(g71, str) else f"{g71}")

    # Include source tabs for traceability
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
        app.logger.exception("Failed to read uploads")
        flash(f"Could not read uploaded files: {e}", "error")
        return redirect(url_for("build_report"))

    try:
        xlsx_bytes = draw_report_by_cellmap(sample_id, tp_df, dc_df)
    except Exception as e:
        app.logger.exception("Report build failed")
        flash(f"Report build failed: {e}", "error")
        return redirect(url_for("build_report"))

    if os.path.exists(tp_path) and not KEEP_UPLOADED_CSVS: os.remove(tp_path)
    if os.path.exists(dc_path) and not KEEP_UPLOADED_CSVS: os.remove(dc_path)

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
