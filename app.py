import os
import io
import csv
import glob
from datetime import datetime, date
from io import BytesIO

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
import xlsxwriter

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
DATA_FOLDER   = os.path.join(BASE_DIR, "data")  # for Google Sheets CSV datasets
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Data files used by the Excel formulas
TOTAL_PRODUCTS_PATH    = os.path.join(DATA_FOLDER, "TotalProducts.csv")
DATA_CONSOLIDATOR_PATH = os.path.join(DATA_FOLDER, "Data_Consolidator.csv")
# Any files matching data/Gen_LIMs_*.csv will be consolidated automatically

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
    patient_name = Column(String, nullable=True)
    test = Column(String, nullable=True)
    result = Column(String, nullable=True)
    collected_date = Column(Date, nullable=True)
    resulted_date = Column(Date, nullable=True)
    pdf_url = Column(String, nullable=True)  # optional link to PDF
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
    if pd.isna(val) or str(val).strip() == "":
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

# Accept common header variants from uploads
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

def _read_csv_df(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)

def _list_gen_lims_csvs():
    return sorted(glob.glob(os.path.join(DATA_FOLDER, "Gen_LIMs_*.csv")))

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
    end   = request.args.get("end", "").strip()

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

    # Show which datasets are currently present
    datasets_state = {
        "TotalProducts": os.path.exists(TOTAL_PRODUCTS_PATH),
        "Data_Consolidator": os.path.exists(DATA_CONSOLIDATOR_PATH),
        "Gen_LIMs_*": len(_list_gen_lims_csvs()) > 0
    }

    return render_template("dashboard.html", user=u, reports=reports, datasets_state=datasets_state)

@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    db = SessionLocal()
    r = db.get(Report, report_id)
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
        df = pd.read_csv(parse_path)
    except Exception:
        try:
            df = pd.read_excel(parse_path)
        except Exception as e:
            flash(f"Could not read file: {e}", "error")
            if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
                os.remove(saved_path)
            return redirect(url_for("dashboard"))

    df.columns = [str(c).strip() for c in df.columns]

    c_lab_id = get_col(df, "lab_id")
    c_client = get_col(df, "client")
    if not c_lab_id or not c_client:
        flash("CSV must include Lab ID and Client columns (various names accepted).", "error")
        if os.path.exists(saved_path) and (not keep or not KEEP_UPLOADED_CSVS):
            os.remove(saved_path)
        return redirect(url_for("dashboard"))

    c_patient  = get_col(df, "patient_name")
    c_test     = get_col(df, "test")
    c_result   = get_col(df, "result")
    c_collected= get_col(df, "collected_date")
    c_resulted = get_col(df, "resulted_date")
    c_pdf      = get_col(df, "pdf_url")

    db = SessionLocal()
    created, updated = 0, 0
    try:
        for _, row in df.iterrows():
            lab_id = str(row[c_lab_id]).strip()
            if lab_id == "" or lab_id.lower() == "nan":
                continue
            client = str(row[c_client]).strip() if c_client else CLIENT_NAME

            existing = db.query(Report).filter(Report.lab_id == lab_id).one_or_none()
            if not existing:
                existing = Report(lab_id=lab_id, client=client)
                db.add(existing)
                created += 1
            else:
                updated += 1

            if c_patient:  existing.patient_name = None if pd.isna(row[c_patient]) else str(row[c_patient])
            if c_test:     existing.test         = None if pd.isna(row[c_test]) else str(row[c_test])
            if c_result:   existing.result       = None if pd.isna(row[c_result]) else str(row[c_result])
            if c_collected:existing.collected_date = parse_date(row[c_collected])
            if c_resulted: existing.resulted_date  = parse_date(row[c_resulted])
            if c_pdf:      existing.pdf_url      = None if pd.isna(row[c_pdf]) else str(row[c_pdf])

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

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    """Upload Google Sheets CSVs used by the Excel formulas."""
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))
    if u["role"] != "admin":
        flash("Admins only.", "error")
        return redirect(url_for("dashboard"))

    kind = request.form.get("dataset_kind")  # "TOTAL" | "CONSOLIDATOR" | "GEN"
    f = request.files.get("dataset_file")
    if not f or kind not in {"TOTAL", "CONSOLIDATOR", "GEN"}:
        flash("Choose dataset (TotalProducts / Data_Consolidator / Gen_LIMs_*) and a CSV file.", "error")
        return redirect(url_for("dashboard"))

    if kind == "TOTAL":
        dest = TOTAL_PRODUCTS_PATH
    elif kind == "CONSOLIDATOR":
        dest = DATA_CONSOLIDATOR_PATH
    else:
        # Keep original filename for multiple Gen_LIMs_* files
        dest = os.path.join(DATA_FOLDER, secure_filename(f.filename))

    f.save(dest)
    log_action(u["username"], u["role"], "upload_dataset", f"{kind} -> {os.path.basename(dest)}")
    flash(f"{kind} dataset updated.", "success")
    return redirect(url_for("dashboard"))

# ---------- Excel builder with your formulas ----------
def write_report_workbook_v2(rep: Report) -> BytesIO:
    """
    Builds an .xlsx with:
      - Report (template + formulas mapped below)
      - TotalProducts (from TotalProducts.csv)
      - Data_Consolidator (from Data_Consolidator.csv)
      - Gen_Combined (ID, Analyte, Result, ValueW from all Gen_LIMs_*.csv)
    """
    # Load datasets
    tp_df = _read_csv_df(TOTAL_PRODUCTS_PATH)
    dc_df = _read_csv_df(DATA_CONSOLIDATOR_PATH)

    gen_dfs = []
    for p in _list_gen_lims_csvs():
        df = _read_csv_df(p)
        if df.empty:
            continue
        # Need at least 23 columns to safely pick C(2), F(5), M(12), W(22)
        if df.shape[1] >= 23:
            sub = pd.DataFrame({
                "ID":      df.iloc[:, 2],
                "Analyte": df.iloc[:, 5],
                "Result":  df.iloc[:, 12],
                "ValueW":  df.iloc[:, 22],
            })
            sub = sub[(sub["ID"].astype(str).str.strip() != "")]
            gen_dfs.append(sub)
    gen_combined = pd.concat(gen_dfs, ignore_index=True) if gen_dfs else pd.DataFrame(columns=["ID","Analyte","Result","ValueW"])

    out = BytesIO()
    wb  = xlsxwriter.Workbook(out, {"in_memory": True})
    wsR = wb.add_worksheet("Report")
    wsTP = wb.add_worksheet("TotalProducts")
    wsDC = wb.add_worksheet("Data_Consolidator")
    wsGC = wb.add_worksheet("Gen_Combined")

    # Write data sheets with headers
    if not tp_df.empty:
        for c, col in enumerate(tp_df.columns): wsTP.write(0, c, col)
        for r_i, row in tp_df.iterrows():
            for c, col in enumerate(tp_df.columns): wsTP.write(r_i+1, c, row[col])

    if not dc_df.empty:
        for c, col in enumerate(dc_df.columns): wsDC.write(0, c, col)
        for r_i, row in dc_df.iterrows():
            for c, col in enumerate(dc_df.columns): wsDC.write(r_i+1, c, row[col])

    for c, col in enumerate(gen_combined.columns): wsGC.write(0, c, col)
    for r_i, row in gen_combined.iterrows():
        wsGC.write(r_i+1, 0, row["ID"])
        wsGC.write(r_i+1, 1, row["Analyte"])
        wsGC.write(r_i+1, 2, row["Result"])
        wsGC.write(r_i+1, 3, row["ValueW"])

    bold = wb.add_format({"bold": True})

    # Optional header from DB
    wsR.write("B6","Client:",bold);   wsR.write("C6", rep.client or "")
    wsR.write("B7","Patient:",bold);  wsR.write("C7", rep.patient_name or "")
    wsR.write("B8","Lab ID:",bold);   wsR.write("C8", rep.lab_id or "")
    wsR.write("B9","Test:",bold);     wsR.write("C9", rep.test or "")

    # ======= YOUR FORMULAS (Excel 365/2021) =======
    wsR.write_formula("I3",  "=H12")
    wsR.write_formula("H12", "=E17")
    wsR.write_formula("D12", "=TODAY()")
    wsR.write_formula("F12", "=H17")

    wsR.data_validation("E17", {"validate":"list", "source":"=TotalProducts!I2:I1048576"})
    wsR.write_formula("A17", '=INDEX(TotalProducts!B:B, MATCH(E17, TotalProducts!I:I, 0))')
    wsR.write_formula("D17", '=INDEX(TotalProducts!H:H, MATCH(E17, TotalProducts!I:I, 0))')
    wsR.write_formula("G17", '=INDEX(TotalProducts!D:D, MATCH(E17, TotalProducts!I:I, 0))')
    wsR.write_formula("H17", '=INDEX(TotalProducts!G:G, MATCH(E17, TotalProducts!I:I, 0))')

    # A28 / G28 via Gen_Combined in place of GETSHEETNAMES/QUERY
    wsR.write_formula("A28",
        '=IFERROR(INDEX(FILTER(Gen_Combined!B:B,'
        '(LEFT(Gen_Combined!A:A, LEN(E17))=E17)*'
        '((Gen_Combined!B:B="Bisphenol S")+(Gen_Combined!B:B="PFAS"))'
        '),1),"Not Found")'
    )
    wsR.write_formula("D28", "=VLOOKUP(E17, TotalProducts!I:M, 5, FALSE)")
    wsR.write_formula("E28", "=1*G28")
    wsR.write_formula("G28",
        '=IFERROR(INDEX(FILTER(Gen_Combined!D:D,'
        '(LEFT(Gen_Combined!A:A, LEN(E17))=E17)*'
        '((Gen_Combined!B:B="Bisphenol S")+(Gen_Combined!B:B="PFAS"))'
        '),1),"Not Found")'
    )
    wsR.write_formula("H28",
        '=INDEX('
        'FILTER(Data_Consolidator!E:E,'
        ' ISNUMBER(SEARCH(E17,Data_Consolidator!A:A))'
        ' * (1-ISNUMBER(SEARCH("spike",LOWER(Data_Consolidator!A:A))))'
        ' * (1-ISNUMBER(SEARCH("blank",LOWER(Data_Consolidator!A:A))))'
        ' * (1-ISNUMBER(SEARCH("calibrant",LOWER(Data_Consolidator!A:A))))'
        ' * ((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)'
    )

    wsR.write_formula("I54", "=H12")
    wsR.write_formula("A57", "=A10")
    wsR.write_formula("F57", "=F10")
    wsR.write_formula("D59", "=TODAY()")
    wsR.write_formula("F59", "=F12")
    wsR.write_formula("H59", "=E17")

    sheetname_let = (
        'LET('
        'SheetName, INDEX(FILTER(Data_Consolidator!G:G,'
        ' (LEFT(Data_Consolidator!A:A, LEN(E17))=E17)'
        ' * (Data_Consolidator!A:A<>"Method Blank")'
        ' * (Data_Consolidator!A:A<>"Calibration Blank")'
        '),1), SheetName)'
    )

    wsR.write_formula("G65",
        f'=IFERROR({sheetname_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!D:D,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Method Blank",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("A65",
        f'=IFERROR({sheetname_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!B:B,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Method Blank",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("D65",
        f'=IFERROR({sheetname_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!C:C,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Method Blank",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("E65", "=1*G65")
    wsR.write_formula("F65", "=F28")
    wsR.write_formula("H65",
        f'=IFERROR({sheetname_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!E:E,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Method Blank",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )

    ms1_let = (
        'LET('
        'SheetName, INDEX(FILTER(Data_Consolidator!G:G,'
        ' (LEFT(Data_Consolidator!A:A, LEN(E17))=E17)'
        ' * (Data_Consolidator!A:A<>"Method Blank")'
        ' * (Data_Consolidator!A:A<>"Calibration Blank")'
        '),1), SheetName)'
    )
    wsR.write_formula("A68",
        f'=IFERROR({ms1_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!B:B,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Matrix Spike 1",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("D68",
        f'=IFERROR({ms1_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!C:C,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Matrix Spike 1",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("G68",
        f'=IFERROR({ms1_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!D:D,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Matrix Spike 1",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("E68", "=1*G68")
    wsR.write_formula("F68", "=F65")

    # Extract numeric from H68 for %Recovery calcs
    num_from_H68 = (
        'LET('
        's,H68,'
        'chars,MID(s,SEQUENCE(LEN(s)),1),'
        'nums,IF(((chars>="0")*(chars<="9"))+(chars="."),chars,""),'
        'VALUE(TEXTJOIN("",,nums))'
        ')'
    )
    wsR.write_formula("G71",
        '=IFERROR('
        f'(D71*100)/(LET(SheetName,INDEX(FILTER(Data_Consolidator!G:G,LEFT(Data_Consolidator!A:A,LEN(E17))=E17),1),'
        'MatrixSpikeName,INDEX(FILTER(Data_Consolidator!A:A,(Data_Consolidator!G:G=SheetName)*(ISNUMBER(SEARCH("Matrix Spike 1",Data_Consolidator!A:A)))),1),'
        'ParentID,TRIM(TEXTAFTER(MatrixSpikeName,": ")),'
        'OriginalResult,INDEX(FILTER(Data_Consolidator!C:C,(Data_Consolidator!G:G=SheetName)*(LEFT(Data_Consolidator!A:A,LEN(ParentID))=ParentID)*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))),1),'
        f'OriginalResult)+{num_from_H68}'
        '), "Calculation Error")'
    )
    dup_let = (
        'LET('
        'SheetName, INDEX(FILTER(Data_Consolidator!G:G,'
        ' (LEFT(Data_Consolidator!A:A, LEN(E17))=E17)'
        ' * (Data_Consolidator!A:A<>"Method Blank")'
        ' * (Data_Consolidator!A:A<>"Calibration Blank")'
        '),1), SheetName)'
    )
    wsR.write_formula("F71",
        f'=IFERROR({dup_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!D:D,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Matrix Spike Duplicate",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("E71", "=F68")
    wsR.write_formula("D71",
        f'=IFERROR({dup_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!C:C,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Matrix Spike Duplicate",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("A71",
        f'=IFERROR({dup_let[:-1]},'
        'INDEX(FILTER(Data_Consolidator!B:B,'
        '(Data_Consolidator!G:G=SheetName)'
        '*(ISNUMBER(SEARCH("Matrix Spike Duplicate",Data_Consolidator!A:A)))'
        '*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))'
        '),1)),"Not Found")'
    )
    wsR.write_formula("I68",
        '=IFERROR('
        f'(D68*100)/(LET(SheetName,INDEX(FILTER(Data_Consolidator!G:G,LEFT(Data_Consolidator!A:A,LEN(E17))=E17),1),'
        'MatrixSpikeName,INDEX(FILTER(Data_Consolidator!A:A,(Data_Consolidator!G:G=SheetName)*(ISNUMBER(SEARCH("Matrix Spike 1",Data_Consolidator!A:A)))),1),'
        'ParentID,TRIM(TEXTAFTER(MatrixSpikeName,": ")),'
        'OriginalResult,INDEX(FILTER(Data_Consolidator!C:C,(Data_Consolidator!G:G=SheetName)*(LEFT(Data_Consolidator!A:A,LEN(ParentID))=ParentID)*((Data_Consolidator!B:B="Bisphenol S")+(Data_Consolidator!B:B="PFAS"))),1),'
        f'OriginalResult)+{num_from_H68}'
        '), "Calculation Error")'
    )

    wsR.set_landscape()
    wsR.fit_to_pages(1, 1)

    wb.close()
    out.seek(0)
    return out

@app.route("/download_report_xlsx/<int:report_id>")
def download_report_xlsx(report_id):
    u = current_user()
    if not u["username"]:
        return redirect(url_for("home"))

    dbs = SessionLocal()
    rep = dbs.get(Report, report_id)
    dbs.close()
    if not rep:
        flash("Report not found", "error")
        return redirect(url_for("dashboard"))
    if u["role"] == "client" and rep.client != u["client_name"]:
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    xbytes = write_report_workbook_v2(rep)
    fname  = f"Report_{rep.lab_id or report_id}.xlsx"
    return send_file(
        xbytes,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=fname,
    )

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

# ----------- Minimal health check for Render -----------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
