import os
import io
import re
import math
from datetime import datetime, date
from typing import List, Optional, Dict, Any

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text as sql_text
import pandas as pd
import json
from functools import wraps

# ------------------- Config -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME = os.getenv("CLIENT_NAME", "Artemis")

KEEP_UPLOADED_CSVS = str(os.getenv("KEEP_UPLOADED_CSVS", "true")).lower() == "true"
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PFAS_LIST: List[str] = [
    "PFOA", "PFOS", "PFNA", "FOSAA", "N-MeFOSAA", "N-EtFOSAA",
    "SAmPAP", "PFOSA", "N-MeFOSA", "N-MeFOSE", "N-EtFOSA", "N-EtFOSE",
    "diSAmPAP",
]

# ------------------- App & DB -------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

@app.context_processor
def _inject_current_app():
    return {"current_app": current_app}

DB_PATH = os.path.join(BASE_DIR, "app.db")
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False, index=True)
    test = Column(String, nullable=True)     
    result = Column(String, nullable=True)   
    collected_date = Column(Date, nullable=True)  
    resulted_date = Column(Date, nullable=True)   
    pdf_url = Column(String, nullable=True)       
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    project_lead = Column(String, nullable=True)
    address = Column(String, nullable=True)
    sample_name = Column(String, nullable=True)
    prepared_by = Column(String, nullable=True)
    matrix = Column(String, nullable=True)
    prepared_date = Column(String, nullable=True)
    qualifiers = Column(String, nullable=True)
    asin = Column(String, nullable=True)
    product_weight_g = Column(String, nullable=True)
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)
    mb_analyte = Column(String, nullable=True)
    mb_result = Column(String, nullable=True)
    mb_mrl = Column(String, nullable=True)
    mb_units = Column(String, nullable=True)
    mb_dilution = Column(String, nullable=True)
    acq_datetime = Column(String, nullable=True)  
    sheet_name = Column(String, nullable=True)    
    ms1_analyte = Column(String, nullable=True)
    ms1_result = Column(String, nullable=True)
    ms1_mrl = Column(String, nullable=True)
    ms1_units = Column(String, nullable=True)
    ms1_dilution = Column(String, nullable=True)
    ms1_fortified_level = Column(String, nullable=True)
    ms1_pct_rec = Column(String, nullable=True)
    ms1_pct_rec_limits = Column(String, nullable=True)  
    msd_analyte = Column(String, nullable=True)
    msd_result = Column(String, nullable=True)
    msd_units = Column(String, nullable=True)
    msd_dilution = Column(String, nullable=True)
    msd_pct_rec = Column(String, nullable=True)
    msd_pct_rec_limits = Column(String, nullable=True)
    msd_pct_rpd = Column(String, nullable=True)
    msd_pct_rpd_limit = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)  
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)

class ChainOfCustody(Base):
    __tablename__ = "coc_records"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, unique=True, index=True)
    client_name = Column(String)
    sample_name = Column(String)
    asin = Column(String)
    sample_type = Column(String)
    product_link = Column(String, nullable=True)
    matrix = Column(String, nullable=True)
    anticipated_chemical = Column(String, nullable=True)
    expected_delivery_date = Column(String, nullable=True)
    storage_bin_no = Column(String, nullable=True)
    analyzed = Column(String, nullable=True)
    analysis_date = Column(String, nullable=True)
    results_ng_g = Column(String, nullable=True)
    comments = Column(Text, nullable=True)
    sample_condition = Column(String, nullable=True)
    weight_grams = Column(String, nullable=True)
    carrier_name = Column(String, nullable=True)
    tracking_number = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    project_lead = Column(String, nullable=True)
    address = Column(String, nullable=True)
    status = Column(String, default="Pending")
    location = Column(String, default="Intake")
    received_at = Column(DateTime, nullable=True)
    received_by = Column(String, nullable=True)
    received_by_role = Column(String, nullable=True)
    received_via_file = Column(String, nullable=True)

Base.metadata.create_all(engine)
# ------------------- Helpers & Auth -------------------
PFAS_SET_UPPER = {a.upper() for a in PFAS_LIST}
STATIC_ANALYTES_LIST = [
    "PFOA", "PFOS", "PFNA", "FOSAA", "N-MeFOSAA", "N-EtFOSAA",
    "SAmPAP", "PFOSA", "N-MeFOSA", "N-MeFOSE", "N-EtFOSA", "N-EtFOSE",
    "diSAmPAP", "Bisphenol S"
]

def current_user():
    return {"username": session.get("username"), "role": session.get("role"), "client_name": session.get("client_name")}

def require_login(role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "username" not in session: return redirect(url_for("home"))
            if role and session.get("role") != role: return redirect(url_for("dashboard"))
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def log_action(username, role, action, details=""):
    db = SessionLocal()
    try:
        db.add(AuditLog(username=username, role=role, action=action, details=details))
        db.commit()
    except Exception: db.rollback()
    finally: db.close()

def parse_date(val):
    if not val: return None
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try: return datetime.strptime(s, fmt).date()
        except: pass
    return None

def _normalize_lab_id(lab_id: str) -> str:
    s = (lab_id or "").strip()
    pattern = r"\s+[\-\+]?\d*\.?\d+(?:ppb|ppt|ng\/g|ug\/g|\s\d*|\s.*)?$"
    return re.sub(pattern, "", s, flags=re.IGNORECASE).strip()

@app.route("/")
def home():
    if session.get("username"): return redirect(url_for("portal_choice"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    role, username, password = request.form.get("role"), request.form.get("username", "").strip(), request.form.get("password", "").strip()
    if role == "admin" and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"], session["role"] = username, "admin"
        return redirect(url_for("portal_choice"))
    if role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"], session["role"], session["client_name"] = username, "client", CLIENT_NAME
        return redirect(url_for("portal_choice"))
    flash("Invalid credentials", "error")
    return redirect(url_for("home"))

@app.route("/portal")
def portal_choice():
    u = current_user()
    if not u["username"]: return redirect(url_for("home"))
    return render_template("portal_choice.html", user=u)

@app.route("/dashboard")
def dashboard():
    u = current_user()
    if not u["username"]: return redirect(url_for("home"))
    db = SessionLocal()
    try:
        q = db.query(Report)
        if u["role"] == "client": q = q.filter(Report.client == u["client_name"])
        reports = q.order_by(Report.id.desc()).limit(500).all()
    finally: db.close()
    return render_template("dashboard.html", user=u, reports=reports)

# ------------------- THE REPAIRED COC LIST ROUTE -------------------
@app.route("/coc")
def coc_list():
    """FIXED: Matches url_for('coc_list') in portal_choice.html"""
    u = current_user()
    if not u.get("username"): return redirect(url_for("home"))
    lab_id_q = request.args.get("lab_id", "").strip()
    per_page_raw = request.args.get("per_page", "50").lower()
    page = max(1, int(request.args.get("page", 1)))
    db = SessionLocal()
    try:
        q = db.query(ChainOfCustody)
        if u["role"] != "admin": q = q.filter(ChainOfCustody.client_name == (u.get("client_name") or ""))
        if lab_id_q: q = q.filter(ChainOfCustody.lab_id.contains(lab_id_q))
        total = q.count()
        q = q.order_by(ChainOfCustody.received_at.desc().nullslast(), ChainOfCustody.id.desc())
        if per_page_raw == "all": records, total_pages = q.all(), 1
        else:
            pp = int(per_page_raw)
            total_pages = max(1, math.ceil(total / pp))
            records = q.offset((page - 1) * pp).limit(pp).all()
        return render_template("coc_list.html", records=records, user=u, total=total, page=page, total_pages=total_pages, per_page=per_page_raw, per_page_options=["10","25","50","100","all"])
    finally: db.close()

# ------------------- THE REPAIRED SINGLE PDF & PRINT ROUTES -------------------
@app.route("/coc/<int:record_id>/print")
def coc_print_single(record_id):
    """FIXED: Matches url_for('coc_print_single') in coc_list.html"""
    u = current_user()
    db = SessionLocal()
    try:
        rec = db.get(ChainOfCustody, record_id)
        if not rec: return redirect(url_for("coc_list"))
        return render_template("coc_print.html", records=[rec], user=u, now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), single=True)
    finally: db.close()

@app.route("/coc/<int:record_id>/pdf")
def coc_pdf_single(record_id):
    """FIXED: Matches url_for('coc_pdf_single') in coc_list.html"""
    u = current_user()
    db = SessionLocal()
    try:
        rec = db.get(ChainOfCustody, record_id)
        if not rec: return redirect(url_for("coc_list"))
        pdf_io = _build_coc_pdf([rec], title=f"Chain of Custody - {rec.lab_id}")
        return send_file(pdf_io, mimetype="application/pdf", as_attachment=True, download_name=f"COC_{rec.lab_id}.pdf")
    finally: db.close()

    def _ingest_total_products_for_coc(df: pd.DataFrame, u: Dict[str, Any], filename: str) -> str:
    """FIXED: Robustly handles TBD placeholders and duplicates within same file upload."""
    PLACEHOLDER_LAB_IDS = {"", "X", "NA", "N/A", "TBD", "NONE", "NULL"}
    created, updated, skipped = 0, 0, 0
    db = SessionLocal()
    seen = {} # Cache to handle duplicates in the SAME Excel sheet
    try:
        for _, row in df.iterrows():
            raw_lab = str(row.get("Laboratory ID", row.get("Sample ID", ""))).strip()
            if not raw_lab: skipped += 1; continue
            lab_id = _normalize_lab_id(raw_lab).strip()
            if lab_id.upper() in PLACEHOLDER_LAB_IDS: skipped += 1; continue

            rec = seen.get(lab_id)
            if rec is None:
                rec = db.query(ChainOfCustody).filter(ChainOfCustody.lab_id == lab_id).one_or_none()

            if not rec:
                rec = ChainOfCustody(lab_id=lab_id)
                db.add(rec); created += 1; db.flush() # Ensure it exists for the next row
            else: updated += 1
            
            seen[lab_id] = rec
            rec.sample_name = str(row.get("Product Name", "")).strip()
            rec.client_name = u.get("client_name") or CLIENT_NAME
            rec.status = "Pending"
            rec.received_via_file = filename
        db.commit()
    except Exception as e: db.rollback(); return f"COC error: {e}"
    finally: db.close()
    return f"COC process: {created} new, {updated} updated."

def _build_coc_pdf(records, title="Chain of Custody"):
    """Full ReportLab Builder with original landscaping and styles."""
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    data = [["SAMPLE NAME", "LAB ID#", "ASIN", "TYPE", "SHIPPED BY", "CONDITION", "STATUS"]]
    for r in records:
        data.append([r.sample_name or "", r.lab_id or "", r.asin or "", r.sample_type or "", r.carrier_name or "", r.sample_condition or "", r.status or ""])
    
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    doc.build([Paragraph(title, styles["Title"]), Spacer(1, 12), table])
    buf.seek(0)
    return buf

@app.route("/coc/receive", methods=["POST"])
def coc_receive_selected():
    u = current_user()
    ids = request.form.getlist("selected_ids")
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        for rid in ids:
            rec = db.get(ChainOfCustody, int(rid))
            if rec: rec.status, rec.received_at, rec.received_by = "Received", now, u["username"]
        db.commit()
    finally: db.close()
    return redirect(url_for("coc_list"))

@app.errorhandler(404)
def not_found(e): return render_template("error.html", code=404, message="Not found"), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
        
