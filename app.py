import os
import io
import re
import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any 

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash, jsonify
)
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text as sql_text
import pandas as pd

# Import formula hooks
try:
    from formula_hooks import compute_fields
except ImportError:
    compute_fields = None

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
    computed = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChainOfCustody(Base):
    __tablename__ = "coc_records"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, unique=True, index=True)
    client_name = Column(String)
    sample_name = Column(String)
    asin = Column(String)
    sample_type = Column(String)
    status = Column(String, default="Received") 
    location = Column(String, default="Intake")
    received_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False) 
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ------------------- Logic -------------------

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
    except: db.rollback()
    finally: db.close()

@app.route("/")
def home():
    if session.get("username"): return redirect(url_for("portal_choice"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    role = request.form.get("role")
    username, password = request.form.get("username", "").strip(), request.form.get("password", "").strip()
    if role == "admin" and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"], session["role"] = username, "admin"
        log_action(username, "admin", "login", "Admin logged in")
        return redirect(url_for("portal_choice"))
    elif role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"], session["role"], session["client_name"] = username, "client", CLIENT_NAME
        log_action(username, "client", "login", "Client logged in")
        return redirect(url_for("portal_choice"))
    flash("Invalid credentials")
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
    q = db.query(Report)
    if u["role"] == "client": q = q.filter(Report.client == u["client_name"])
    
    lab_id = request.args.get("lab_id", "").strip()
    if lab_id:
        q = q.filter(Report.lab_id == lab_id)
    
    reports = q.order_by(Report.resulted_date.desc()).limit(500).all()
    db.close()
    return render_template("dashboard.html", user=u, reports=reports)
    # ------------------- Restored Report Helpers -------------------

PFAS_LIST = ["PFOA","PFOS","PFNA","FOSAA","N-MeFOSAA","N-EtFOSAA","SAmPAP","PFOSA","N-MeFOSA","N-MeFOSE","N-EtFOSA","N-EtFOSE","diSAmPAP"]
PFAS_SET_UPPER = {a.upper() for a in PFAS_LIST}
STATIC_ANALYTES_LIST = ["PFOA", "PFOS", "PFNA", "FOSAA", "N-MeFOSAA", "N-EtFOSAA", "SAmPAP", "PFOSA", "N-MeFOSA", "N-MeFOSE", "N-EtFOSA", "N-EtFOSE", "diSAmPAP", "Bisphenol S"]

def _normalize_lab_id(lab_id: str) -> str:
    s = (lab_id or "").strip()
    if not s: return s
    pattern = r'\s+[\-\+]?\d*\.?\d+(?:ppb|ppt|ng\/g|ug\/g|\s\d*|\s.*)?$'
    normalized = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
    return normalized or s

def _generate_report_table_html(reports, app_instance):
    html_rows = []
    for r in reports:
        detail_url = url_for('report_detail', report_id=r.id)
        row = f"<tr><td><a class='link' href='{detail_url}'>{r.lab_id}</a></td><td>{r.client}</td><td>{r.sample_name or ''}</td><td>{r.test or ''}</td><td>{r.result or ''}</td><td>{r.sample_units or ''}</td><td>{r.resulted_date or ''}</td></tr>"
        html_rows.append(row)
    return "\n".join(html_rows)

def _get_structured_qc_data(r):
    sample_map, mb_map, ms1_map, msd_map = {}, {}, {}, {}
    if r.pdf_url:
        for item in r.pdf_url.split(' | '):
            if ': ' in item:
                analyte, res = item.split(': ', 1)
                sample_map[analyte.strip()] = {'sample_result': res.strip()}
    if r.acq_datetime: # MB Accumulation
        for item in r.acq_datetime.split(' | '):
            parts = item.split('|')
            if len(parts) >= 5:
                mb_map[parts[0].strip()] = {'mb_result': parts[1], 'mb_mrl': parts[2], 'mb_units': parts[3], 'mb_dilution': parts[4]}
    if r.sheet_name: # MS1 Accumulation
        for item in r.sheet_name.split(' | '):
            parts = item.split('|')
            if len(parts) >= 7:
                ms1_map[parts[0].strip()] = {'ms1_result': parts[1], 'ms1_mrl': parts[2], 'ms1_units': parts[3], 'ms1_dilution': parts[4], 'ms1_fortified': parts[5], 'ms1_rec': parts[6]}
    if r.ms1_pct_rec_limits: # MSD Accumulation
        for item in r.ms1_pct_rec_limits.split(' | '):
            parts = item.split('|')
            if len(parts) >= 7:
                msd_map[parts[0].strip()] = {'msd_result': parts[1], 'msd_units': parts[2], 'msd_dilution': parts[3], 'msd_rec': parts[4], 'msd_rec_lim': parts[5], 'msd_rpd': parts[6]}

    final_list = []
    for a in STATIC_ANALYTES_LIST:
        data = {'analyte': a}
        data.update(sample_map.get(a, {}))
        data.update(mb_map.get(a, {}))
        data.update(ms1_map.get(a, {}))
        data.update(msd_map.get(a, {}))
        final_list.append(data)
    return final_list

# ------------------- Chain of Custody Routes -------------------

@app.route("/coc")
def coc_list():
    u = current_user()
    if not u["username"]: return redirect(url_for("home"))
    db = SessionLocal()
    if u["role"] == "admin":
        records = db.query(ChainOfCustody).all()
    else:
        records = db.query(ChainOfCustody).filter_by(client_name=u["client_name"]).all()
    db.close()
    return render_template("coc_list.html", records=records, user=u)

@app.route("/coc/edit/<int:record_id>", methods=["GET", "POST"])
def coc_edit(record_id):
    u = current_user()
    if u["role"] != "admin": return "Unauthorized", 403
    db = SessionLocal()
    record = db.query(ChainOfCustody).get(record_id)
    if request.method == "POST":
        old_status = record.status
        new_status = request.form.get("status")
        new_loc = request.form.get("location")
        if old_status != new_status or record.location != new_loc:
            record.status = new_status
            record.location = new_loc
            log_action(u["username"], u["role"], "COC_UPDATE", f"Sample {record.lab_id}: Status to {new_status}, Loc to {new_loc}")
            db.commit()
            flash("Chain of Custody Updated and Logged")
    history = db.query(AuditLog).filter(AuditLog.details.contains(record.lab_id)).order_by(AuditLog.at.desc()).all()
    db.close()
    return render_template("coc_edit.html", record=record, history=history)

# ------------------- Report Detail & Upload -------------------

@app.route("/report/<int:report_id>")
def report_detail(report_id):
    u = current_user()
    if not u["username"]: return redirect(url_for("home"))
    db = SessionLocal()
    r = db.query(Report).get(report_id)
    db.close()
    if not r: return redirect(url_for("dashboard"))
    
    structured_qc = _get_structured_qc_data(r)
    p = {"analyte_list": structured_qc, "client_info": {"client": r.client, "address": r.address}, "sample_summary": {"sample_name": r.sample_name}}
    return render_template("report_detail.html", user=u, r=r, p=p)

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    u = current_user()
    if u["role"] != "admin": return "Unauthorized", 403
    f = request.files.get("csv_file")
    if not f: return redirect(url_for("dashboard"))
    
    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)
    
    df = pd.read_csv(path, header=1).fillna("") 
    db = SessionLocal()
    created, updated = 0, 0
    for _, row in df.iterrows():
        lab_id = _normalize_lab_id(str(row.get("Sample ID", "")))
        if not lab_id: continue
        
        report = db.query(Report).filter_by(lab_id=lab_id).first()
        if not report:
            report = Report(lab_id=lab_id, client=str(row.get("Client", CLIENT_NAME)))
            db.add(report)
            created += 1
        else: updated += 1
        
        # Auto-create COC record if it doesn't exist
        coc = db.query(ChainOfCustody).filter_by(lab_id=lab_id).first()
        if not coc:
            db.add(ChainOfCustody(lab_id=lab_id, client_name=report.client, sample_name=str(row.get("Product Name", ""))))

    db.commit()
    db.close()
    flash(f"Processed: {created} new, {updated} updated.")
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
