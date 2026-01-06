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

# --- Original Report Model ---
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
    matrix = Column(String, nullable=True)
    asin = Column(String, nullable=True)
    product_weight_g = Column(String, nullable=True)
    sample_mrl = Column(String, nullable=True)
    sample_units = Column(String, nullable=True)
    sample_dilution = Column(String, nullable=True)
    sample_analyzed = Column(String, nullable=True)
    sample_qualifier = Column(String, nullable=True)
    # QC Storage
    acq_datetime = Column(String, nullable=True) # MB Accumulation
    sheet_name = Column(String, nullable=True)   # MS1 Accumulation
    ms1_pct_rec_limits = Column(String, nullable=True) # MSD Accumulation
    msd_pct_rpd_limit = Column(String, nullable=True)

# --- NEW Chain of Custody Model ---
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

# --- Audit Log Model ---
class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)
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
    finally:
        db.close()

def _normalize_lab_id(lab_id: str) -> str:
    s = (lab_id or "").strip()
    pattern = r'\s+[\-\+]?\d*\.?\d+(?:ppb|ppt|ng\/g|ug\/g|\s\d*|\s.*)?$'
    normalized = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
    return normalized or s

# ------------------- Routes -------------------

@app.route("/")
def home():
    if session.get("username"):
        return redirect(url_for("portal_choice"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    role = request.form.get("role")
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if role == "admin" and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"], session["role"] = username, "admin"
        return redirect(url_for("portal_choice"))
    elif role == "client" and username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["username"], session["role"], session["client_name"] = username, "client", CLIENT_NAME
        return redirect(url_for("portal_choice"))
    else:
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
    q = db.query(Report)
    if u["role"] == "client":
        q = q.filter(Report.client == u["client_name"])

    lab_id_filter = request.args.get("lab_id", "").strip()
    if lab_id_filter:
        q = q.filter(Report.lab_id == _normalize_lab_id(lab_id_filter))

    reports = q.order_by(Report.resulted_date.desc()).limit(500).all()
    db.close()
    return render_template("dashboard.html", user=u, reports=reports)

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
        old_status, new_status = record.status, request.form.get("status")
        if old_status != new_status:
            record.status = new_status
            log_action(u["username"], u["role"], "COC_UPDATE", f"Sample {record.lab_id} moved to {new_status}")
            db.commit()
            flash("Status Updated and Logged")
    history = db.query(AuditLog).filter(AuditLog.details.contains(record.lab_id)).order_by(AuditLog.at.desc()).all()
    db.close()
    return render_template("coc_edit.html", record=record, history=history)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# Keep your existing helper functions like _ingest_master_upload here...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
