import os
import io
import re
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
import json

# ------------------- Config -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro#123")
CLIENT_USERNAME = os.getenv("CLIENT_USERNAME", "client")
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD", "Client#123")
CLIENT_NAME = os.getenv("CLIENT_NAME", "Center for Consumer Safety LLC")

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- App Initialization -------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ------------------- DB Setup -------------------
DB_PATH = os.path.join(BASE_DIR, "app.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ------------------- Models -------------------

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, unique=True, index=True)
    client = Column(String)
    patient = Column(String)
    test = Column(String)
    result = Column(String)
    collected_date = Column(Date)
    resulted_date = Column(Date)
    pdf_url = Column(String)
    sample_mrl = Column(String)
    sample_units = Column(String)
    sample_dilution = Column(String)
    sample_analyzed = Column(String)
    sample_qualifier = Column(String)
    computed = Column(Text) 

class ChainOfCustody(Base):
    __tablename__ = "coc_records"
    id = Column(Integer, primary_key=True)
    lab_id = Column(String, unique=True, index=True)
    client_name = Column(String)
    project_lead = Column(String)
    sample_name = Column(String)
    asin = Column(String)
    sample_type = Column(String)
    received_at = Column(DateTime)
    received_by = Column(String)
    condition = Column(String)
    status = Column(String, default="Received") 
    location = Column(String, default="Intake Storage")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    role = Column(String)
    action = Column(String)
    details = Column(Text)
    at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ------------------- Helpers -------------------

def log_action(user, role, action, details):
    db = SessionLocal()
    log = AuditLog(username=user, role=role, action=action, details=details)
    db.add(log)
    db.commit()
    db.close()

def get_session_user():
    return {
        "username": session.get("user"),
        "role": session.get("role"),
        "client_name": session.get("client_name")
    }

# ------------------- Routes -------------------

@app.route("/")
def home():
    if session.get("user"):
        return redirect(url_for("portal_choice"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["user"] = username
        session["role"] = "admin"
        log_action(username, "admin", "login", "Admin logged in")
        return redirect(url_for("portal_choice"))
    elif username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        session["user"] = username
        session["role"] = "client"
        session["client_name"] = CLIENT_NAME
        log_action(username, "client", "login", f"Client '{CLIENT_NAME}' logged in")
        return redirect(url_for("portal_choice"))
    
    flash("Invalid Credentials", "error")
    return redirect(url_for("home"))

@app.route("/portal")
def portal_choice():
    u = get_session_user()
    if not u["username"]: return redirect(url_for("home"))
    return render_template("portal_choice.html", user=u)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/coc")
def coc_list():
    u = get_session_user()
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
    u = get_session_user()
    if u["role"] != "admin": return "Unauthorized", 403
    db = SessionLocal()
    record = db.query(ChainOfCustody).get(record_id)
    
    if request.method == "POST":
        old_status = record.status
        old_loc = record.location
        new_status = request.form.get("status")
        new_loc = request.form.get("location")
        
        if old_status != new_status or old_loc != new_loc:
            record.status = new_status
            record.location = new_loc
            log_action(u["username"], u["role"], "COC_UPDATE", 
                       f"Updated Sample {record.lab_id}: Status {old_status}->{new_status}, Loc {old_loc}->{new_loc}")
            db.commit()
            flash("Chain of Custody Updated", "success")

    history = db.query(AuditLog).filter(AuditLog.details.contains(record.lab_id)).order_by(AuditLog.at.desc()).all()
    db.close()
    return render_template("coc_edit.html", record=record, history=history)

# Placeholder for existing dashboard/report routes
@app.route("/dashboard")
def dashboard():
    u = get_session_user()
    if not u["username"]: return redirect(url_for("home"))
    return "Lab Reports Dashboard Placeholder - Please integrate your previous report logic here."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
