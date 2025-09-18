
from __future__ import annotations
import os, json
from datetime import datetime
from dateutil import parser as dtparser

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ---------------- App & DB ----------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///portal.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------- Models ------------------
class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, unique=True, index=True, nullable=False)
    name = db.Column(db.String, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    role = db.Column(db.String, default="client")  # 'admin' or 'client'
    is_active = db.Column(db.Boolean, default=True)

    def get_id(self):
        return str(self.id)

class Report(db.Model):
    __tablename__ = "reports"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, index=True, nullable=False)
    specimen = db.Column(db.String, index=True, nullable=False)
    client = db.Column(db.String)
    report_id = db.Column(db.String)
    notes = db.Column(db.String)
    extra_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables & seed default users if empty
with app.app_context():
    db.create_all()

    # ---- Embedded seeding (for free Render without shell) ----
    if User.query.count() == 0:
        ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@envirolabs.local").lower()
        ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Enviro!123")
        # Create admin
        admin = User(
            email=ADMIN_EMAIL,
            name="Enviro Admin",
            role="admin",
            password_hash=generate_password_hash(ADMIN_PASSWORD),
            is_active=True,
        )
        db.session.add(admin)
        # Create a few client users
        default_pw = os.getenv("CLIENT_PASSWORD", "Client!123")
        for i in range(1, 6):
            u = User(
                email=f"client{i}@envirolabs.local",
                name=f"Client {i}",
                role="client",
                password_hash=generate_password_hash(default_pw),
                is_active=True,
            )
            db.session.add(u)
        db.session.commit()
        print(f"[SEED] Created admin {ADMIN_EMAIL} and 5 clients (default client pw set).")

# ---------------- Auth --------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))

# basic CSRF token
def csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = os.urandom(16).hex()
        session["_csrf_token"] = token
    return token
app.jinja_env.globals["csrf_token"] = csrf_token

# brand available everywhere
@app.context_processor
def inject_brand():
    return {"brand": "Enviro Labs"}

# --------------- Routes -------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("admin" if current_user.role == "admin" else "reports"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    mode = request.args.get("mode", "client").lower()  # 'admin' or 'client'
    if request.method == "POST":
        if request.form.get("csrf_token") != session.get("_csrf_token"):
            flash("Invalid CSRF token", "error")
            return redirect(url_for("login", mode=mode))
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            # enforce role based on mode
            if mode == "admin" and user.role != "admin":
                flash("This area is for Admin only. Use the Client Login.", "error")
                return redirect(url_for("login", mode="admin"))
            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("admin" if user.role == "admin" else "reports"))
        flash("Invalid credentials", "error")
    return render_template("login.html", mode=mode)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out.", "success")
    return redirect(url_for("login"))

@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if current_user.role != "admin":
        flash("You do not have permission to access Admin.", "error")
        return redirect(url_for("reports"))

    if request.method == "POST":
        if request.form.get("csrf_token") != session.get("_csrf_token"):
            flash("Invalid CSRF token", "error")
            return redirect(url_for("admin"))
        file = request.files.get("csv")
        if not file or file.filename == "":
            flash("Please select a CSV file.", "error")
            return redirect(url_for("admin"))
        try:
            df = pd.read_csv(file)
        except Exception:
            file.stream.seek(0)
            try:
                df = pd.read_excel(file)
            except Exception as e:
                flash(f"Upload failed: {e}", "error")
                return redirect(url_for("admin"))

        # column mapping helpers
        cols = {c.lower().strip(): c for c in df.columns}
        def pick(*names):
            for n in names:
                for k, orig in cols.items():
                    if k == n.lower() or k.replace(" ", "") == n.lower().replace(" ", ""):
                        return orig
            return None

        c_date = pick("date", "collection_date", "reported", "report_date")
        c_spec = pick("specimen", "specimen_name", "sample", "sample_name")
        c_client = pick("client", "patient", "site", "customer")
        c_rid = pick("report_id", "accession", "accession_id", "case_id")
        c_notes = pick("notes", "comment", "comments")

        missing = [name for name, col in [("date", c_date), ("specimen", c_spec)] if col is None]
        if missing:
            flash(f"Missing required columns: {', '.join(missing)}", "error")
            return redirect(url_for("admin"))

        created = 0
        for _, row in df.iterrows():
            raw_date = row[c_date]
            try:
                d = dtparser.parse(str(raw_date)).date()
            except Exception:
                continue
            specimen = str(row[c_spec]).strip()
            if not specimen:
                continue
            client = str(row[c_client]).strip() if c_client else None
            rid = str(row[c_rid]).strip() if c_rid else None
            notes = str(row[c_notes]).strip() if c_notes else None

            extra = {}
            for col in df.columns:
                if col not in {c_date, c_spec, c_client, c_rid, c_notes}:
                    val = row[col]
                    if pd.notna(val):
                        extra[col] = str(val)

            rpt = Report(
                date=datetime(d.year, d.month, d.day),
                specimen=specimen,
                client=client,
                report_id=rid,
                notes=notes,
                extra_json=json.dumps(extra) if extra else None,
            )
            db.session.add(rpt)
            created += 1
        db.session.commit()
        flash(f"Imported {created} rows.", "success")
        return redirect(url_for("admin"))

    total = Report.query.count()
    last10 = Report.query.order_by(Report.id.desc()).limit(10).all()
    return render_template("admin.html", total=total, last10=last10)

@app.route("/reports")
@login_required
def reports():
    q = Report.query
    specimen = request.args.get("specimen", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    if specimen:
        q = q.filter(Report.specimen.ilike(f"%{specimen}%"))
    if date_from:
        try:
            d1 = dtparser.parse(date_from).date()
            q = q.filter(Report.date >= datetime(d1.year, d1.month, d1.day))
        except Exception:
            flash("Invalid start date", "error")
    if date_to:
        try:
            d2 = dtparser.parse(date_to).date()
            q = q.filter(Report.date <= datetime(d2.year, d2.month, d2.day, 23, 59, 59))
        except Exception:
            flash("Invalid end date", "error")

    results = q.order_by(Report.date.desc(), Report.id.desc()).limit(500).all()
    return render_template("reports.html", results=results, specimen=specimen, date_from=date_from, date_to=date_to)

@app.route("/reports/export")
@login_required
def export_reports():
    q = Report.query
    specimen = request.args.get("specimen", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    if specimen:
        q = q.filter(Report.specimen.ilike(f"%{specimen}%"))
    if date_from:
        d1 = dtparser.parse(date_from).date()
        q = q.filter(Report.date >= datetime(d1.year, d1.month, d1.day))
    if date_to:
        d2 = dtparser.parse(date_to).date()
        q = q.filter(Report.date <= datetime(d2.year, d2.month, d2.day, 23, 59, 59))

    rows = q.order_by(Report.date.desc(), Report.id.desc()).all()
    payload = []
    for r in rows:
        extra = json.loads(r.extra_json) if r.extra_json else {}
        payload.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "specimen": r.specimen,
            "client": r.client or "",
            "report_id": r.report_id or "",
            "notes": r.notes or "",
            **extra,
        })

    if not payload:
        flash("No data to export.", "error")
        return redirect(url_for("reports"))

    out = "export.csv"
    pd.DataFrame(payload).to_csv(out, index=False)
    return send_file(out, as_attachment=True)

# Local run (Render uses gunicorn entrypoint)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

