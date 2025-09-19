# Enviro Labs LIMS — Formula Hook Add‑On

This package lets your **admin CSV uploads** automatically compute derived values
and show them on each report.

## 1) Add this file
Copy `formula_hooks.py` to the **repo root** (same folder as `app.py`).

## 2) Edit `app.py`

### 2.1 Imports (top of file)
Add:
```python
import json
from formula_hooks import compute_fields
```

### 2.2 Database model
In `class Report(Base)`, add a column:
```python
computed = Column(Text, nullable=True)  # JSON of derived fields
```

### 2.3 Lightweight migration (after `Base.metadata.create_all(engine)`)
Add:
```python
with engine.begin() as conn:
    try:
        cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(reports)").fetchall()]
        if 'computed' not in cols:
            conn.exec_driver_sql("ALTER TABLE reports ADD COLUMN computed TEXT")
    except Exception:
        pass
```

### 2.4 Compute during CSV upload
Inside your CSV import loop (after you build `lab_id`, `client`, etc.) add:
```python
# Make a case-preserving dict of the raw row
raw_row = {str(k): (None if (pd.isna(row[k])) else row[k]) for k in df.columns}

# Run your formulas
try:
    computed_fields = compute_fields(raw_row) or {}
except Exception as e:
    computed_fields = {"_formula_error": str(e)[:300]}

# When creating/updating the Report object, persist computed JSON
report.computed = json.dumps(computed_fields, ensure_ascii=False)
# (or existing.computed = ... if you're updating an existing record)
```

### 2.5 Pass computed values to the template
In your `report_detail` route, before `render_template(...)`:
```python
computed = {}
try:
    if r.computed:
        computed = json.loads(r.computed)
except Exception:
    computed = {"_parse_error": True}

return render_template("report_detail.html", user=u, r=r, computed=computed)
```

### 2.6 Include computed fields in Export CSV (optional)
When exporting rows to CSV, flatten `computed` into extra columns:
```python
base = { ... existing columns ... }
try:
    comp = json.loads(r.computed) if r.computed else {}
    for k, v in comp.items():
        base[f"Computed: {k}"] = v
except Exception:
    base["Computed: _parse_error"] = True
```

## 3) Template update
In `templates/report_detail.html`, add this block where you want computed fields to render:
```html
{% if computed and computed|length %}
<section class="card">
  <h3>Derived Values</h3>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>
        {% for k, v in computed.items() %}
        <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</section>
{% endif %}
```

(optional) In your CSS, make it pretty:
```css
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin-top:12px}
.card h3{margin:0 0 8px 0}
.tablewrap table{width:100%;border-collapse:collapse}
.tablewrap th,.tablewrap td{border:1px solid #e5e7eb;padding:8px 10px;text-align:left}
.tablewrap thead th{background:#f7f9fb;color:#374151}
```

## 4) Deploy
- Commit changes to `main`.
- In Render, **Clear build cache** → **Deploy**.
- Upload a CSV as Admin. The **Derived Values** section should populate.

---

### Writing your formulas

Open `formula_hooks.py` and modify `compute_fields(row)`.
You can reference columns by their header names exactly as they appear in your CSV.
Helpers available: `get(row, name)`, `to_float(x)`, and `contains(text, needle)`.

Example:
```python
def compute_fields(row):
    fields = {}
    value = to_float(get(row, "PFAS Total"))
    action = to_float(get(row, "ActionLevel", 70))
    if value is not None and action is not None:
        fields["Interpretation"] = "Above AL" if value > action else "Within Range"
    return fields
```
