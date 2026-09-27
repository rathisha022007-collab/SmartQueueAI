
from flask import Flask, request, jsonify, redirect, url_for, render_template_string
from flask_cors import CORS

import sqlite3
import os
from datetime import datetime, timedelta
import io
from flask import Response
import qrcode

import numpy as np
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "SmartQueue AI"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "smartqueue.db")

PORT = int(os.environ.get("PORT", 5000))

HISTORY_DAYS = 7
NO_SHOW_SECONDS = 120
AVERAGE_SERVICE_TIME = 6

# Queue categories and their related services.
# The student first selects a category (Bank / Hospital / College),
# then the service dropdown changes automatically.
CATEGORY_SERVICES = {
    "Bank": [
        "Cash Deposit",
        "Cash Withdrawal",
        "Account Opening",
        "Passbook Update",
        "KYC Update",
        "Loan Enquiry"
    ],
    "Hospital": [
        "OP Registration",
        "Doctor Consultation",
        "Lab Test",
        "Pharmacy",
        "Billing",
        "Insurance Desk"
    ],
    "College": [
        "Admission",
        "Exam Cell",
        "Fees Payment",
        "Certificate Request",
        "Bonafide Certificate",
        "Scholarship",
        "Placement Cell"
    ]
}

SERVICES = [service for items in CATEGORY_SERVICES.values() for service in items]

COUNTERS = 3


# ============================================================
# APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# DATABASE
# ============================================================

def db():

    connection = sqlite3.connect(DB_FILE)

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = db()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS counters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            current_token TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT NOT NULL,
            token_number INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            service TEXT NOT NULL,
            queue_type TEXT DEFAULT 'College',
            status TEXT NOT NULL,
            counter_id INTEGER,
            joined_at TEXT NOT NULL,
            called_at TEXT,
            completed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT NOT NULL,
            user_name TEXT NOT NULL,
            queue_type TEXT DEFAULT 'College',
            service TEXT NOT NULL,
            counter_id INTEGER,
            joined_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        )
    """)

    # Upgrade older databases safely.
    columns = [r["name"] for r in cursor.execute("PRAGMA table_info(queue)").fetchall()]
    if "priority" not in columns:
        cursor.execute("ALTER TABLE queue ADD COLUMN priority INTEGER DEFAULT 0")
    if "arrived" not in columns:
        cursor.execute("ALTER TABLE queue ADD COLUMN arrived INTEGER DEFAULT 0")
    if "queue_type" not in columns:
        cursor.execute("ALTER TABLE queue ADD COLUMN queue_type TEXT DEFAULT 'College'")

    history_columns = [r["name"] for r in cursor.execute("PRAGMA table_info(history)").fetchall()]
    if "queue_type" not in history_columns:
        cursor.execute("ALTER TABLE history ADD COLUMN queue_type TEXT DEFAULT 'College'")

    # HARD RULE: one counter can have only ONE serving person at a time.
    # This database constraint also protects against two admin/browser requests
    # arriving at almost the same time.
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_serving_per_counter
        ON queue(counter_id)
        WHERE status = 'serving' AND counter_id IS NOT NULL
    """)

    count = cursor.execute(
        "SELECT COUNT(*) AS total FROM counters"
    ).fetchone()["total"]

    if count == 0:

        for number in range(1, COUNTERS + 1):

            cursor.execute(
                """
                INSERT INTO counters
                (name, active, current_token)
                VALUES (?, 1, NULL)
                """,
                (f"Counter {number}",)
            )

    connection.commit()

    connection.close()


# ============================================================
# HISTORY CLEANUP
# ============================================================

def clean_history():

    connection = db()

    cutoff = (
        datetime.now() -
        timedelta(days=HISTORY_DAYS)
    ).isoformat()

    connection.execute(
        """
        DELETE FROM history
        WHERE completed_at < ?
        """,
        (cutoff,)
    )

    connection.commit()

    connection.close()


# ============================================================
# ML MODEL
# ============================================================

def create_model():

    X = []
    y = []

    for people in range(0, 51):

        for counters in range(1, 6):

            for service_time in [4, 5, 6, 7, 8, 10]:

                waiting = (
                    people *
                    service_time
                ) / counters

                X.append([
                    people,
                    counters,
                    service_time
                ])

                y.append(waiting)

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model


MODEL = create_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_wait(people_ahead, active_counters):

    if active_counters < 1:

        active_counters = 1

    average_service_time = 6

    prediction = MODEL.predict([
        [
            people_ahead,
            active_counters,
            average_service_time
        ]
    ])[0]

    return max(
        0,
        int(round(prediction))
    )


# ============================================================
# TOKEN
# ============================================================

def generate_token(service):

    connection = db()

    row = connection.execute(
        """
        SELECT MAX(token_number) AS maximum
        FROM queue
        WHERE service = ?
        """,
        (service,)
    ).fetchone()

    last_number = row["maximum"] or 0

    number = last_number + 1

    prefix = service[:2].upper()

    token = f"{prefix}-{number:03d}"

    connection.close()

    return token, number


# ============================================================
# QUEUE INFORMATION
# ============================================================

def queue_info(queue_id):

    connection = db()

    customer = connection.execute(
        "SELECT * FROM queue WHERE id = ?",
        (queue_id,)
    ).fetchone()

    if customer is None:
        connection.close()
        return None

    # Priority queue: higher priority first, then earlier token.
    ahead = connection.execute(
        """
        SELECT COUNT(*) AS total
        FROM queue
        WHERE status = 'waiting'
        AND (priority > ? OR (priority = ? AND id < ?))
        """,
        (customer["priority"] or 0, customer["priority"] or 0, queue_id)
    ).fetchone()["total"]

    active = connection.execute(
        "SELECT COUNT(*) AS total FROM counters WHERE active = 1"
    ).fetchone()["total"]
    active = max(1, active)

    if customer["status"] == "waiting":
        people_ahead = ahead
        position = ahead + 1
    else:
        people_ahead = 0
        position = 0

    wait = predict_wait(people_ahead, active)
    turn_time = (datetime.now() + timedelta(minutes=wait)).strftime("%I:%M %p")

    counter_name = None
    if customer["counter_id"]:
        counter = connection.execute(
            "SELECT name FROM counters WHERE id = ?",
            (customer["counter_id"],)
        ).fetchone()
        if counter:
            counter_name = counter["name"]

    waiting = connection.execute(
        "SELECT COUNT(*) AS total FROM queue WHERE status='waiting'"
    ).fetchone()["total"]
    serving = connection.execute(
        "SELECT COUNT(*) AS total FROM queue WHERE status='serving'"
    ).fetchone()["total"]
    total_live = waiting + serving
    load_ratio = total_live / active
    if load_ratio <= 3:
        load_label = "LOW"
        load_pct = min(100, int(load_ratio / 3 * 100))
    elif load_ratio <= 7:
        load_label = "MEDIUM"
        load_pct = min(100, int(load_ratio / 7 * 100))
    else:
        load_label = "HIGH"
        load_pct = 100

    # Smart counter recommendation: prefer a free counter, otherwise the least busy one.
    counters = connection.execute(
        "SELECT * FROM counters WHERE active=1 ORDER BY id"
    ).fetchall()
    best_name = None
    best_score = 10**9
    for c in counters:
        busy = connection.execute(
            "SELECT COUNT(*) AS total FROM queue WHERE status='serving' AND counter_id=?",
            (c["id"],)
        ).fetchone()["total"]
        score = -1 if busy == 0 else busy * AVERAGE_SERVICE_TIME
        if score < best_score:
            best_score = score
            best_name = c["name"]

    # Simple AI crowd forecast based on the same Random Forest idea used by the wait model.
    forecast = int(round(
        max(0, waiting * 0.75 + (10 if 9 <= datetime.now().hour <= 11 else 5))
    ))

    no_show_remaining = None
    if customer["status"] == "serving" and customer["called_at"]:
        elapsed = int((datetime.now() - datetime.fromisoformat(customer["called_at"])).total_seconds())
        no_show_remaining = max(0, NO_SHOW_SECONDS - elapsed)

    connection.close()

    return {
        "id": customer["id"],
        "token": customer["token"],
        "name": customer["user_name"],
        "queue_type": customer["queue_type"],
        "service": customer["service"],
        "status": customer["status"],
        "counter": counter_name,
        "people_ahead": people_ahead,
        "position": position,
        "wait": wait,
        "turn_time": turn_time,
        "active_counters": active,
        "service_time": AVERAGE_SERVICE_TIME,
        "queue_load": load_label,
        "queue_load_pct": load_pct,
        "recommended_counter": best_name,
        "forecast_30m": forecast,
        "priority": bool(customer["priority"] or 0),
        "arrived": bool(customer["arrived"] or 0),
        "no_show_remaining": no_show_remaining
    }



# ============================================================
# ANALYTICS PAGE
# ============================================================
ANALYTICS = r"""
<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartQueue Analytics</title>
<style>body{margin:0;background:#080a16;color:white;font-family:Arial}.header{padding:20px 7%;border-bottom:1px solid #22263a}.header a{color:white;text-decoration:none}.container{width:92%;max-width:1200px;margin:30px auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.card{background:#111426;border:1px solid #252a40;border-radius:18px;padding:22px}.big{font-size:32px;font-weight:900;color:#8b7cff}.section{margin-top:18px;background:#111426;border:1px solid #252a40;border-radius:18px;padding:22px}.bars{display:grid;grid-template-columns:repeat(24,1fr);gap:5px;height:210px;align-items:end}.barbox{height:100%;display:flex;flex-direction:column;justify-content:end;text-align:center;font-size:10px;color:#858ca0}.bar{background:linear-gradient(#8b7cff,#43e8ff);min-height:5px;border-radius:5px 5px 0 0}.servicebar{height:10px;background:#292e40;border-radius:10px;overflow:hidden;margin:5px 0 12px}.servicefill{height:100%;background:#55e6aa}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.panel{background:#0d1020;border:1px solid #252a40;border-radius:14px;padding:16px}.pill{display:inline-block;padding:7px 12px;border-radius:20px;background:#25293c;margin:4px}.low{border-left:5px solid #55e6aa}.medium{border-left:5px solid #ffd166}.high{border-left:5px solid #ff667d}.note{color:#8e95a9;line-height:1.6}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.bars{grid-template-columns:repeat(12,1fr)}} </style><style id="smartqueue-aurora-theme">
:root{
  --sq-bg:#050816;--sq-panel:rgba(12,18,38,.78);--sq-panel2:rgba(18,25,51,.82);
  --sq-border:rgba(132,151,255,.20);--sq-text:#f7f9ff;--sq-muted:#98a4c4;
  --sq-cyan:#38e8ff;--sq-violet:#8b6cff;--sq-pink:#ff4fd8;--sq-green:#45e6a7;
  --sq-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at 10% 10%,rgba(56,232,255,.10),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(139,108,255,.14),transparent 30%),
    radial-gradient(circle at 50% 100%,rgba(255,79,216,.08),transparent 30%),
    var(--sq-bg)!important;
  color:var(--sq-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;
  min-height:100vh;position:relative;overflow-x:hidden;
}
body:before{content:"";position:fixed;inset:-35%;pointer-events:none;z-index:-1;background:conic-gradient(from 180deg at 50% 50%,transparent,rgba(56,232,255,.045),transparent 28%,rgba(139,108,255,.055),transparent 60%,rgba(255,79,216,.04),transparent);animation:sqSpin 22s linear infinite}
body:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 85%)}
@keyframes sqSpin{to{transform:rotate(360deg)}}
@keyframes sqFloat{50%{transform:translateY(-5px)}}
@keyframes sqPulse{50%{box-shadow:0 0 0 7px rgba(56,232,255,.02),0 0 28px rgba(56,232,255,.22)}}
.navbar,.header{background:rgba(5,8,22,.72)!important;border-color:var(--sq-border)!important;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;z-index:5}
.logo{letter-spacing:.2px!important}.logo:before{content:"◉";display:inline-grid;place-items:center;width:30px;height:30px;margin-right:9px;border-radius:10px;background:linear-gradient(135deg,var(--sq-cyan),var(--sq-violet));box-shadow:0 0 25px rgba(56,232,255,.28);font-size:12px;color:white}
.logo span,.gradient,.big,.token{color:var(--sq-cyan)!important;background:linear-gradient(90deg,var(--sq-cyan),#a68bff 48%,var(--sq-pink));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.nav a,.header a{transition:.2s;color:#cdd5f1!important}.nav a:hover,.header a:hover{color:white!important;text-shadow:0 0 16px rgba(56,232,255,.55)}
.hero{position:relative}.hero:before{content:"";position:absolute;width:280px;height:280px;right:4%;top:8%;border-radius:50%;background:radial-gradient(circle,rgba(139,108,255,.20),transparent 68%);filter:blur(8px);pointer-events:none}
.badge{border:1px solid rgba(56,232,255,.28)!important;background:rgba(56,232,255,.07)!important;box-shadow:0 0 25px rgba(56,232,255,.08);backdrop-filter:blur(10px)}
.main-card,.preview,.mini,.feature,.card,.section,.panel,.container>div,.pass,.table-wrap,.admin-card,.stat-card{
  background:linear-gradient(145deg,rgba(18,26,54,.88),rgba(8,13,30,.82))!important;
  border:1px solid var(--sq-border)!important;box-shadow:var(--sq-shadow)!important;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.main-card{border-radius:28px!important;position:relative;overflow:hidden}.main-card:before{content:"";position:absolute;left:-20%;top:-30%;width:55%;height:70%;background:radial-gradient(circle,rgba(56,232,255,.14),transparent 68%);pointer-events:none}
.mini,.feature,.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}.mini:hover,.feature:hover,.card:hover,.panel:hover{transform:translateY(-4px);border-color:rgba(56,232,255,.36)!important;box-shadow:0 20px 45px rgba(0,0,0,.30),0 0 28px rgba(56,232,255,.07)!important}
input,select,textarea{background:rgba(4,8,20,.72)!important;color:#f7f9ff!important;border:1px solid rgba(132,151,255,.24)!important;border-radius:13px!important;outline:none!important;transition:.2s!important}
input:focus,select:focus,textarea:focus{border-color:var(--sq-cyan)!important;box-shadow:0 0 0 4px rgba(56,232,255,.08),0 0 24px rgba(56,232,255,.10)!important}
input::placeholder{color:#697594!important}
button,.join,[type=submit]{border:0!important;border-radius:14px!important;background:linear-gradient(100deg,var(--sq-violet),#596eff 48%,var(--sq-cyan))!important;color:white!important;font-weight:800!important;box-shadow:0 12px 30px rgba(91,103,255,.24)!important;transition:transform .18s,box-shadow .18s,filter .18s!important;cursor:pointer}
button:hover,.join:hover,[type=submit]:hover{transform:translateY(-2px)!important;filter:saturate(1.15)!important;box-shadow:0 16px 38px rgba(56,232,255,.18)!important}
.feature-icon{background:linear-gradient(135deg,rgba(56,232,255,.18),rgba(139,108,255,.18))!important;border:1px solid rgba(56,232,255,.18)!important;box-shadow:0 0 24px rgba(56,232,255,.08)}
.status,.pill,.badge,.tag{border:1px solid rgba(132,151,255,.18)!important;background:rgba(255,255,255,.055)!important;border-radius:999px!important}
.low{border-left-color:var(--sq-green)!important}.medium{border-left-color:#ffd166!important}.high{border-left-color:#ff5b78!important}
.servicebar,.barbox>div[style*="background"],#crowdBar{background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink))!important}
.bars .bar{background:linear-gradient(180deg,var(--sq-cyan),var(--sq-violet))!important;box-shadow:0 0 15px rgba(56,232,255,.18)}
table{border-collapse:separate!important;border-spacing:0 8px!important;width:100%!important}th{color:#8e9abc!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.08em}td{background:rgba(12,18,38,.72)!important;border-top:1px solid rgba(132,151,255,.12)!important;border-bottom:1px solid rgba(132,151,255,.12)!important}td:first-child{border-left:1px solid rgba(132,151,255,.12)!important;border-radius:12px 0 0 12px}td:last-child{border-right:1px solid rgba(132,151,255,.12)!important;border-radius:0 12px 12px 0}
.container{position:relative}.note,.small,.hero-text{color:var(--sq-muted)!important}
.qr{box-shadow:0 0 0 8px rgba(255,255,255,.035),0 18px 40px rgba(0,0,0,.35)!important}
.pass{border-radius:30px!important;position:relative;overflow:hidden}.pass:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:6px;background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink));box-shadow:0 0 28px rgba(56,232,255,.4)}
.hero h1{letter-spacing:-.045em;text-shadow:0 8px 35px rgba(139,108,255,.15)}
.features h2,h1,h2,h3{letter-spacing:-.02em}
@media(max-width:800px){.navbar,.header{padding-left:5%!important;padding-right:5%!important}.hero{grid-template-columns:1fr!important}.cards{grid-template-columns:1fr!important}.grid{grid-template-columns:1fr!important}.feature-grid{grid-template-columns:1fr!important}.bars{grid-template-columns:repeat(12,1fr)!important}.main-card{margin-top:10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>

</head>
<body><div class="header"><a href="/">← SmartQueue AI</a> &nbsp; <a href="/admin">Admin</a></div>
<div class="container"><h1>📊 SmartQueue Analytics</h1>
<div class="cards"><div class="card"><div class="big" id="today">0</div>Completed Today</div><div class="card"><div class="big" id="avg">0</div>Avg Queue Time (min)</div><div class="card"><div class="big" id="density">-</div>Crowd Density</div><div class="card"><div class="big" id="peak">-</div>Peak Hour</div></div>
<div class="grid">
<div class="panel"><h2>🧭 Predicted Peak Time</h2><div class="big" id="predictedPeak">-</div><p class="note">Predicted from hourly history and current live queue pressure for the next 6 hours.</p></div>
<div class="panel"><h2>⚖️ Automatic Counter Balancing</h2><div id="counters"></div><p class="note" id="counterAdvice">-</p></div>
</div>
<div class="section"><h2>🔥 Queue Heatmap / Hourly Activity</h2><div class="bars" id="bars"></div></div>
<div class="section"><h2>🧾 Service-wise Usage</h2><div id="services"></div></div>
<div class="section"><h2>👥 Live Crowd Density</h2><p id="crowdText" class="note"></p><div style="height:18px;background:#292e40;border-radius:20px;overflow:hidden"><div id="crowdBar" style="height:100%;width:0%;background:linear-gradient(90deg,#55e6aa,#ffd166,#ff667d)"></div></div></div>
<div class="section"><h2>🤖 AI Crowd Forecast</h2><p id="forecast" class="note"></p></div>
<div class="section"><p class="note">The demo prediction uses available queue history and live queue pressure. More real queue records can improve future predictions.</p></div>
</div>
<script>fetch('/api/analytics').then(r=>r.json()).then(d=>{document.getElementById('today').innerText=d.today_completed;document.getElementById('avg').innerText=d.avg_wait;document.getElementById('density').innerText=d.crowd_density;document.getElementById('peak').innerText=d.peak_hour;document.getElementById('predictedPeak').innerText=d.predicted_peak_hour;let max=Math.max(1,...d.hourly.map(x=>x.count));document.getElementById('bars').innerHTML=d.hourly.map(x=>`<div class="barbox"><div class="bar" style="height:${Math.max(5,x.count/max*185)}px"></div>${String(x.hour).padStart(2,'0')}</div>`).join('');let sm=Math.max(1,...d.services.map(x=>x.count));document.getElementById('services').innerHTML=d.services.map(x=>`<b>${x.service}</b> — ${x.count}<div class="servicebar"><div class="servicefill" style="width:${x.count/sm*100}%"></div></div>`).join('');document.getElementById('forecast').innerText=`Current waiting: ${d.current_waiting} people · Predicted in 30 minutes: ${d.forecast_30m} people · Load: ${d.load}`;document.getElementById('crowdText').innerText=`${d.current_waiting} waiting + ${d.counter_status.filter(x=>x.busy).length} serving · ${d.crowd_density} density · ${d.crowd_density_pct}% pressure`;document.getElementById('crowdBar').style.width=d.crowd_density_pct+'%';document.getElementById('counterAdvice').innerText='Recommended free counter: '+d.recommended_free_counter;document.getElementById('counters').innerHTML=d.counter_status.map(c=>`<span class="pill ${c.busy?'high':'low'}">${c.name}: ${c.busy?'BUSY '+c.token:'FREE'}</span>`).join('');});</script></body></html>
"""

# ============================================================
# HOME PAGE
# ============================================================

HOME = r"""
<!DOCTYPE html>

<html>

<head>

<title>SmartQueue AI</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        Arial,
        sans-serif;

    background: #080a16;

    color: white;
}

.navbar {

    height: 75px;

    padding: 0 7%;

    display: flex;

    align-items: center;

    justify-content: space-between;

    border-bottom:
        1px solid rgba(255,255,255,.08);

    background:
        rgba(8,10,22,.85);

    backdrop-filter: blur(15px);

}

.logo {

    font-size: 23px;

    font-weight: 800;

    letter-spacing: .5px;

}

.logo span {

    color: #8b7cff;

}

.nav a {

    color: #c7c9d8;

    text-decoration: none;

    margin-left: 25px;

    font-size: 14px;

}

.hero {

    min-height:
        calc(100vh - 75px);

    display: grid;

    grid-template-columns:
        1.15fr .85fr;

    gap: 60px;

    padding:
        70px 7%;

    align-items: center;

    background:

        radial-gradient(
            circle at 15% 20%,
            rgba(125,92,255,.25),
            transparent 35%
        ),

        radial-gradient(
            circle at 80% 80%,
            rgba(0,220,255,.12),
            transparent 30%
        );

}

.badge {

    display: inline-block;

    padding: 8px 14px;

    border-radius: 50px;

    background:
        rgba(139,124,255,.12);

    border:
        1px solid rgba(139,124,255,.3);

    color: #aaa1ff;

    font-size: 13px;

    margin-bottom: 20px;

}

.hero h1 {

    font-size:
        clamp(42px, 6vw, 76px);

    line-height: 1.03;

    margin: 0;

}

.gradient {

    background:
        linear-gradient(
            90deg,
            #8b7cff,
            #43e8ff
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

}

.hero-text {

    color: #a9adbf;

    max-width: 600px;

    line-height: 1.7;

    font-size: 17px;

    margin:
        25px 0 30px;

}

.main-card {

    background:
        rgba(255,255,255,.06);

    border:
        1px solid rgba(255,255,255,.1);

    border-radius: 28px;

    padding: 32px;

    box-shadow:
        0 25px 80px
        rgba(0,0,0,.35);

}

.main-card h2 {

    margin-top: 0;

}

label {

    display: block;

    color: #cfd1dd;

    margin:
        17px 0 8px;

    font-size: 14px;

}

input,
select {

    width: 100%;

    padding: 15px;

    border-radius: 12px;

    border:
        1px solid #2d3043;

    background: #111426;

    color: white;

    outline: none;

    font-size: 15px;

}

input:focus,
select:focus {

    border-color:
        #8b7cff;

}

.join {

    width: 100%;

    margin-top: 22px;

    padding: 15px;

    border: none;

    border-radius: 12px;

    background:
        linear-gradient(
            90deg,
            #7564ff,
            #36d9f7
        );

    color: white;

    font-size: 16px;

    font-weight: 800;

    cursor: pointer;

}

.join:hover {

    transform:
        translateY(-1px);

}

.preview {

    margin-top: 25px;

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 12px;

}

.mini {

    padding: 18px;

    border-radius: 15px;

    background:
        rgba(255,255,255,.045);

    border:
        1px solid rgba(255,255,255,.07);

}

.mini strong {

    display: block;

    font-size: 25px;

    margin-bottom: 5px;

}

.mini small {

    color: #9095aa;

}

.features {

    padding:
        70px 7%;

    background: #0d1020;

}

.features h2 {

    font-size: 35px;

}

.feature-grid {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 18px;

}

.feature {

    padding: 25px;

    background:
        rgba(255,255,255,.04);

    border:
        1px solid rgba(255,255,255,.07);

    border-radius: 20px;

}

.feature-icon {

    font-size: 30px;

}

.feature h3 {

    margin-bottom: 8px;

}

.feature p {

    color: #9297ab;

    line-height: 1.6;

    font-size: 14px;

}

.footer {

    padding: 25px;

    text-align: center;

    color: #686d82;

}

@media(max-width:900px) {

    .hero {

        grid-template-columns: 1fr;

    }

    .feature-grid {

        grid-template-columns:
            1fr 1fr;

    }

}

@media(max-width:600px) {

    .nav {

        display: none;

    }

    .feature-grid {

        grid-template-columns: 1fr;

    }

}

</style>

<style id="smartqueue-aurora-theme">
:root{
  --sq-bg:#050816;--sq-panel:rgba(12,18,38,.78);--sq-panel2:rgba(18,25,51,.82);
  --sq-border:rgba(132,151,255,.20);--sq-text:#f7f9ff;--sq-muted:#98a4c4;
  --sq-cyan:#38e8ff;--sq-violet:#8b6cff;--sq-pink:#ff4fd8;--sq-green:#45e6a7;
  --sq-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at 10% 10%,rgba(56,232,255,.10),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(139,108,255,.14),transparent 30%),
    radial-gradient(circle at 50% 100%,rgba(255,79,216,.08),transparent 30%),
    var(--sq-bg)!important;
  color:var(--sq-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;
  min-height:100vh;position:relative;overflow-x:hidden;
}
body:before{content:"";position:fixed;inset:-35%;pointer-events:none;z-index:-1;background:conic-gradient(from 180deg at 50% 50%,transparent,rgba(56,232,255,.045),transparent 28%,rgba(139,108,255,.055),transparent 60%,rgba(255,79,216,.04),transparent);animation:sqSpin 22s linear infinite}
body:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 85%)}
@keyframes sqSpin{to{transform:rotate(360deg)}}
@keyframes sqFloat{50%{transform:translateY(-5px)}}
@keyframes sqPulse{50%{box-shadow:0 0 0 7px rgba(56,232,255,.02),0 0 28px rgba(56,232,255,.22)}}
.navbar,.header{background:rgba(5,8,22,.72)!important;border-color:var(--sq-border)!important;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;z-index:5}
.logo{letter-spacing:.2px!important}.logo:before{content:"◉";display:inline-grid;place-items:center;width:30px;height:30px;margin-right:9px;border-radius:10px;background:linear-gradient(135deg,var(--sq-cyan),var(--sq-violet));box-shadow:0 0 25px rgba(56,232,255,.28);font-size:12px;color:white}
.logo span,.gradient,.big,.token{color:var(--sq-cyan)!important;background:linear-gradient(90deg,var(--sq-cyan),#a68bff 48%,var(--sq-pink));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.nav a,.header a{transition:.2s;color:#cdd5f1!important}.nav a:hover,.header a:hover{color:white!important;text-shadow:0 0 16px rgba(56,232,255,.55)}
.hero{position:relative}.hero:before{content:"";position:absolute;width:280px;height:280px;right:4%;top:8%;border-radius:50%;background:radial-gradient(circle,rgba(139,108,255,.20),transparent 68%);filter:blur(8px);pointer-events:none}
.badge{border:1px solid rgba(56,232,255,.28)!important;background:rgba(56,232,255,.07)!important;box-shadow:0 0 25px rgba(56,232,255,.08);backdrop-filter:blur(10px)}
.main-card,.preview,.mini,.feature,.card,.section,.panel,.container>div,.pass,.table-wrap,.admin-card,.stat-card{
  background:linear-gradient(145deg,rgba(18,26,54,.88),rgba(8,13,30,.82))!important;
  border:1px solid var(--sq-border)!important;box-shadow:var(--sq-shadow)!important;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.main-card{border-radius:28px!important;position:relative;overflow:hidden}.main-card:before{content:"";position:absolute;left:-20%;top:-30%;width:55%;height:70%;background:radial-gradient(circle,rgba(56,232,255,.14),transparent 68%);pointer-events:none}
.mini,.feature,.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}.mini:hover,.feature:hover,.card:hover,.panel:hover{transform:translateY(-4px);border-color:rgba(56,232,255,.36)!important;box-shadow:0 20px 45px rgba(0,0,0,.30),0 0 28px rgba(56,232,255,.07)!important}
input,select,textarea{background:rgba(4,8,20,.72)!important;color:#f7f9ff!important;border:1px solid rgba(132,151,255,.24)!important;border-radius:13px!important;outline:none!important;transition:.2s!important}
input:focus,select:focus,textarea:focus{border-color:var(--sq-cyan)!important;box-shadow:0 0 0 4px rgba(56,232,255,.08),0 0 24px rgba(56,232,255,.10)!important}
input::placeholder{color:#697594!important}
button,.join,[type=submit]{border:0!important;border-radius:14px!important;background:linear-gradient(100deg,var(--sq-violet),#596eff 48%,var(--sq-cyan))!important;color:white!important;font-weight:800!important;box-shadow:0 12px 30px rgba(91,103,255,.24)!important;transition:transform .18s,box-shadow .18s,filter .18s!important;cursor:pointer}
button:hover,.join:hover,[type=submit]:hover{transform:translateY(-2px)!important;filter:saturate(1.15)!important;box-shadow:0 16px 38px rgba(56,232,255,.18)!important}
.feature-icon{background:linear-gradient(135deg,rgba(56,232,255,.18),rgba(139,108,255,.18))!important;border:1px solid rgba(56,232,255,.18)!important;box-shadow:0 0 24px rgba(56,232,255,.08)}
.status,.pill,.badge,.tag{border:1px solid rgba(132,151,255,.18)!important;background:rgba(255,255,255,.055)!important;border-radius:999px!important}
.low{border-left-color:var(--sq-green)!important}.medium{border-left-color:#ffd166!important}.high{border-left-color:#ff5b78!important}
.servicebar,.barbox>div[style*="background"],#crowdBar{background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink))!important}
.bars .bar{background:linear-gradient(180deg,var(--sq-cyan),var(--sq-violet))!important;box-shadow:0 0 15px rgba(56,232,255,.18)}
table{border-collapse:separate!important;border-spacing:0 8px!important;width:100%!important}th{color:#8e9abc!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.08em}td{background:rgba(12,18,38,.72)!important;border-top:1px solid rgba(132,151,255,.12)!important;border-bottom:1px solid rgba(132,151,255,.12)!important}td:first-child{border-left:1px solid rgba(132,151,255,.12)!important;border-radius:12px 0 0 12px}td:last-child{border-right:1px solid rgba(132,151,255,.12)!important;border-radius:0 12px 12px 0}
.container{position:relative}.note,.small,.hero-text{color:var(--sq-muted)!important}
.qr{box-shadow:0 0 0 8px rgba(255,255,255,.035),0 18px 40px rgba(0,0,0,.35)!important}
.pass{border-radius:30px!important;position:relative;overflow:hidden}.pass:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:6px;background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink));box-shadow:0 0 28px rgba(56,232,255,.4)}
.hero h1{letter-spacing:-.045em;text-shadow:0 8px 35px rgba(139,108,255,.15)}
.features h2,h1,h2,h3{letter-spacing:-.02em}
@media(max-width:800px){.navbar,.header{padding-left:5%!important;padding-right:5%!important}.hero{grid-template-columns:1fr!important}.cards{grid-template-columns:1fr!important}.grid{grid-template-columns:1fr!important}.feature-grid{grid-template-columns:1fr!important}.bars{grid-template-columns:repeat(12,1fr)!important}.main-card{margin-top:10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>

</head>

<body>

<div class="navbar">

    <div class="logo">
        SmartQueue <span>AI</span>
    </div>

    <div class="nav">

        <a href="/">Home</a>

        <a href="/admin">Admin</a>

        <a href="/analytics">Analytics</a>
        <a href="/history">History</a>

    </div>

</div>


<section class="hero">

<div>

    <div class="badge">
        ✦ AI POWERED QUEUE MANAGEMENT
    </div>

    <h1>
        Skip the line.<br>
        <span class="gradient">
            Know your time.
        </span>
    </h1>

    <p class="hero-text">

        SmartQueue AI lets users choose a queue type
        such as Bank, Hospital or College, then
        shows only the services related to that
        category with AI-powered waiting time.

    </p>


    <div class="preview">

        <div class="mini">

            <strong>AI</strong>

            <small>
                Waiting-time prediction
            </small>

        </div>

        <div class="mini">

            <strong>LIVE</strong>

            <small>
                Real-time queue tracking
            </small>

        </div>

    </div>

</div>


<div class="main-card">

    <h2>
        Join the Queue
    </h2>

    <p style="color:#8f94aa;">
        Enter your details and get
        your digital token instantly.
    </p>

    <form action="/join" method="POST">

        <label>
            Student Name
        </label>

        <input
            name="name"
            placeholder="Enter your name"
            required
        >

        <label>
            Select Queue Type
        </label>

        <select
            id="queue_type"
            name="queue_type"
            required
            onchange="updateServices()"
        >
            <option value="">Choose category</option>
            {% for category in category_services.keys() %}
            <option value="{{ category }}">{{ category }}</option>
            {% endfor %}
        </select>

        <label>
            Select Service
        </label>

        <select
            id="service"
            name="service"
            required
            disabled
        >
            <option value="">First choose a category</option>
        </select>

        <div id="serviceHint" style="margin-top:8px;color:#7f879f;font-size:12px;">
            Select Bank, Hospital or College to see its related services.
        </div>

        <script>
        const categoryServices = {{ category_services | tojson }};

        function updateServices() {
            const category = document.getElementById('queue_type').value;
            const service = document.getElementById('service');
            const hint = document.getElementById('serviceHint');

            service.innerHTML = '';

            if (!category) {
                service.disabled = true;
                service.innerHTML = '<option value="">First choose a category</option>';
                hint.textContent = 'Select Bank, Hospital or College to see its related services.';
                return;
            }

            service.disabled = false;
            service.innerHTML = '<option value="">Choose service</option>';

            categoryServices[category].forEach(function(item) {
                const option = document.createElement('option');
                option.value = item;
                option.textContent = item;
                service.appendChild(option);
            });

            hint.textContent = category + ' services available: ' + categoryServices[category].length;
        }
        </script>

        <label style="display:flex;align-items:center;gap:8px;color:#aeb4c8;margin-top:15px;">
            <input type="checkbox" name="priority" value="1" style="width:auto;">
            ⭐ Priority request (admin can verify)
        </label>

        <button class="join">
            GET DIGITAL TOKEN →
        </button>

        <div style="margin-top:18px;padding:15px;background:#0d1020;border:1px solid #292e45;border-radius:16px;text-align:center;">
            <img src="/qr" style="width:145px;height:145px;background:white;padding:7px;border-radius:10px;">
            <div style="font-size:12px;color:#8f96aa;margin-top:7px;">📱 Scan QR to open SmartQueue AI</div>
        </div>

    </form>

</div>

</section>


<section class="features">

<h2>
    One queue. Complete visibility.
</h2>

<div class="feature-grid">

<div class="feature">

    <div class="feature-icon">
        🎟️
    </div>

    <h3>
        Digital Token
    </h3>

    <p>
        Get your unique token without
        physically standing in a queue.
    </p>

</div>


<div class="feature">

    <div class="feature-icon">
        🤖
    </div>

    <h3>
        AI Prediction
    </h3>

    <p>
        Machine learning estimates
        your waiting time from live queue data.
    </p>

</div>


<div class="feature">

    <div class="feature-icon">
        📡
    </div>

    <h3>
        Live Tracking
    </h3>

    <p>
        See people ahead, position,
        status and counter in real time.
    </p>

</div>


<div class="feature">

    <div class="feature-icon">
        🔔
    </div>

    <h3>
        Smart Alerts
    </h3>

    <p>
        Get alerts as your turn gets
        closer and when you are called.
    </p>

</div>

</div>

</section>


<div class="footer">

    SmartQueue AI · Intelligent Queue Management

</div>

</body>

</html>
"""


# ============================================================
# STUDENT PAGE
# ============================================================

QUEUE = r"""
<!DOCTYPE html>

<html>

<head>

<title>{{ data.token }} - SmartQueue AI</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        sans-serif;

    background:
        #080a16;

    color: white;

}

.header {

    padding:
        20px 7%;

    border-bottom:
        1px solid #1d2031;

    display: flex;

    justify-content:
        space-between;

}

.header a {

    color: white;

    text-decoration: none;

}

.container {

    max-width: 1050px;

    margin:
        35px auto;

    padding:
        0 20px;

}

.top {

    text-align: center;

}

.top p {

    color: #9297aa;

}

.token-card {

    margin-top: 25px;

    padding: 35px;

    border-radius: 28px;

    text-align: center;

    background:
        linear-gradient(
            145deg,
            #15182c,
            #0d1020
        );

    border:
        1px solid #272a40;

}

.token-label {

    color: #858ba2;

    font-size: 13px;

    letter-spacing: 2px;

}

.token {

    font-size:
        clamp(50px,8vw,85px);

    font-weight: 900;

    margin:
        10px 0;

    background:
        linear-gradient(
            90deg,
            #8b7cff,
            #43e8ff
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

}

.status {

    display: inline-block;

    padding:
        8px 18px;

    border-radius: 30px;

    background:
        rgba(80,255,180,.1);

    color: #61f7b1;

    border:
        1px solid
        rgba(80,255,180,.2);

    font-weight: bold;

}


.predict {

    margin-top: 25px;

    padding: 28px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(117,100,255,.2),
            rgba(54,217,247,.08)
        );

    border:
        1px solid
        rgba(139,124,255,.25);

}

.predict-title {

    color: #aeb2c5;

    font-size: 14px;

}

.wait {

    font-size: 55px;

    font-weight: 900;

    margin:
        8px 0;

}

.wait span {

    font-size: 18px;

    color: #8d92a7;

}

.prediction-note {

    color: #777d93;

    font-size: 13px;

}

.grid {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 15px;

    margin-top: 18px;

}

.info {

    padding: 22px;

    border-radius: 18px;

    background: #111426;

    border:
        1px solid #22263a;

}

.info-number {

    font-size: 30px;

    font-weight: 800;

    color: #8b7cff;

}

.info-label {

    margin-top: 7px;

    color: #858ba0;

    font-size: 13px;

}

.notify {

    width: 100%;

    margin-top: 20px;

    padding: 15px;

    border: none;

    border-radius: 12px;

    background:
        #ffffff;

    color: #10121e;

    font-weight: 800;

    cursor: pointer;

}

.notify.enabled {

    background:
        #51e6a4;

}

.alert {

    display: none;

    margin-top: 18px;

    padding: 18px;

    border-radius: 15px;

    background:
        #251e43;

    border:
        1px solid #5547a0;

}

.refresh {

    text-align: center;

    color: #666c80;

    margin-top: 20px;

    font-size: 12px;

}

.home {

    display: inline-block;

    margin-top: 25px;

    padding:
        12px 20px;

    border-radius: 10px;

    background: #20243a;

    color: white;

    text-decoration: none;

}

@media(max-width:800px) {

    .grid {

        grid-template-columns:
            1fr 1fr;

    }

}

@media(max-width:500px) {

    .grid {

        grid-template-columns:
            1fr;

    }

}

</style>

<style id="smartqueue-aurora-theme">
:root{
  --sq-bg:#050816;--sq-panel:rgba(12,18,38,.78);--sq-panel2:rgba(18,25,51,.82);
  --sq-border:rgba(132,151,255,.20);--sq-text:#f7f9ff;--sq-muted:#98a4c4;
  --sq-cyan:#38e8ff;--sq-violet:#8b6cff;--sq-pink:#ff4fd8;--sq-green:#45e6a7;
  --sq-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at 10% 10%,rgba(56,232,255,.10),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(139,108,255,.14),transparent 30%),
    radial-gradient(circle at 50% 100%,rgba(255,79,216,.08),transparent 30%),
    var(--sq-bg)!important;
  color:var(--sq-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;
  min-height:100vh;position:relative;overflow-x:hidden;
}
body:before{content:"";position:fixed;inset:-35%;pointer-events:none;z-index:-1;background:conic-gradient(from 180deg at 50% 50%,transparent,rgba(56,232,255,.045),transparent 28%,rgba(139,108,255,.055),transparent 60%,rgba(255,79,216,.04),transparent);animation:sqSpin 22s linear infinite}
body:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 85%)}
@keyframes sqSpin{to{transform:rotate(360deg)}}
@keyframes sqFloat{50%{transform:translateY(-5px)}}
@keyframes sqPulse{50%{box-shadow:0 0 0 7px rgba(56,232,255,.02),0 0 28px rgba(56,232,255,.22)}}
.navbar,.header{background:rgba(5,8,22,.72)!important;border-color:var(--sq-border)!important;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;z-index:5}
.logo{letter-spacing:.2px!important}.logo:before{content:"◉";display:inline-grid;place-items:center;width:30px;height:30px;margin-right:9px;border-radius:10px;background:linear-gradient(135deg,var(--sq-cyan),var(--sq-violet));box-shadow:0 0 25px rgba(56,232,255,.28);font-size:12px;color:white}
.logo span,.gradient,.big,.token{color:var(--sq-cyan)!important;background:linear-gradient(90deg,var(--sq-cyan),#a68bff 48%,var(--sq-pink));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.nav a,.header a{transition:.2s;color:#cdd5f1!important}.nav a:hover,.header a:hover{color:white!important;text-shadow:0 0 16px rgba(56,232,255,.55)}
.hero{position:relative}.hero:before{content:"";position:absolute;width:280px;height:280px;right:4%;top:8%;border-radius:50%;background:radial-gradient(circle,rgba(139,108,255,.20),transparent 68%);filter:blur(8px);pointer-events:none}
.badge{border:1px solid rgba(56,232,255,.28)!important;background:rgba(56,232,255,.07)!important;box-shadow:0 0 25px rgba(56,232,255,.08);backdrop-filter:blur(10px)}
.main-card,.preview,.mini,.feature,.card,.section,.panel,.container>div,.pass,.table-wrap,.admin-card,.stat-card{
  background:linear-gradient(145deg,rgba(18,26,54,.88),rgba(8,13,30,.82))!important;
  border:1px solid var(--sq-border)!important;box-shadow:var(--sq-shadow)!important;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.main-card{border-radius:28px!important;position:relative;overflow:hidden}.main-card:before{content:"";position:absolute;left:-20%;top:-30%;width:55%;height:70%;background:radial-gradient(circle,rgba(56,232,255,.14),transparent 68%);pointer-events:none}
.mini,.feature,.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}.mini:hover,.feature:hover,.card:hover,.panel:hover{transform:translateY(-4px);border-color:rgba(56,232,255,.36)!important;box-shadow:0 20px 45px rgba(0,0,0,.30),0 0 28px rgba(56,232,255,.07)!important}
input,select,textarea{background:rgba(4,8,20,.72)!important;color:#f7f9ff!important;border:1px solid rgba(132,151,255,.24)!important;border-radius:13px!important;outline:none!important;transition:.2s!important}
input:focus,select:focus,textarea:focus{border-color:var(--sq-cyan)!important;box-shadow:0 0 0 4px rgba(56,232,255,.08),0 0 24px rgba(56,232,255,.10)!important}
input::placeholder{color:#697594!important}
button,.join,[type=submit]{border:0!important;border-radius:14px!important;background:linear-gradient(100deg,var(--sq-violet),#596eff 48%,var(--sq-cyan))!important;color:white!important;font-weight:800!important;box-shadow:0 12px 30px rgba(91,103,255,.24)!important;transition:transform .18s,box-shadow .18s,filter .18s!important;cursor:pointer}
button:hover,.join:hover,[type=submit]:hover{transform:translateY(-2px)!important;filter:saturate(1.15)!important;box-shadow:0 16px 38px rgba(56,232,255,.18)!important}
.feature-icon{background:linear-gradient(135deg,rgba(56,232,255,.18),rgba(139,108,255,.18))!important;border:1px solid rgba(56,232,255,.18)!important;box-shadow:0 0 24px rgba(56,232,255,.08)}
.status,.pill,.badge,.tag{border:1px solid rgba(132,151,255,.18)!important;background:rgba(255,255,255,.055)!important;border-radius:999px!important}
.low{border-left-color:var(--sq-green)!important}.medium{border-left-color:#ffd166!important}.high{border-left-color:#ff5b78!important}
.servicebar,.barbox>div[style*="background"],#crowdBar{background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink))!important}
.bars .bar{background:linear-gradient(180deg,var(--sq-cyan),var(--sq-violet))!important;box-shadow:0 0 15px rgba(56,232,255,.18)}
table{border-collapse:separate!important;border-spacing:0 8px!important;width:100%!important}th{color:#8e9abc!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.08em}td{background:rgba(12,18,38,.72)!important;border-top:1px solid rgba(132,151,255,.12)!important;border-bottom:1px solid rgba(132,151,255,.12)!important}td:first-child{border-left:1px solid rgba(132,151,255,.12)!important;border-radius:12px 0 0 12px}td:last-child{border-right:1px solid rgba(132,151,255,.12)!important;border-radius:0 12px 12px 0}
.container{position:relative}.note,.small,.hero-text{color:var(--sq-muted)!important}
.qr{box-shadow:0 0 0 8px rgba(255,255,255,.035),0 18px 40px rgba(0,0,0,.35)!important}
.pass{border-radius:30px!important;position:relative;overflow:hidden}.pass:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:6px;background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink));box-shadow:0 0 28px rgba(56,232,255,.4)}
.hero h1{letter-spacing:-.045em;text-shadow:0 8px 35px rgba(139,108,255,.15)}
.features h2,h1,h2,h3{letter-spacing:-.02em}
@media(max-width:800px){.navbar,.header{padding-left:5%!important;padding-right:5%!important}.hero{grid-template-columns:1fr!important}.cards{grid-template-columns:1fr!important}.grid{grid-template-columns:1fr!important}.feature-grid{grid-template-columns:1fr!important}.bars{grid-template-columns:repeat(12,1fr)!important}.main-card{margin-top:10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>

</head>

<body>


<div class="header">

    <a href="/">
        ← SmartQueue AI
    </a>

    <a href="/admin">
        Admin
    </a>
    <a href="/analytics">Analytics</a>

</div>


<div class="container">


<div class="top">

    <h1>
        Your Queue Dashboard
    </h1>

    <p>
        Hi {{ data.name }} · {{ data.service }}
    </p>

</div>


<div class="token-card">

    <div class="token-label">
        YOUR DIGITAL TOKEN
    </div>

    <div class="token" id="token">
        {{ data.token }}
    </div>

    <div
        class="status"
        id="status"
    >
        {{ data.status.upper() }}
    </div>


    <!-- MAIN PREDICTION -->

    <div class="predict">

        <div class="predict-title">
            🤖 AI ESTIMATED WAITING TIME
        </div>

        <div class="wait">

            <span id="wait">
                {{ data.wait }}
            </span>

            <span>
                minutes
            </span>

        </div>

        <div
            class="prediction-note"
            id="predictionNote"
        >

            Based on
            {{ data.people_ahead }}
            people ahead and
            {{ data.active_counters }}
            active counters

        </div>

    </div>


    <div class="grid">


        <div class="info">

            <div
                class="info-number"
                id="people"
            >
                {{ data.people_ahead }}
            </div>

            <div class="info-label">
                PEOPLE AHEAD
            </div>

        </div>


        <div class="info">

            <div
                class="info-number"
                id="position"
            >
                {{ data.position }}
            </div>

            <div class="info-label">
                QUEUE POSITION
            </div>

        </div>


        <div class="info">

            <div
                class="info-number"
                id="counter"
            >
                {{ data.counter or "-" }}
            </div>

            <div class="info-label">
                COUNTER
            </div>

        </div>


        <div class="info">

            <div
                class="info-number"
                id="service"
            >
                {{ data.service }}
            </div>

            <div class="info-label">
                SERVICE
            </div>

        </div>


    </div>




    <div style="margin-top:18px;padding:18px;border-radius:18px;background:#101426;border:1px solid #252a40;text-align:left;">
        <div style="display:flex;justify-content:space-between;color:#9ba1b3;font-size:13px;">
            <span>🟢 QUEUE LOAD</span><b id="loadLabel">{{ data.queue_load }} · {{ data.queue_load_pct }}%</b>
        </div>
        <div style="height:10px;background:#282d40;border-radius:10px;overflow:hidden;margin-top:8px;">
            <div id="loadFill" style="height:100%;width:{{ data.queue_load_pct }}%;background:linear-gradient(90deg,#55e6aa,#45d8ff);"></div>
        </div>
        <p id="turnTime" style="color:#aeb4c5;margin:14px 0 5px;">🕒 Expected turn: {{ data.turn_time }}</p>
        <p id="recommend" style="color:#aeb4c5;margin:5px 0;">🧭 Smart counter: {{ data.recommended_counter or "Waiting for availability" }}</p>
        <p id="forecast" style="color:#aeb4c5;margin:5px 0;">🔮 AI forecast next 30 min: {{ data.forecast_30m }} people</p>
        {% if data.priority %}<p style="color:#ffd66b;margin:8px 0 0;">⭐ Priority token</p>{% endif %}
    </div>

    <div style="margin-top:18px;padding:18px;border:1px dashed #4b5270;border-radius:16px;text-align:left;">
        <b>🎟️ Digital Queue Pass</b><br><br>
        Token: <b>{{ data.token }}</b><br>
        Service: {{ data.service }}<br>
        Expected Turn: <span id="passTime">{{ data.turn_time }}</span><br>
        Status: <span id="passStatus">{{ data.status.upper() }}</span>
        <div style="margin-top:12px;"><a href="/qr/pass/{{ data.id }}" target="_blank" style="display:inline-block;padding:10px 14px;border-radius:9px;background:#20243a;color:white;text-decoration:none;">▣ OPEN QR PASS</a></div>
    </div>

    <button
        class="notify"
        id="notifyButton"
        onclick="enableNotifications()"
    >
        🔔 ENABLE ALERTS
    </button>


    <div
        class="alert"
        id="alertBox"
    >
    </div>

    <button id="arrivalButton" onclick="confirmArrival()" style="display:none;width:100%;margin-top:12px;padding:14px;border:0;border-radius:12px;background:#55e6aa;color:#07100d;font-weight:900;">
        ✅ I'M AT THE COUNTER
    </button>


    <a
        href="/"
        class="home"
    >
        ← Back to Home
    </a>


    <div class="refresh">
        Live update every 3 seconds
    </div>


</div>

</div>


<script>

let lastPeople =
    {{ data.people_ahead }};

let lastStatus =
    "{{ data.status }}";

let sent = {};


function beep() {

    try {

        const AudioContext =
            window.AudioContext ||
            window.webkitAudioContext;

        if (!AudioContext) return;

        const audio =
            new AudioContext();

        const oscillator =
            audio.createOscillator();

        const gain =
            audio.createGain();

        oscillator.connect(gain);

        gain.connect(
            audio.destination
        );

        oscillator.frequency.value =
            850;

        gain.gain.setValueAtTime(
            0.15,
            audio.currentTime
        );

        gain.gain.exponentialRampToValueAtTime(
            0.001,
            audio.currentTime + 0.7
        );

        oscillator.start();

        oscillator.stop(
            audio.currentTime + 0.7
        );

    } catch(error) {

        console.log(error);

    }

}


function showAlert(message) {

    const box =
        document.getElementById(
            "alertBox"
        );

    box.innerText = message;

    box.style.display = "block";

    beep();

    if (
        "Notification" in window &&
        Notification.permission === "granted"
    ) {

        try {

            new Notification(
                "SmartQueue AI",
                {
                    body: message
                }
            );

        } catch(error) {

            console.log(error);

        }

    }

}


async function enableNotifications() {

    if (!("Notification" in window)) {

        alert(
            "Your browser does not support notifications."
        );

        return;

    }


    try {

        const permission =
            await Notification.requestPermission();


        const button =
            document.getElementById(
                "notifyButton"
            );


        if (permission === "granted") {

            button.innerText =
                "✓ ALERTS ENABLED";

            button.classList.add(
                "enabled"
            );

            showAlert(
                "Notifications enabled. We will alert you as your turn gets closer."
            );

        }
        else {

            button.innerText =
                "⚠ ALERTS BLOCKED";

            alert(
                "Notification permission was not granted."
            );

        }

    } catch(error) {

        console.log(error);

    }

}


function updatePage(data) {

    document.getElementById(
        "people"
    ).innerText =
        data.people_ahead;

    document.getElementById(
        "position"
    ).innerText =
        data.position || "-";

    document.getElementById(
        "wait"
    ).innerText =
        data.wait;

    document.getElementById("turnTime").innerText =
        "🕒 Expected turn: " + data.turn_time;
    document.getElementById("passTime").innerText = data.turn_time;
    document.getElementById("passStatus").innerText = data.status.toUpperCase();
    document.getElementById("loadLabel").innerText = data.queue_load + " · " + data.queue_load_pct + "%";
    document.getElementById("loadFill").style.width = data.queue_load_pct + "%";
    document.getElementById("recommend").innerText = "🧭 Smart counter: " + (data.recommended_counter || "Waiting for availability");
    document.getElementById("forecast").innerText = "🔮 AI forecast next 30 min: " + data.forecast_30m + " people";
    document.getElementById("arrivalButton").style.display = data.status === "serving" ? "block" : "none";

    document.getElementById(
        "counter"
    ).innerText =
        data.counter || "-";

    document.getElementById(
        "status"
    ).innerText =
        data.status.toUpperCase();


    document.getElementById(
        "predictionNote"
    ).innerText =
        "Based on " +
        data.people_ahead +
        " people ahead and " +
        data.active_counters +
        " active counters";


    if (
        data.status === "serving" &&
        lastStatus !== "serving"
    ) {

        showAlert(
            "🎉 YOUR TURN! Please proceed to " +
            (data.counter || "the counter") +
            "."
        );

    }


    if (data.status === "no_show" && lastStatus !== "no_show") {
        showAlert("⚠️ Your token was marked NO-SHOW. Please contact the admin.");
    }

    if (
        data.status === "waiting"
    ) {

        const p =
            data.people_ahead;


        if (
            p <= 5 &&
            p > 3 &&
            !sent["5"]
        ) {

            showAlert(
                "⏳ Your turn is getting closer. 5 or fewer people are ahead."
            );

            sent["5"] = true;

        }


        if (
            p <= 3 &&
            p > 2 &&
            !sent["3"]
        ) {

            showAlert(
                "🔔 Only 3 people are ahead of you."
            );

            sent["3"] = true;

        }


        if (
            p <= 2 &&
            p > 1 &&
            !sent["2"]
        ) {

            showAlert(
                "⚡ Only 2 people are ahead of you. Please be ready."
            );

            sent["2"] = true;

        }


        if (
            p <= 1 &&
            !sent["1"]
        ) {

            showAlert(
                "🚨 You are next! Please proceed near the counter."
            );

            sent["1"] = true;

        }

    }


    lastPeople =
        data.people_ahead;

    lastStatus =
        data.status;

}


function confirmArrival() {
    fetch("/api/queue/{{ data.id }}/arrived", {method:"POST"})
    .then(response => response.json())
    .then(data => showAlert(data.message));
}

function refreshQueue() {

    fetch(
        "/api/queue/{{ data.id }}"
    )

    .then(response =>
        response.json()
    )

    .then(data => {

        if (!data.error) {

            updatePage(data);

        }

    })

    .catch(error => {

        console.log(
            "Queue update error:",
            error
        );

    });

}


setInterval(
    refreshQueue,
    3000
);

</script>


</body>

</html>
"""


# ============================================================
# ADMIN PAGE
# ============================================================

ADMIN = r"""
<!DOCTYPE html>

<html>

<head>

<title>SmartQueue AI Admin</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

body {

    margin: 0;

    font-family: Arial;

    background: #080a16;

    color: white;

}

.header {

    padding: 20px 6%;

    border-bottom:
        1px solid #202337;

    display: flex;

    justify-content:
        space-between;

}

.header a {

    color: white;

    text-decoration: none;

}

.container {

    width: 92%;

    max-width: 1250px;

    margin: 30px auto;

}

.stats {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 15px;

}

.stat {

    background: #111426;

    border: 1px solid #23273c;

    padding: 25px;

    border-radius: 18px;

}

.stat strong {

    display: block;

    font-size: 35px;

    color: #8b7cff;

}

.control {

    margin-top: 20px;

    padding: 20px;

    background: #111426;

    border-radius: 18px;

}

select,
button {

    padding: 12px;

    border-radius: 9px;

    border: 1px solid #30344b;

}

select {

    background: #080a16;

    color: white;

}

button {

    background: #7564ff;

    color: white;

    cursor: pointer;

    margin-left: 7px;

}

.table-card {

    margin-top: 20px;

    padding: 20px;

    background: #111426;

    border-radius: 18px;

    overflow-x: auto;

}

table {

    width: 100%;

    border-collapse: collapse;

}

th,
td {

    padding: 13px;

    border-bottom:
        1px solid #24283c;

    text-align: left;

}

.badge {

    padding: 5px 9px;

    border-radius: 20px;

    background: #25293c;

}

@media(max-width:800px) {

    .stats {

        grid-template-columns:
            1fr 1fr;

    }

}

</style>

<style id="smartqueue-aurora-theme">
:root{
  --sq-bg:#050816;--sq-panel:rgba(12,18,38,.78);--sq-panel2:rgba(18,25,51,.82);
  --sq-border:rgba(132,151,255,.20);--sq-text:#f7f9ff;--sq-muted:#98a4c4;
  --sq-cyan:#38e8ff;--sq-violet:#8b6cff;--sq-pink:#ff4fd8;--sq-green:#45e6a7;
  --sq-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at 10% 10%,rgba(56,232,255,.10),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(139,108,255,.14),transparent 30%),
    radial-gradient(circle at 50% 100%,rgba(255,79,216,.08),transparent 30%),
    var(--sq-bg)!important;
  color:var(--sq-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;
  min-height:100vh;position:relative;overflow-x:hidden;
}
body:before{content:"";position:fixed;inset:-35%;pointer-events:none;z-index:-1;background:conic-gradient(from 180deg at 50% 50%,transparent,rgba(56,232,255,.045),transparent 28%,rgba(139,108,255,.055),transparent 60%,rgba(255,79,216,.04),transparent);animation:sqSpin 22s linear infinite}
body:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 85%)}
@keyframes sqSpin{to{transform:rotate(360deg)}}
@keyframes sqFloat{50%{transform:translateY(-5px)}}
@keyframes sqPulse{50%{box-shadow:0 0 0 7px rgba(56,232,255,.02),0 0 28px rgba(56,232,255,.22)}}
.navbar,.header{background:rgba(5,8,22,.72)!important;border-color:var(--sq-border)!important;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;z-index:5}
.logo{letter-spacing:.2px!important}.logo:before{content:"◉";display:inline-grid;place-items:center;width:30px;height:30px;margin-right:9px;border-radius:10px;background:linear-gradient(135deg,var(--sq-cyan),var(--sq-violet));box-shadow:0 0 25px rgba(56,232,255,.28);font-size:12px;color:white}
.logo span,.gradient,.big,.token{color:var(--sq-cyan)!important;background:linear-gradient(90deg,var(--sq-cyan),#a68bff 48%,var(--sq-pink));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.nav a,.header a{transition:.2s;color:#cdd5f1!important}.nav a:hover,.header a:hover{color:white!important;text-shadow:0 0 16px rgba(56,232,255,.55)}
.hero{position:relative}.hero:before{content:"";position:absolute;width:280px;height:280px;right:4%;top:8%;border-radius:50%;background:radial-gradient(circle,rgba(139,108,255,.20),transparent 68%);filter:blur(8px);pointer-events:none}
.badge{border:1px solid rgba(56,232,255,.28)!important;background:rgba(56,232,255,.07)!important;box-shadow:0 0 25px rgba(56,232,255,.08);backdrop-filter:blur(10px)}
.main-card,.preview,.mini,.feature,.card,.section,.panel,.container>div,.pass,.table-wrap,.admin-card,.stat-card{
  background:linear-gradient(145deg,rgba(18,26,54,.88),rgba(8,13,30,.82))!important;
  border:1px solid var(--sq-border)!important;box-shadow:var(--sq-shadow)!important;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.main-card{border-radius:28px!important;position:relative;overflow:hidden}.main-card:before{content:"";position:absolute;left:-20%;top:-30%;width:55%;height:70%;background:radial-gradient(circle,rgba(56,232,255,.14),transparent 68%);pointer-events:none}
.mini,.feature,.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}.mini:hover,.feature:hover,.card:hover,.panel:hover{transform:translateY(-4px);border-color:rgba(56,232,255,.36)!important;box-shadow:0 20px 45px rgba(0,0,0,.30),0 0 28px rgba(56,232,255,.07)!important}
input,select,textarea{background:rgba(4,8,20,.72)!important;color:#f7f9ff!important;border:1px solid rgba(132,151,255,.24)!important;border-radius:13px!important;outline:none!important;transition:.2s!important}
input:focus,select:focus,textarea:focus{border-color:var(--sq-cyan)!important;box-shadow:0 0 0 4px rgba(56,232,255,.08),0 0 24px rgba(56,232,255,.10)!important}
input::placeholder{color:#697594!important}
button,.join,[type=submit]{border:0!important;border-radius:14px!important;background:linear-gradient(100deg,var(--sq-violet),#596eff 48%,var(--sq-cyan))!important;color:white!important;font-weight:800!important;box-shadow:0 12px 30px rgba(91,103,255,.24)!important;transition:transform .18s,box-shadow .18s,filter .18s!important;cursor:pointer}
button:hover,.join:hover,[type=submit]:hover{transform:translateY(-2px)!important;filter:saturate(1.15)!important;box-shadow:0 16px 38px rgba(56,232,255,.18)!important}
.feature-icon{background:linear-gradient(135deg,rgba(56,232,255,.18),rgba(139,108,255,.18))!important;border:1px solid rgba(56,232,255,.18)!important;box-shadow:0 0 24px rgba(56,232,255,.08)}
.status,.pill,.badge,.tag{border:1px solid rgba(132,151,255,.18)!important;background:rgba(255,255,255,.055)!important;border-radius:999px!important}
.low{border-left-color:var(--sq-green)!important}.medium{border-left-color:#ffd166!important}.high{border-left-color:#ff5b78!important}
.servicebar,.barbox>div[style*="background"],#crowdBar{background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink))!important}
.bars .bar{background:linear-gradient(180deg,var(--sq-cyan),var(--sq-violet))!important;box-shadow:0 0 15px rgba(56,232,255,.18)}
table{border-collapse:separate!important;border-spacing:0 8px!important;width:100%!important}th{color:#8e9abc!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.08em}td{background:rgba(12,18,38,.72)!important;border-top:1px solid rgba(132,151,255,.12)!important;border-bottom:1px solid rgba(132,151,255,.12)!important}td:first-child{border-left:1px solid rgba(132,151,255,.12)!important;border-radius:12px 0 0 12px}td:last-child{border-right:1px solid rgba(132,151,255,.12)!important;border-radius:0 12px 12px 0}
.container{position:relative}.note,.small,.hero-text{color:var(--sq-muted)!important}
.qr{box-shadow:0 0 0 8px rgba(255,255,255,.035),0 18px 40px rgba(0,0,0,.35)!important}
.pass{border-radius:30px!important;position:relative;overflow:hidden}.pass:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:6px;background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink));box-shadow:0 0 28px rgba(56,232,255,.4)}
.hero h1{letter-spacing:-.045em;text-shadow:0 8px 35px rgba(139,108,255,.15)}
.features h2,h1,h2,h3{letter-spacing:-.02em}
@media(max-width:800px){.navbar,.header{padding-left:5%!important;padding-right:5%!important}.hero{grid-template-columns:1fr!important}.cards{grid-template-columns:1fr!important}.grid{grid-template-columns:1fr!important}.feature-grid{grid-template-columns:1fr!important}.bars{grid-template-columns:repeat(12,1fr)!important}.main-card{margin-top:10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>

</head>

<body>


<div class="header">

    <strong>
        SmartQueue AI · Admin
    </strong>

    <div>

        <a href="/">
            Home
        </a>

        &nbsp;&nbsp;

        <a href="/analytics">
            Analytics
        </a>
        &nbsp;&nbsp;
        <a href="/history">
            History
        </a>

    </div>

</div>


<div class="container">


<div class="stats">

    <div class="stat">

        <strong id="waiting">
            0
        </strong>

        Waiting

    </div>

    <div class="stat">

        <strong id="serving">
            0
        </strong>

        Serving

    </div>

    <div class="stat">

        <strong id="completed">
            0
        </strong>

        Completed Today

    </div>

    <div class="stat">

        <strong id="total">
            0
        </strong>

        Active Queue

    </div>

    <div class="stat">
        <strong id="density">-</strong>
        Crowd Density
    </div>

</div>


<div class="control">

    <b>
        Counter:
    </b>

    <select id="counter">

        <option value="0">🤖 Smart Auto — Best Free Counter</option>

        {% for c in counters %}

        <option value="{{ c.id }}">
            {{ c.name }}
        </option>

        {% endfor %}

    </select>


    <button onclick="callNext()">
        CALL NEXT
    </button>

    <span style="color:#8f94aa;margin-left:10px;">
        🤖 Smart Auto chooses a free counter automatically.
    </span>

</div>


<div class="table-card">

<h2>
    Live Queue
</h2>

<table>

<thead>

<tr>

<th>Token</th>
<th>Name</th>
<th>Service</th>
<th>Priority</th>
<th>Status</th>
<th>Counter</th>
<th>Action</th>

</tr>

</thead>

<tbody id="body">

</tbody>

</table>

</div>

</div>


<script>

function callNext() {

    const counter =
        document.getElementById(
            "counter"
        ).value;


    fetch(
        "/api/admin/call-next",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                counter_id: counter
            })
        }
    )

    .then(r => r.json())

    .then(data => {

        alert(data.message);

        load();

    });

}


function complete(id) {

    fetch(
        "/api/admin/complete/" + id,
        {
            method: "POST"
        }
    )

    .then(r => r.json())

    .then(data => {

        alert(data.message);

        load();

    });

}


function skip(id) {

    fetch(
        "/api/admin/skip/" + id,
        {
            method: "POST"
        }
    )

    .then(r => r.json())

    .then(data => {

        alert(data.message);

        load();

    });

}


function noShow(id) {
    fetch("/api/admin/no-show/" + id, {method:"POST"})
    .then(r => r.json()).then(data => { alert(data.message); load(); });
}

function togglePriority(id) {
    fetch("/api/admin/priority/" + id, {method:"POST"})
    .then(r => r.json()).then(() => load());
}

function load() {

    fetch(
        "/api/admin/queue"
    )

    .then(r => r.json())

    .then(data => {

        document.getElementById(
            "waiting"
        ).innerText =
            data.stats.waiting;

        document.getElementById(
            "serving"
        ).innerText =
            data.stats.serving;

        document.getElementById(
            "completed"
        ).innerText =
            data.stats.completed;

        fetch("/api/analytics")
            .then(r => r.json())
            .then(a => {
                const density = document.getElementById("density");
                if (density) density.innerText = a.crowd_density;
            });

        document.getElementById(
            "total"
        ).innerText =
            data.stats.waiting +
            data.stats.serving;


        let html = "";


        data.queue.forEach(
            item => {

                let action = "";


                if (
                    item.status ===
                    "serving"
                ) {

                    action =
                        `<button onclick="complete(${item.id})">COMPLETE</button>
                         <button onclick="noShow(${item.id})">NO-SHOW</button>`;

                }


                if (
                    item.status ===
                    "waiting"
                ) {

                    action =
                        `<button onclick="skip(${item.id})">
                        SKIP
                        </button>
                        <button onclick="togglePriority(${item.id})">${item.priority ? "REMOVE PRIORITY" : "MAKE PRIORITY"}</button>`;

                }


                html += `

                <tr>

                    <td>
                        <b>${item.token}</b>
                    </td>

                    <td>
                        ${item.name}
                    </td>

                    <td>
                        ${item.service}
                    </td>

                    <td>${item.priority ? "⭐ YES" : "-"}</td>

                    <td>
                        <span class="badge">
                            ${item.status}
                        </span>
                    </td>

                    <td>
                        ${item.counter || "-"}
                    </td>

                    <td>
                        ${action}
                    </td>

                </tr>

                `;

            }
        );


        document.getElementById(
            "body"
        ).innerHTML = html;

    });

}


load();

setInterval(
    load,
    3000
);

</script>

</body>

</html>
"""


# ============================================================
# HISTORY
# ============================================================

HISTORY = r"""
<!DOCTYPE html>

<html>

<head>

<title>SmartQueue History</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

body {

    margin: 0;

    font-family: Arial;

    background: #080a16;

    color: white;

}

.header {

    padding: 20px 7%;

    border-bottom:
        1px solid #22263a;

}

.header a {

    color: white;

    text-decoration: none;

}

.container {

    width: 92%;

    max-width: 1100px;

    margin: 35px auto;

}

.card {

    background: #111426;

    border: 1px solid #23273c;

    border-radius: 20px;

    padding: 25px;

    overflow-x: auto;

}

table {

    width: 100%;

    border-collapse: collapse;

}

th,
td {

    padding: 13px;

    border-bottom:
        1px solid #25293c;

    text-align: left;

}

.note {

    margin-top: 20px;

    color: #777d91;

}

</style>

<style id="smartqueue-aurora-theme">
:root{
  --sq-bg:#050816;--sq-panel:rgba(12,18,38,.78);--sq-panel2:rgba(18,25,51,.82);
  --sq-border:rgba(132,151,255,.20);--sq-text:#f7f9ff;--sq-muted:#98a4c4;
  --sq-cyan:#38e8ff;--sq-violet:#8b6cff;--sq-pink:#ff4fd8;--sq-green:#45e6a7;
  --sq-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at 10% 10%,rgba(56,232,255,.10),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(139,108,255,.14),transparent 30%),
    radial-gradient(circle at 50% 100%,rgba(255,79,216,.08),transparent 30%),
    var(--sq-bg)!important;
  color:var(--sq-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;
  min-height:100vh;position:relative;overflow-x:hidden;
}
body:before{content:"";position:fixed;inset:-35%;pointer-events:none;z-index:-1;background:conic-gradient(from 180deg at 50% 50%,transparent,rgba(56,232,255,.045),transparent 28%,rgba(139,108,255,.055),transparent 60%,rgba(255,79,216,.04),transparent);animation:sqSpin 22s linear infinite}
body:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 85%)}
@keyframes sqSpin{to{transform:rotate(360deg)}}
@keyframes sqFloat{50%{transform:translateY(-5px)}}
@keyframes sqPulse{50%{box-shadow:0 0 0 7px rgba(56,232,255,.02),0 0 28px rgba(56,232,255,.22)}}
.navbar,.header{background:rgba(5,8,22,.72)!important;border-color:var(--sq-border)!important;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;z-index:5}
.logo{letter-spacing:.2px!important}.logo:before{content:"◉";display:inline-grid;place-items:center;width:30px;height:30px;margin-right:9px;border-radius:10px;background:linear-gradient(135deg,var(--sq-cyan),var(--sq-violet));box-shadow:0 0 25px rgba(56,232,255,.28);font-size:12px;color:white}
.logo span,.gradient,.big,.token{color:var(--sq-cyan)!important;background:linear-gradient(90deg,var(--sq-cyan),#a68bff 48%,var(--sq-pink));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.nav a,.header a{transition:.2s;color:#cdd5f1!important}.nav a:hover,.header a:hover{color:white!important;text-shadow:0 0 16px rgba(56,232,255,.55)}
.hero{position:relative}.hero:before{content:"";position:absolute;width:280px;height:280px;right:4%;top:8%;border-radius:50%;background:radial-gradient(circle,rgba(139,108,255,.20),transparent 68%);filter:blur(8px);pointer-events:none}
.badge{border:1px solid rgba(56,232,255,.28)!important;background:rgba(56,232,255,.07)!important;box-shadow:0 0 25px rgba(56,232,255,.08);backdrop-filter:blur(10px)}
.main-card,.preview,.mini,.feature,.card,.section,.panel,.container>div,.pass,.table-wrap,.admin-card,.stat-card{
  background:linear-gradient(145deg,rgba(18,26,54,.88),rgba(8,13,30,.82))!important;
  border:1px solid var(--sq-border)!important;box-shadow:var(--sq-shadow)!important;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.main-card{border-radius:28px!important;position:relative;overflow:hidden}.main-card:before{content:"";position:absolute;left:-20%;top:-30%;width:55%;height:70%;background:radial-gradient(circle,rgba(56,232,255,.14),transparent 68%);pointer-events:none}
.mini,.feature,.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}.mini:hover,.feature:hover,.card:hover,.panel:hover{transform:translateY(-4px);border-color:rgba(56,232,255,.36)!important;box-shadow:0 20px 45px rgba(0,0,0,.30),0 0 28px rgba(56,232,255,.07)!important}
input,select,textarea{background:rgba(4,8,20,.72)!important;color:#f7f9ff!important;border:1px solid rgba(132,151,255,.24)!important;border-radius:13px!important;outline:none!important;transition:.2s!important}
input:focus,select:focus,textarea:focus{border-color:var(--sq-cyan)!important;box-shadow:0 0 0 4px rgba(56,232,255,.08),0 0 24px rgba(56,232,255,.10)!important}
input::placeholder{color:#697594!important}
button,.join,[type=submit]{border:0!important;border-radius:14px!important;background:linear-gradient(100deg,var(--sq-violet),#596eff 48%,var(--sq-cyan))!important;color:white!important;font-weight:800!important;box-shadow:0 12px 30px rgba(91,103,255,.24)!important;transition:transform .18s,box-shadow .18s,filter .18s!important;cursor:pointer}
button:hover,.join:hover,[type=submit]:hover{transform:translateY(-2px)!important;filter:saturate(1.15)!important;box-shadow:0 16px 38px rgba(56,232,255,.18)!important}
.feature-icon{background:linear-gradient(135deg,rgba(56,232,255,.18),rgba(139,108,255,.18))!important;border:1px solid rgba(56,232,255,.18)!important;box-shadow:0 0 24px rgba(56,232,255,.08)}
.status,.pill,.badge,.tag{border:1px solid rgba(132,151,255,.18)!important;background:rgba(255,255,255,.055)!important;border-radius:999px!important}
.low{border-left-color:var(--sq-green)!important}.medium{border-left-color:#ffd166!important}.high{border-left-color:#ff5b78!important}
.servicebar,.barbox>div[style*="background"],#crowdBar{background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink))!important}
.bars .bar{background:linear-gradient(180deg,var(--sq-cyan),var(--sq-violet))!important;box-shadow:0 0 15px rgba(56,232,255,.18)}
table{border-collapse:separate!important;border-spacing:0 8px!important;width:100%!important}th{color:#8e9abc!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.08em}td{background:rgba(12,18,38,.72)!important;border-top:1px solid rgba(132,151,255,.12)!important;border-bottom:1px solid rgba(132,151,255,.12)!important}td:first-child{border-left:1px solid rgba(132,151,255,.12)!important;border-radius:12px 0 0 12px}td:last-child{border-right:1px solid rgba(132,151,255,.12)!important;border-radius:0 12px 12px 0}
.container{position:relative}.note,.small,.hero-text{color:var(--sq-muted)!important}
.qr{box-shadow:0 0 0 8px rgba(255,255,255,.035),0 18px 40px rgba(0,0,0,.35)!important}
.pass{border-radius:30px!important;position:relative;overflow:hidden}.pass:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:6px;background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink));box-shadow:0 0 28px rgba(56,232,255,.4)}
.hero h1{letter-spacing:-.045em;text-shadow:0 8px 35px rgba(139,108,255,.15)}
.features h2,h1,h2,h3{letter-spacing:-.02em}
@media(max-width:800px){.navbar,.header{padding-left:5%!important;padding-right:5%!important}.hero{grid-template-columns:1fr!important}.cards{grid-template-columns:1fr!important}.grid{grid-template-columns:1fr!important}.feature-grid{grid-template-columns:1fr!important}.bars{grid-template-columns:repeat(12,1fr)!important}.main-card{margin-top:10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>

</head>

<body>


<div class="header">

    <a href="/">
        ← SmartQueue AI
    </a>

</div>


<div class="container">

<div class="card">

<h2>
Queue History
</h2>

<table>

<thead>

<tr>

<th>Token</th>
<th>Name</th>
<th>Service</th>
<th>Counter</th>
<th>Completed</th>

</tr>

</thead>

<tbody>

{% for h in history %}

<tr>

<td>
<b>{{ h.token }}</b>
</td>

<td>
{{ h.user_name }}
</td>

<td>
{{ h.service }}
</td>

<td>
{{ h.counter_id or "-" }}
</td>

<td>
{{ h.completed_at }}
</td>

</tr>

{% endfor %}

</tbody>

</table>


<div class="note">

History is automatically removed
after 7 days.

</div>

</div>

</div>

</body>

</html>
"""



# ============================================================
# QR DIGITAL PASS
# ============================================================
QR_PASS = r"""
<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Digital Queue Pass</title><style>body{margin:0;background:#080a16;color:white;font-family:Arial;display:flex;justify-content:center;padding:30px}.pass{width:420px;max-width:95%;background:#111426;border:1px solid #30344b;border-radius:24px;padding:28px;text-align:center}.token{font-size:60px;font-weight:900;color:#8b7cff}.qr{width:180px;height:180px;background:white;padding:8px;border-radius:12px}.small{color:#9298aa;line-height:1.7}</style><style id="smartqueue-aurora-theme">
:root{
  --sq-bg:#050816;--sq-panel:rgba(12,18,38,.78);--sq-panel2:rgba(18,25,51,.82);
  --sq-border:rgba(132,151,255,.20);--sq-text:#f7f9ff;--sq-muted:#98a4c4;
  --sq-cyan:#38e8ff;--sq-violet:#8b6cff;--sq-pink:#ff4fd8;--sq-green:#45e6a7;
  --sq-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at 10% 10%,rgba(56,232,255,.10),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(139,108,255,.14),transparent 30%),
    radial-gradient(circle at 50% 100%,rgba(255,79,216,.08),transparent 30%),
    var(--sq-bg)!important;
  color:var(--sq-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;
  min-height:100vh;position:relative;overflow-x:hidden;
}
body:before{content:"";position:fixed;inset:-35%;pointer-events:none;z-index:-1;background:conic-gradient(from 180deg at 50% 50%,transparent,rgba(56,232,255,.045),transparent 28%,rgba(139,108,255,.055),transparent 60%,rgba(255,79,216,.04),transparent);animation:sqSpin 22s linear infinite}
body:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 85%)}
@keyframes sqSpin{to{transform:rotate(360deg)}}
@keyframes sqFloat{50%{transform:translateY(-5px)}}
@keyframes sqPulse{50%{box-shadow:0 0 0 7px rgba(56,232,255,.02),0 0 28px rgba(56,232,255,.22)}}
.navbar,.header{background:rgba(5,8,22,.72)!important;border-color:var(--sq-border)!important;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;z-index:5}
.logo{letter-spacing:.2px!important}.logo:before{content:"◉";display:inline-grid;place-items:center;width:30px;height:30px;margin-right:9px;border-radius:10px;background:linear-gradient(135deg,var(--sq-cyan),var(--sq-violet));box-shadow:0 0 25px rgba(56,232,255,.28);font-size:12px;color:white}
.logo span,.gradient,.big,.token{color:var(--sq-cyan)!important;background:linear-gradient(90deg,var(--sq-cyan),#a68bff 48%,var(--sq-pink));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.nav a,.header a{transition:.2s;color:#cdd5f1!important}.nav a:hover,.header a:hover{color:white!important;text-shadow:0 0 16px rgba(56,232,255,.55)}
.hero{position:relative}.hero:before{content:"";position:absolute;width:280px;height:280px;right:4%;top:8%;border-radius:50%;background:radial-gradient(circle,rgba(139,108,255,.20),transparent 68%);filter:blur(8px);pointer-events:none}
.badge{border:1px solid rgba(56,232,255,.28)!important;background:rgba(56,232,255,.07)!important;box-shadow:0 0 25px rgba(56,232,255,.08);backdrop-filter:blur(10px)}
.main-card,.preview,.mini,.feature,.card,.section,.panel,.container>div,.pass,.table-wrap,.admin-card,.stat-card{
  background:linear-gradient(145deg,rgba(18,26,54,.88),rgba(8,13,30,.82))!important;
  border:1px solid var(--sq-border)!important;box-shadow:var(--sq-shadow)!important;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.main-card{border-radius:28px!important;position:relative;overflow:hidden}.main-card:before{content:"";position:absolute;left:-20%;top:-30%;width:55%;height:70%;background:radial-gradient(circle,rgba(56,232,255,.14),transparent 68%);pointer-events:none}
.mini,.feature,.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}.mini:hover,.feature:hover,.card:hover,.panel:hover{transform:translateY(-4px);border-color:rgba(56,232,255,.36)!important;box-shadow:0 20px 45px rgba(0,0,0,.30),0 0 28px rgba(56,232,255,.07)!important}
input,select,textarea{background:rgba(4,8,20,.72)!important;color:#f7f9ff!important;border:1px solid rgba(132,151,255,.24)!important;border-radius:13px!important;outline:none!important;transition:.2s!important}
input:focus,select:focus,textarea:focus{border-color:var(--sq-cyan)!important;box-shadow:0 0 0 4px rgba(56,232,255,.08),0 0 24px rgba(56,232,255,.10)!important}
input::placeholder{color:#697594!important}
button,.join,[type=submit]{border:0!important;border-radius:14px!important;background:linear-gradient(100deg,var(--sq-violet),#596eff 48%,var(--sq-cyan))!important;color:white!important;font-weight:800!important;box-shadow:0 12px 30px rgba(91,103,255,.24)!important;transition:transform .18s,box-shadow .18s,filter .18s!important;cursor:pointer}
button:hover,.join:hover,[type=submit]:hover{transform:translateY(-2px)!important;filter:saturate(1.15)!important;box-shadow:0 16px 38px rgba(56,232,255,.18)!important}
.feature-icon{background:linear-gradient(135deg,rgba(56,232,255,.18),rgba(139,108,255,.18))!important;border:1px solid rgba(56,232,255,.18)!important;box-shadow:0 0 24px rgba(56,232,255,.08)}
.status,.pill,.badge,.tag{border:1px solid rgba(132,151,255,.18)!important;background:rgba(255,255,255,.055)!important;border-radius:999px!important}
.low{border-left-color:var(--sq-green)!important}.medium{border-left-color:#ffd166!important}.high{border-left-color:#ff5b78!important}
.servicebar,.barbox>div[style*="background"],#crowdBar{background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink))!important}
.bars .bar{background:linear-gradient(180deg,var(--sq-cyan),var(--sq-violet))!important;box-shadow:0 0 15px rgba(56,232,255,.18)}
table{border-collapse:separate!important;border-spacing:0 8px!important;width:100%!important}th{color:#8e9abc!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.08em}td{background:rgba(12,18,38,.72)!important;border-top:1px solid rgba(132,151,255,.12)!important;border-bottom:1px solid rgba(132,151,255,.12)!important}td:first-child{border-left:1px solid rgba(132,151,255,.12)!important;border-radius:12px 0 0 12px}td:last-child{border-right:1px solid rgba(132,151,255,.12)!important;border-radius:0 12px 12px 0}
.container{position:relative}.note,.small,.hero-text{color:var(--sq-muted)!important}
.qr{box-shadow:0 0 0 8px rgba(255,255,255,.035),0 18px 40px rgba(0,0,0,.35)!important}
.pass{border-radius:30px!important;position:relative;overflow:hidden}.pass:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:6px;background:linear-gradient(90deg,var(--sq-cyan),var(--sq-violet),var(--sq-pink));box-shadow:0 0 28px rgba(56,232,255,.4)}
.hero h1{letter-spacing:-.045em;text-shadow:0 8px 35px rgba(139,108,255,.15)}
.features h2,h1,h2,h3{letter-spacing:-.02em}
@media(max-width:800px){.navbar,.header{padding-left:5%!important;padding-right:5%!important}.hero{grid-template-columns:1fr!important}.cards{grid-template-columns:1fr!important}.grid{grid-template-columns:1fr!important}.feature-grid{grid-template-columns:1fr!important}.bars{grid-template-columns:repeat(12,1fr)!important}.main-card{margin-top:10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>

</head><body><div class="pass"><h2>🎟️ SmartQueue AI</h2><div class="small">DIGITAL QUEUE PASS</div><div class="token">{{data.token}}</div><h3>{{data.queue_type}} · {{data.service}}</h3><p class="small">Name: {{data.name}}<br>Expected Turn: {{data.turn_time}}<br>Status: {{data.status.upper()}}</p><img class="qr" src="/qr/token/{{data.id}}"><p class="small">Show this digital pass when your token is called.</p><button onclick="window.print()">PRINT PASS</button></div></body></html>
"""

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    clean_history()
    auto_no_show_cleanup()

    return render_template_string(
        HOME,
        services=SERVICES,
        category_services=CATEGORY_SERVICES
    )


# ============================================================
# JOIN
# ============================================================

@app.route("/join", methods=["POST"])
def join():

    name = request.form.get(
        "name",
        ""
    ).strip()

    queue_type = request.form.get(
        "queue_type",
        ""
    ).strip()

    service = request.form.get(
        "service",
        ""
    ).strip()

    priority = 1 if request.form.get("priority") == "1" else 0


    if not name:

        return "Name is required.", 400


    if queue_type not in CATEGORY_SERVICES:

        return "Invalid queue type.", 400


    if service not in CATEGORY_SERVICES[queue_type]:

        return "Invalid service for the selected category.", 400


    token, number = generate_token(
        service
    )


    now = datetime.now().isoformat()


    connection = db()


    connection.execute(
        """
        INSERT INTO users
        (name, created_at)
        VALUES (?, ?)
        """,
        (name, now)
    )


    cursor = connection.execute(
        """
        INSERT INTO queue
        (
            token,
            token_number,
            user_name,
            service,
            queue_type,
            status,
            counter_id,
            joined_at,
            priority,
            arrived
        )
        VALUES (?, ?, ?, ?, ?, 'waiting', NULL, ?, ?, 0)
        """,
        (
            token,
            number,
            name,
            service,
            queue_type,
            now,
            priority
        )
    )


    queue_id = cursor.lastrowid


    connection.commit()

    connection.close()


    return redirect(
        url_for(
            "queue_page",
            queue_id=queue_id
        )
    )


# ============================================================
# QUEUE PAGE
# ============================================================

@app.route("/queue/<int:queue_id>")
def queue_page(queue_id):

    data = queue_info(queue_id)

    if data is None:

        return "Queue not found.", 404


    return render_template_string(
        QUEUE,
        data=data
    )


# ============================================================
# QUEUE API
# ============================================================

@app.route("/api/queue/<int:queue_id>")
def queue_api(queue_id):

    auto_no_show_cleanup()
    data = queue_info(queue_id)

    if data is None:

        return jsonify({
            "error": "Queue not found"
        }), 404


    return jsonify(data)


# ============================================================
# ADMIN
# ============================================================

@app.route("/admin")
def admin():

    connection = db()

    counters = connection.execute(
        """
        SELECT *
        FROM counters
        ORDER BY id
        """
    ).fetchall()

    connection.close()


    return render_template_string(
        ADMIN,
        counters=counters
    )


# ============================================================
# ADMIN API
# ============================================================

@app.route("/api/admin/queue")
def admin_queue():

    clean_history()
    auto_no_show_cleanup()


    connection = db()


    rows = connection.execute(
        """
        SELECT
            q.*,
            c.name AS counter_name
        FROM queue q
        LEFT JOIN counters c
            ON q.counter_id = c.id
        WHERE q.status IN
            ('waiting', 'serving')
        ORDER BY q.id
        """
    ).fetchall()


    waiting = connection.execute(
        """
        SELECT COUNT(*) AS total
        FROM queue
        WHERE status = 'waiting'
        """
    ).fetchone()["total"]


    serving = connection.execute(
        """
        SELECT COUNT(*) AS total
        FROM queue
        WHERE status = 'serving'
        """
    ).fetchone()["total"]


    today = datetime.now().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    ).isoformat()


    completed = connection.execute(
        """
        SELECT COUNT(*) AS total
        FROM history
        WHERE completed_at >= ?
        """,
        (today,)
    ).fetchone()["total"]


    connection.close()


    queue = []


    for row in rows:

        queue.append({

            "id": row["id"],

            "token": row["token"],

            "name": row["user_name"],

            "queue_type": row["queue_type"],

            "service": row["service"],

            "status": row["status"],

            "priority": bool(row["priority"] or 0),
            "joined_at": row["joined_at"],

            "counter":
                row["counter_name"]

        })


    return jsonify({

        "queue": queue,

        "stats": {

            "waiting": waiting,

            "serving": serving,

            "completed": completed

        }

    })


# ============================================================
# CALL NEXT
# ============================================================

@app.route(
    "/api/admin/call-next",
    methods=["POST"]
)
def call_next():
    """Call exactly one waiting person to a selected counter.

    HARD RULE:
    - One counter can have only one serving person.
    - The same waiting person cannot be assigned twice by concurrent requests.
    - A counter is released only when that serving token completes/skips/is no-show.
    """
    data = request.get_json(silent=True) or {}

    try:
        counter_id = int(data.get("counter_id"))
    except Exception:
        return jsonify({"message": "Invalid counter."}), 400

    connection = db()

    try:
        # Lock SQLite for this short assignment transaction. This prevents two
        # simultaneous Call Next requests from selecting the same waiting token.
        connection.execute("BEGIN IMMEDIATE")

        # SMART AUTO (0): automatically choose a FREE counter. This is the
        # automatic counter-balancing feature. A busy counter is never selected.
        if counter_id == 0:
            free_counters = connection.execute(
                """
                SELECT c.*
                FROM counters c
                WHERE c.active = 1
                  AND NOT EXISTS (
                      SELECT 1 FROM queue q
                      WHERE q.status = 'serving'
                        AND q.counter_id = c.id
                  )
                ORDER BY c.id ASC
                """
            ).fetchall()

            if not free_counters:
                connection.rollback()
                return jsonify({
                    "message": "All counters are currently busy. Please wait until a counter is released.",
                    "all_busy": True
                }), 409

            # Balance usage by choosing the free counter with the lowest
            # number of completed tokens today; ties go to the lowest ID.
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            scores = []
            for c in free_counters:
                used = connection.execute(
                    """
                    SELECT COUNT(*) AS n FROM history
                    WHERE counter_id = ? AND completed_at >= ?
                    """,
                    (c["id"], today_start)
                ).fetchone()["n"]
                scores.append((used, c["id"], c))
            scores.sort(key=lambda x: (x[0], x[1]))
            counter = scores[0][2]
            counter_id = counter["id"]
        else:
            counter = connection.execute(
                """
                SELECT * FROM counters
                WHERE id = ? AND active = 1
                """,
                (counter_id,)
            ).fetchone()

            if counter is None:
                connection.rollback()
                return jsonify({"message": "Counter not found or inactive."}), 404

            # HARD ONE-PER-COUNTER CHECK.
            busy = connection.execute(
                """
                SELECT * FROM queue
                WHERE status = 'serving'
                  AND counter_id = ?
                LIMIT 1
                """,
                (counter_id,)
            ).fetchone()

            if busy:
                connection.rollback()
                return jsonify({
                    "message": f"{counter['name']} is BUSY with {busy['token']}. Complete or skip that person first.",
                    "busy": True,
                    "token": busy["token"]
                }), 409

        # Every free counter can serve any service. The next person is selected
        # from the common queue using priority first, then arrival order.
        customer = connection.execute(
            """
            SELECT * FROM queue
            WHERE status = 'waiting'
            ORDER BY priority DESC, id ASC
            LIMIT 1
            """
        ).fetchone()

        if customer is None:
            connection.rollback()
            return jsonify({"message": "No students are waiting."})

        now = datetime.now().isoformat()

        connection.execute(
            """
            UPDATE queue
            SET status = 'serving',
                counter_id = ?,
                called_at = ?
            WHERE id = ?
              AND status = 'waiting'
            """,
            (counter_id, now, customer["id"])
        )

        # Keep the counter's current token in sync.
        connection.execute(
            """
            UPDATE counters
            SET current_token = ?
            WHERE id = ?
            """,
            (customer["token"], counter_id)
        )

        connection.commit()

        return jsonify({
            "message": f"{customer['token']} called to {counter['name']}.",
            "token": customer["token"],
            "counter": counter["name"]
        })

    except sqlite3.IntegrityError:
        # The unique partial index is the final safety layer.
        try:
            connection.rollback()
        except Exception:
            pass
        return jsonify({
            "message": "This counter is already serving another person. Please complete or skip that person first.",
            "busy": True
        }), 409
    except Exception as exc:
        try:
            connection.rollback()
        except Exception:
            pass
        return jsonify({"message": f"Unable to call next: {exc}"}), 500
    finally:
        connection.close()


# ============================================================
# COMPLETE
# ============================================================

@app.route(
    "/api/admin/complete/<int:queue_id>",
    methods=["POST"]
)
def complete(queue_id):

    connection = db()


    customer = connection.execute(
        """
        SELECT *
        FROM queue
        WHERE id = ?
        """,
        (queue_id,)
    ).fetchone()


    if customer is None:

        connection.close()

        return jsonify({
            "message": "Queue not found."
        })


    now = datetime.now().isoformat()


    connection.execute(
        """
        UPDATE queue

        SET
            status = 'completed',
            completed_at = ?

        WHERE id = ?
        """,
        (
            now,
            queue_id
        )
    )


    connection.execute(
        """
        INSERT INTO history
        (
            token,
            user_name,
            queue_type,
            service,
            counter_id,
            joined_at,
            completed_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            customer["token"],
            customer["user_name"],
            customer["queue_type"],
            customer["service"],
            customer["counter_id"],
            customer["joined_at"],
            now
        )
    )


    if customer["counter_id"]:

        connection.execute(
            """
            UPDATE counters

            SET current_token = NULL

            WHERE id = ?
            """,
            (
                customer["counter_id"],
            )
        )


    connection.commit()

    connection.close()


    clean_history()


    return jsonify({

        "message":
            f"{customer['token']} completed."

    })


# ============================================================
# SKIP
# ============================================================

@app.route(
    "/api/admin/skip/<int:queue_id>",
    methods=["POST"]
)
def skip(queue_id):

    connection = db()


    customer = connection.execute(
        """
        SELECT *
        FROM queue
        WHERE id = ?
        """,
        (queue_id,)
    ).fetchone()


    if customer is None:

        connection.close()

        return jsonify({
            "message": "Queue not found."
        })


    connection.execute(
        """
        UPDATE queue

        SET status = 'skipped'

        WHERE id = ?
        """,
        (queue_id,)
    )


    if customer["counter_id"]:

        connection.execute(
            """
            UPDATE counters

            SET current_token = NULL

            WHERE id = ?
            """,
            (
                customer["counter_id"],
            )
        )


    connection.commit()

    connection.close()


    return jsonify({

        "message":
            f"{customer['token']} skipped."

    })


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    clean_history()


    connection = db()


    rows = connection.execute(
        """
        SELECT *
        FROM history
        ORDER BY id DESC
        """
    ).fetchall()


    connection.close()


    return render_template_string(
        HISTORY,
        history=rows
    )



# ============================================================
# EXTRA FEATURE ROUTES
# ============================================================
@app.route("/qr")
def qr_home():
    img=qrcode.make(request.host_url)
    buf=io.BytesIO(); img.save(buf,format="PNG")
    return Response(buf.getvalue(),mimetype="image/png")

@app.route("/qr/token/<int:queue_id>")
def qr_token(queue_id):
    target=request.host_url.rstrip("/")+url_for("queue_page",queue_id=queue_id)
    img=qrcode.make(target)
    buf=io.BytesIO(); img.save(buf,format="PNG")
    return Response(buf.getvalue(),mimetype="image/png")

@app.route("/qr/pass/<int:queue_id>")
def qr_pass(queue_id):
    data=queue_info(queue_id)
    if data is None: return "Queue not found.",404
    return render_template_string(QR_PASS,data=data)

@app.route("/api/queue/<int:queue_id>/arrived",methods=["POST"])
def confirm_arrival(queue_id):
    connection=db()
    row=connection.execute("SELECT * FROM queue WHERE id=?",(queue_id,)).fetchone()
    if row is None:
        connection.close(); return jsonify({"message":"Queue not found."}),404
    connection.execute("UPDATE queue SET arrived=1 WHERE id=?",(queue_id,))
    connection.commit(); connection.close()
    return jsonify({"message":"Arrival confirmed. Please stay near the counter."})

@app.route("/api/admin/no-show/<int:queue_id>",methods=["POST"])
def admin_no_show(queue_id):
    connection=db()
    row=connection.execute("SELECT * FROM queue WHERE id=?",(queue_id,)).fetchone()
    if row is None:
        connection.close(); return jsonify({"message":"Queue not found."}),404
    if row["status"]!="serving":
        connection.close(); return jsonify({"message":"Only a serving token can be marked no-show."})
    connection.execute("UPDATE queue SET status='no_show' WHERE id=?",(queue_id,))
    if row["counter_id"]:
        connection.execute("UPDATE counters SET current_token=NULL WHERE id=?",(row["counter_id"],))
    connection.commit(); connection.close()
    return jsonify({"message":f"{row['token']} marked NO-SHOW. Counter released."})

@app.route("/api/admin/priority/<int:queue_id>",methods=["POST"])
def admin_priority(queue_id):
    connection=db()
    row=connection.execute("SELECT priority FROM queue WHERE id=?",(queue_id,)).fetchone()
    if row is None:
        connection.close(); return jsonify({"message":"Queue not found."}),404
    new_value=0 if row["priority"] else 1
    connection.execute("UPDATE queue SET priority=? WHERE id=?",(new_value,queue_id))
    connection.commit(); connection.close()
    return jsonify({"message":"Priority updated."})

@app.route("/analytics")
def analytics_page():
    return render_template_string(ANALYTICS)

@app.route("/api/analytics")
def analytics_api():
    clean_history()
    connection=db()
    today=datetime.now().replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
    today_completed=connection.execute("SELECT COUNT(*) AS n FROM history WHERE completed_at>=?",(today,)).fetchone()["n"]
    total=connection.execute("SELECT COUNT(*) AS n FROM history").fetchone()["n"]
    rows=connection.execute("SELECT joined_at,completed_at FROM history").fetchall()
    durations=[]
    for row in rows:
        try:
            durations.append(max(0,(datetime.fromisoformat(row["completed_at"])-datetime.fromisoformat(row["joined_at"])).total_seconds()/60))
        except Exception: pass
    avg_wait=int(round(np.mean(durations))) if durations else AVERAGE_SERVICE_TIME
    hourly=[]
    for h in range(24):
        n=connection.execute("SELECT COUNT(*) AS n FROM history WHERE CAST(strftime('%H',completed_at) AS INTEGER)=?",(h,)).fetchone()["n"]
        hourly.append({"hour":h,"count":n})
    peak=max(hourly,key=lambda x:x["count"])
    services=[]
    for category, category_items in CATEGORY_SERVICES.items():
        category_count=connection.execute("SELECT COUNT(*) AS n FROM history WHERE queue_type=?",(category,)).fetchone()["n"]
        services.append({"service":category,"count":category_count})
        for service in category_items:
            n=connection.execute("SELECT COUNT(*) AS n FROM history WHERE service=?",(service,)).fetchone()["n"]
            services.append({"service":f"{category} - {service}","count":n})
    current_waiting=connection.execute("SELECT COUNT(*) AS n FROM queue WHERE status='waiting'").fetchone()["n"]
    active=max(1,connection.execute("SELECT COUNT(*) AS n FROM counters WHERE active=1").fetchone()["n"])
    total_live=connection.execute("SELECT COUNT(*) AS n FROM queue WHERE status IN ('waiting','serving')").fetchone()["n"]
    ratio=total_live/active
    load='LOW' if ratio<=3 else 'MEDIUM' if ratio<=7 else 'HIGH'
    load_pct=min(100, int((ratio/10)*100))
    forecast=max(0,int(round(current_waiting*.75+(10 if 9<=datetime.now().hour<=11 else 5))))

    # 7) PEAK-TIME PREDICTION: use recent history by hour plus current live
    # queue pressure to estimate which upcoming hour is likely to be busiest.
    current_hour=datetime.now().hour
    upcoming=[]
    for offset in range(1,7):
        h=(current_hour+offset)%24
        historical=next((x["count"] for x in hourly if x["hour"]==h),0)
        score=historical + current_waiting*0.35
        upcoming.append({"hour":h,"historical_count":historical,"score":round(score,1)})
    predicted_peak=max(upcoming,key=lambda x:x["score"]) if upcoming else {"hour":current_hour,"historical_count":0,"score":0}

    # 8) CROWD DENSITY: live queue pressure per active counter.
    density_pct=min(100,int(ratio/10*100))
    density='LOW' if ratio<=3 else 'MEDIUM' if ratio<=7 else 'HIGH'

    # Counter balancing status: free/busy counters and the next recommended free counter.
    counter_rows=connection.execute("SELECT id,name FROM counters WHERE active=1 ORDER BY id").fetchall()
    counter_status=[]
    for c in counter_rows:
        busy_row=connection.execute("SELECT token FROM queue WHERE status='serving' AND counter_id=? LIMIT 1",(c["id"],)).fetchone()
        counter_status.append({"id":c["id"],"name":c["name"],"busy":bool(busy_row),"token":busy_row["token"] if busy_row else None})
    free=[c for c in counter_status if not c["busy"]]
    recommended_free=free[0]["name"] if free else "All counters busy"

    connection.close()
    return jsonify({"today_completed":today_completed,"total_history":total,"avg_wait":avg_wait,"peak_hour":f"{peak['hour']:02d}:00","hourly":hourly,"services":services,"current_waiting":current_waiting,"forecast_30m":forecast,"load":load,"load_pct":load_pct,"crowd_density":density,"crowd_density_pct":density_pct,"predicted_peak_hour":f"{predicted_peak['hour']:02d}:00","predicted_peak_score":predicted_peak["score"],"upcoming_peak":upcoming,"counter_status":counter_status,"recommended_free_counter":recommended_free})

# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "OK",

        "application":
            APP_NAME,

        "time":
            datetime.now().isoformat()

    })



def auto_no_show_cleanup():
    connection=db()
    cutoff=(datetime.now()-timedelta(seconds=NO_SHOW_SECONDS)).isoformat()
    rows=connection.execute("SELECT id,counter_id FROM queue WHERE status='serving' AND arrived=0 AND called_at IS NOT NULL AND called_at<?",(cutoff,)).fetchall()
    for row in rows:
        connection.execute("UPDATE queue SET status='no_show' WHERE id=?",(row["id"],))
        if row["counter_id"]:
            connection.execute("UPDATE counters SET current_token=NULL WHERE id=?",(row["counter_id"],))
    connection.commit(); connection.close()

# ============================================================
# START
# ============================================================

init_database()

clean_history()


if __name__ == "__main__":

    print(
        "SMARTQUEUE AI SERVER STARTING..."
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False
    )
