from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import sqlite3, os, io
from datetime import datetime, timedelta
import qrcode
from sklearn.ensemble import RandomForestRegressor

APP_NAME='SmartQueue AI'
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_FILE=os.path.join(BASE_DIR,'smartqueue.db')
PORT=int(os.environ.get('PORT',5000))
HISTORY_DAYS=7
NO_SHOW_SECONDS=120
AVERAGE_SERVICE_TIME=6
COUNTERS=3

CATEGORY_SERVICES={
 'Bank':['Cash Deposit','Cash Withdrawal','Account Opening','Passbook Update','KYC Update','Loan Enquiry'],
 'Hospital':['OP Registration','Doctor Consultation','Lab Test','Pharmacy','Billing','Insurance Desk'],
 'College':['Admission','Exam Cell','Fees Payment','Certificate Request','Bonafide Certificate','Scholarship','Placement Cell']
}
app=Flask(__name__); CORS(app)

def db():
    c=sqlite3.connect(DB_FILE); c.row_factory=sqlite3.Row; return c

def init_db():
    c=db(); x=c.cursor()
    x.execute('CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT,created_at TEXT)')
    x.execute('CREATE TABLE IF NOT EXISTS counters(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT,active INTEGER DEFAULT 1,current_token TEXT)')
    x.execute('''CREATE TABLE IF NOT EXISTS queue(
      id INTEGER PRIMARY KEY AUTOINCREMENT,token TEXT,token_number INTEGER,user_name TEXT,
      service TEXT,queue_type TEXT DEFAULT 'College',status TEXT,counter_id INTEGER,
      joined_at TEXT,called_at TEXT,completed_at TEXT,priority INTEGER DEFAULT 0,arrived INTEGER DEFAULT 0)''')
    x.execute('''CREATE TABLE IF NOT EXISTS history(
      id INTEGER PRIMARY KEY AUTOINCREMENT,token TEXT,user_name TEXT,queue_type TEXT,
      service TEXT,counter_id INTEGER,joined_at TEXT,completed_at TEXT)''')
    cols=[r['name'] for r in x.execute('PRAGMA table_info(queue)').fetchall()]
    if 'priority' not in cols:x.execute('ALTER TABLE queue ADD COLUMN priority INTEGER DEFAULT 0')
    if 'arrived' not in cols:x.execute('ALTER TABLE queue ADD COLUMN arrived INTEGER DEFAULT 0')
    if 'queue_type' not in cols:x.execute("ALTER TABLE queue ADD COLUMN queue_type TEXT DEFAULT 'College'")
    x.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_serving_counter ON queue(counter_id) WHERE status='serving' AND counter_id IS NOT NULL")
    if x.execute('SELECT COUNT(*) n FROM counters').fetchone()['n']==0:
        for i in range(1,COUNTERS+1):x.execute('INSERT INTO counters(name,active) VALUES(?,1)',(f'Counter {i}',))
    c.commit();c.close()

def clean_history():
    c=db();cut=(datetime.now()-timedelta(days=HISTORY_DAYS)).isoformat();c.execute('DELETE FROM history WHERE completed_at<?',(cut,));c.commit();c.close()

def model():
    X=[];y=[]
    for p in range(51):
      for n in range(1,6):
       for s in [4,5,6,7,8,10]:X.append([p,n,s]);y.append(p*s/n)
    m=RandomForestRegressor(n_estimators=120,random_state=42);m.fit(X,y);return m
MODEL=model()
def wait_time(a,c):return max(0,int(round(MODEL.predict([[a,max(1,c),AVERAGE_SERVICE_TIME]])[0])))

def token_for(service):
    c=db();n=c.execute('SELECT MAX(token_number) n FROM queue WHERE service=?',(service,)).fetchone()['n'] or 0;c.close();n+=1;return f'{service[:2].upper()}-{n:03d}',n

def auto_no_show():
    c=db();cut=datetime.now()-timedelta(seconds=NO_SHOW_SECONDS)
    rows=c.execute("SELECT * FROM queue WHERE status='serving' AND called_at IS NOT NULL").fetchall()
    for r in rows:
      try:old=datetime.fromisoformat(r['called_at'])
      except:continue
      if old<cut:
        c.execute("UPDATE queue SET status='skipped',completed_at=? WHERE id=?",(datetime.now().isoformat(),r['id']))
        if r['counter_id']:c.execute('UPDATE counters SET current_token=NULL WHERE id=?',(r['counter_id'],))
    c.commit();c.close()

def info(qid):
    auto_no_show();c=db();r=c.execute('SELECT * FROM queue WHERE id=?',(qid,)).fetchone()
    if not r:c.close();return None
    p=r['priority'] or 0
    ahead=c.execute("SELECT COUNT(*) n FROM queue WHERE status='waiting' AND (priority>? OR (priority=? AND id<?))",(p,p,qid)).fetchone()['n']
    active=max(1,c.execute('SELECT COUNT(*) n FROM counters WHERE active=1').fetchone()['n'])
    waiting=c.execute("SELECT COUNT(*) n FROM queue WHERE status='waiting'").fetchone()['n'];serving=c.execute("SELECT COUNT(*) n FROM queue WHERE status='serving'").fetchone()['n']
    pos=ahead+1 if r['status']=='waiting' else 0;wt=wait_time(ahead if r['status']=='waiting' else 0,active)
    counter=None
    if r['counter_id']:
      z=c.execute('SELECT name FROM counters WHERE id=?',(r['counter_id'],)).fetchone();counter=z['name'] if z else None
    ratio=(waiting+serving)/active
    load='LOW' if ratio<=3 else ('MEDIUM' if ratio<=7 else 'HIGH');pct=min(100,int(ratio/3*100)) if load=='LOW' else (min(100,int(ratio/7*100)) if load=='MEDIUM' else 100)
    best=None
    for z in c.execute('SELECT * FROM counters WHERE active=1 ORDER BY id').fetchall():
      busy=c.execute("SELECT COUNT(*) n FROM queue WHERE status='serving' AND counter_id=?",(z['id'],)).fetchone()['n']
      if best is None or busy<best[0]:best=(busy,z['name'])
    forecast=max(0,int(round(waiting*.75+(10 if 9<=datetime.now().hour<=11 else 5))))
    rem=None
    if r['status']=='serving' and r['called_at']:rem=max(0,NO_SHOW_SECONDS-int((datetime.now()-datetime.fromisoformat(r['called_at'])).total_seconds()))
    c.close();return {'id':qid,'token':r['token'],'name':r['user_name'],'queue_type':r['queue_type'],'service':r['service'],'status':r['status'],'counter':counter,'people_ahead':ahead if r['status']=='waiting' else 0,'position':pos,'wait':wt,'turn_time':(datetime.now()+timedelta(minutes=wt)).strftime('%I:%M %p'),'active_counters':active,'queue_load':load,'queue_load_pct':pct,'recommended_counter':best[1] if best else None,'forecast_30m':forecast,'priority':bool(p),'arrived':bool(r['arrived'] or 0),'no_show_remaining':rem}

HOME='''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmartQueue AI</title><style>
*{box-sizing:border-box}body{margin:0;background:#050816;color:#f7f9ff;font-family:Inter,Arial;background-image:radial-gradient(circle at 10% 10%,#38e8ff18,transparent 28%),radial-gradient(circle at 90% 15%,#8b6cff22,transparent 30%)}nav{padding:22px 6%;border-bottom:1px solid #293252;background:#050816dd;display:flex;justify-content:space-between}nav a{color:#38e8ff;text-decoration:none;margin-left:18px}.wrap{width:88%;max-width:1200px;margin:auto}.hero{display:grid;grid-template-columns:1.1fr .9fr;gap:35px;align-items:center;padding:75px 0 40px}.badge{display:inline-block;padding:8px 13px;border:1px solid #38e8ff55;border-radius:99px;color:#38e8ff;font-size:12px;font-weight:bold}h1{font-size:clamp(44px,6vw,78px);line-height:.98;background:linear-gradient(90deg,#fff,#38e8ff,#a68bff,#ff4fd8);-webkit-background-clip:text;color:transparent}.hero p{color:#9aa6c6;font-size:18px;line-height:1.7}.card{background:#111a38dd;border:1px solid #2b365c;border-radius:25px;padding:27px;box-shadow:0 25px 70px #0008}.label{display:block;color:#aeb8d5;font-size:13px;margin:13px 0 7px}input,select{width:100%;padding:14px;border-radius:12px;border:1px solid #2b365c;background:#050817;color:white}.types{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.type{padding:13px;text-align:center;border:1px solid #2b365c;border-radius:13px;cursor:pointer}.active{border-color:#38e8ff!important;background:#38e8ff12}.services{display:flex;flex-wrap:wrap;gap:7px}.service{padding:8px 11px;border:1px solid #2b365c;border-radius:99px;font-size:12px;cursor:pointer}.btn{width:100%;padding:15px;border:0;border-radius:13px;background:linear-gradient(100deg,#8b6cff,#596eff,#38e8ff);color:white;font-weight:900;cursor:pointer;margin-top:15px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:25px}.stat{padding:16px;border:1px solid #2b365c;border-radius:16px}.stat b{display:block;font-size:24px}.stat span{color:#9aa6c6;font-size:12px}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:70px}.step{padding:20px;border:1px solid #2b365c;border-radius:18px;background:#ffffff05}.step b{color:#38e8ff}.step p{color:#9aa6c6;font-size:13px}@media(max-width:800px){.hero,.steps{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.wrap{width:92%}}
</style><nav><b>◉ SMARTQUEUE AI</b><div><a href="/admin">Admin</a><a href="/analytics">Analytics</a></div></nav><main class="wrap"><section class="hero"><div><span class="badge">● LIVE AI QUEUE INTELLIGENCE</span><h1>Know your place.<br>Own your time.</h1><p>Join a digital queue, get an AI-estimated waiting time, track your position and receive alerts when your turn is approaching.</p><div class="stats"><div class="stat"><b>AI</b><span>Wait prediction</span></div><div class="stat"><b>LIVE</b><span>Queue updates</span></div><div class="stat"><b>QR</b><span>Digital pass</span></div></div></div><div class="card"><h2>Start your queue</h2><label class="label">Your Name</label><input id="name" placeholder="Enter your name"><label class="label">Choose Place</label><div class="types"><div class="type active" data-t="Bank">🏦<br>Bank</div><div class="type" data-t="Hospital">🏥<br>Hospital</div><div class="type" data-t="College">🎓<br>College</div></div><label class="label">Service</label><div id="services" class="services"></div><label><input id="priority" type="checkbox" style="width:auto"> Request priority queue</label><button class="btn" onclick="joinQ()">GET DIGITAL TOKEN →</button><div id="msg" style="color:#ff8b9f;margin-top:10px"></div></div></section><section class="steps"><div class="step"><b>01</b><h3>Choose place</h3><p>Bank, Hospital or College.</p></div><div class="step"><b>02</b><h3>Pick service</h3><p>Select your required service.</p></div><div class="step"><b>03</b><h3>Get token</h3><p>Track queue and receive alerts.</p></div></section></main><script>
const S={{services|tojson}};let type='Bank',service=S.Bank[0];function render(){let b=document.getElementById('services');b.innerHTML='';S[type].forEach((s,i)=>{let d=document.createElement('div');d.className='service '+(i==0?'active':'');d.textContent=s;d.onclick=()=>{document.querySelectorAll('.service').forEach(x=>x.classList.remove('active'));d.classList.add('active');service=s};b.appendChild(d)})}document.querySelectorAll('.type').forEach(x=>x.onclick=()=>{document.querySelectorAll('.type').forEach(y=>y.classList.remove('active'));x.classList.add('active');type=x.dataset.t;service=S[type][0];render()});async function joinQ(){let name=document.getElementById('name').value.trim(),m=document.getElementById('msg');if(!name){m.textContent='Please enter your name.';return}let r=await fetch('/api/join',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,queue_type:type,service,priority:document.getElementById('priority').checked})});let x=await r.json();if(x.success)location.href='/token/'+x.id;else m.textContent=x.error||'Unable to join.'}render();</script>'''

TOKEN='''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{i.token}}</title><style>body{margin:0;background:#050816;color:white;font-family:Arial}.wrap{max-width:900px;width:92%;margin:30px auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.card{background:#111a38;border:1px solid #2b365c;border-radius:22px;padding:25px}.token{font-size:60px;font-weight:900;background:linear-gradient(90deg,#38e8ff,#a68bff,#ff4fd8);-webkit-background-clip:text;color:transparent}.muted{color:#9aa6c6}.value{font-size:28px;font-weight:900}.bar{height:10px;background:#202844;border-radius:20px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#38e8ff,#8b6cff,#ff4fd8)}button{padding:12px 15px;border:0;border-radius:11px;background:#6f63ff;color:white;font-weight:800;margin:4px;cursor:pointer}.qr{background:white;padding:8px;max-width:180px}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style><div class="wrap"><a href="/" style="color:#38e8ff">← New Queue</a><div class="grid"><div class="card"><span class="muted">DIGITAL TOKEN</span><div class="token">{{i.token}}</div><p>{{i.name}} · {{i.queue_type}} · {{i.service}}</p><h3>People ahead</h3><div id="ahead" class="value">{{i.people_ahead}}</div><h3>Estimated wait</h3><div class="value"><span id="wait">{{i.wait}}</span> min</div><p class="muted">Turn: <span id="turn">{{i.turn_time}}</span></p></div><div class="card"><h2>LIVE QUEUE</h2><p>Status: <b id="status">{{i.status|upper}}</b></p><p>Load: <b id="load">{{i.queue_load}}</b></p><div class="bar"><div id="bar" class="fill" style="width:{{i.queue_load_pct}}%"></div></div><p>Smart counter: <b id="counter">{{i.recommended_counter}}</b></p><p>30-min forecast: <b id="forecast">{{i.forecast_30m}}</b></p><button onclick="alerts()">🔔 ENABLE ALERTS</button><button onclick="arrived()">📍 I'M AT THE COUNTER</button><p id="msg" class="muted">Live tracking enabled.</p><img class="qr" src="/qr/token/{{i.id}}"></div></div></div><script>const id={{i.id}};let enabled=false;function alerts(){if('Notification'in window)Notification.requestPermission().then(()=>enabled=true);else enabled=true;document.getElementById('msg').textContent='Alerts enabled.'}function note(t,b){document.getElementById('msg').textContent=b;if(enabled&&Notification.permission==='granted')new Notification(t,{body:b})}async function arrived(){await fetch('/api/queue/'+id+'/arrived',{method:'POST'});document.getElementById('msg').textContent='Arrival recorded.'}async function up(){let r=await fetch('/api/queue/'+id),x=await r.json();if(!x.success)return;document.getElementById('ahead').textContent=x.people_ahead;document.getElementById('wait').textContent=x.wait;document.getElementById('turn').textContent=x.turn_time;document.getElementById('status').textContent=x.status.toUpperCase();document.getElementById('load').textContent=x.queue_load;document.getElementById('bar').style.width=x.queue_load_pct+'%';document.getElementById('counter').textContent=x.recommended_counter||'-';document.getElementById('forecast').textContent=x.forecast_30m;if(x.status==='serving')note('Your turn is next','Proceed to '+(x.counter||'the counter'));else if([5,3,2,1].includes(x.people_ahead))note('Queue update',x.people_ahead+' people ahead of you.')}setInterval(up,3000);</script>'''

ADMIN='''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Admin</title><style>body{background:#050816;color:white;font-family:Arial;margin:0}.wrap{width:94%;max-width:1200px;margin:25px auto}.card{background:#111a38;border:1px solid #2b365c;border-radius:18px;padding:20px;margin:12px 0}.btn{padding:9px 12px;border:0;border-radius:9px;background:#6f63ff;color:white;font-weight:700;cursor:pointer;margin:3px}.token{color:#38e8ff;font-weight:900;font-size:22px}</style><div class="wrap"><h1>SmartQueue AI — Admin Dashboard</h1><div id="out"></div><a href="/" style="color:#38e8ff">Home</a> · <a href="/analytics" style="color:#38e8ff">Analytics</a></div><script>async function load(){let x=await(await fetch('/api/admin')).json(),h='<div class="card"><h2>Counters</h2>';x.counters.forEach(c=>h+=`<p>${c.name}: <b>${c.current_token||'Free'}</b></p>`);h+='</div><div class="card"><h2>Live Queue</h2>';x.queue.forEach(q=>h+=`<p><span class="token">${q.token}</span> — ${q.user_name} — ${q.queue_type}/${q.service} — ${q.status}<br><button class="btn" onclick="callNext()">CALL NEXT</button><button class="btn" onclick="act(${q.id},'complete')">COMPLETE</button><button class="btn" onclick="act(${q.id},'skip')">SKIP</button><button class="btn" onclick="prio(${q.id})">PRIORITY</button></p>`);h+='</div>';document.getElementById('out').innerHTML=h}async function callNext(){await fetch('/api/admin/call-next',{method:'POST'});load()}async function act(id,a){await fetch('/api/admin/'+a+'/'+id,{method:'POST'});load()}async function prio(id){await fetch('/api/admin/priority/'+id,{method:'POST'});load()}load();setInterval(load,4000)</script>'''

ANALYTICS='''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analytics</title><style>body{background:#050816;color:white;font-family:Arial;margin:0}.wrap{width:92%;max-width:1100px;margin:30px auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#111a38;border:1px solid #2b365c;border-radius:18px;padding:20px}.big{font-size:30px;color:#38e8ff}.bars{height:200px;display:flex;align-items:end;gap:4px}.b{flex:1;background:linear-gradient(#38e8ff,#8b6cff);min-height:3px}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}}</style><div class="wrap"><h1>SmartQueue Analytics</h1><div class="cards"><div class="card"><div class="big" id="live">0</div>Live</div><div class="card"><div class="big" id="done">0</div>Completed</div><div class="card"><div class="big">6 min</div>Avg service time</div><div class="card"><div class="big" id="hist">0</div>History</div></div><div class="card" style="margin-top:15px"><h2>Hourly Activity</h2><div class="bars" id="bars"></div></div><div class="card" style="margin-top:15px"><h2>Services</h2><div id="services"></div></div><p><a href="/" style="color:#38e8ff">Home</a></p></div><script>async function load(){let x=await(await fetch('/api/analytics')).json();live.textContent=x.live;done.textContent=x.completed;hist.textContent=x.history;let m=Math.max(1,...x.hourly);bars.innerHTML=x.hourly.map(v=>`<div class="b" title="${v}" style="height:${v/m*100}%"></div>`).join('');services.innerHTML=Object.entries(x.services).map(([k,v])=>`<p>${k}: <b>${v}</b></p>`).join('')}load();setInterval(load,5000)</script>'''

@app.route('/')
def home():return render_template_string(HOME,services=CATEGORY_SERVICES)
@app.route('/token/<int:qid>')
def token(qid):
    i=info(qid)
    return render_template_string(TOKEN,i=i) if i else ('Token not found',404)
@app.route('/admin')
def admin():return render_template_string(ADMIN)
@app.route('/analytics')
def analytics():return render_template_string(ANALYTICS)
@app.post('/api/join')
def join():
    d=request.get_json(silent=True) or {};name=str(d.get('name','')).strip();qt=str(d.get('queue_type',''));service=str(d.get('service','')).strip()
    if not name:return jsonify(success=False,error='Name is required'),400
    if qt not in CATEGORY_SERVICES or service not in CATEGORY_SERVICES[qt]:return jsonify(success=False,error='Invalid category/service'),400
    tok,num=token_for(service);now=datetime.now().isoformat();c=db();cur=c.cursor();cur.execute('INSERT INTO users(name,created_at) VALUES(?,?)',(name,now));cur.execute('INSERT INTO queue(token,token_number,user_name,service,queue_type,status,joined_at,priority) VALUES(?,?,?,?,?,?,?,?)',(tok,num,name,service,qt,'waiting',now,1 if d.get('priority') else 0));qid=cur.lastrowid;c.commit();c.close();return jsonify(success=True,id=qid,token=tok)
@app.get('/api/queue/<int:qid>')
def qapi(qid):
    i=info(qid);return jsonify(success=True,**i) if i else (jsonify(success=False),404)
@app.post('/api/queue/<int:qid>/arrived')
def arrived(qid):
    c=db();c.execute('UPDATE queue SET arrived=1 WHERE id=?',(qid,));c.commit();c.close();return jsonify(success=True)
@app.get('/api/admin')
def admin_api():
    auto_no_show();c=db();cs=[dict(r) for r in c.execute('SELECT * FROM counters ORDER BY id').fetchall()];qs=[dict(r) for r in c.execute("SELECT * FROM queue WHERE status IN ('waiting','serving') ORDER BY priority DESC,id").fetchall()];c.close();return jsonify(counters=cs,queue=qs)
def finish(qid,status):
    c=db();r=c.execute('SELECT * FROM queue WHERE id=?',(qid,)).fetchone()
    if not r:c.close();return False
    now=datetime.now().isoformat()
    if status=='completed':c.execute('INSERT INTO history(token,user_name,queue_type,service,counter_id,joined_at,completed_at) VALUES(?,?,?,?,?,?,?)',(r['token'],r['user_name'],r['queue_type'],r['service'],r['counter_id'],r['joined_at'],now))
    c.execute('UPDATE queue SET status=?,completed_at=? WHERE id=?',(status,now,qid))
    if r['counter_id']:c.execute('UPDATE counters SET current_token=NULL WHERE id=?',(r['counter_id'],))
    c.commit();c.close();return True
@app.post('/api/admin/<action>/<int:qid>')
def admin_action(action,qid):
    if action in ('complete','skip'):return jsonify(success=finish(qid,'completed' if action=='complete' else 'skipped'))
    if action=='priority':
      c=db();r=c.execute('SELECT priority FROM queue WHERE id=?',(qid,)).fetchone();
      if not r:c.close();return jsonify(success=False)
      v=0 if r['priority'] else 1;c.execute('UPDATE queue SET priority=? WHERE id=?',(v,qid));c.commit();c.close();return jsonify(success=True,priority=bool(v))
    return jsonify(success=False),404
@app.post('/api/admin/call-next')
def call_next():
    auto_no_show();c=db();free=c.execute("SELECT * FROM counters WHERE active=1 AND id NOT IN (SELECT counter_id FROM queue WHERE status='serving' AND counter_id IS NOT NULL) ORDER BY id LIMIT 1").fetchone()
    if not free:c.close();return jsonify(success=False,error='All counters are busy')
    r=c.execute("SELECT * FROM queue WHERE status='waiting' ORDER BY priority DESC,id LIMIT 1").fetchone()
    if not r:c.close();return jsonify(success=False,error='Queue is empty')
    now=datetime.now().isoformat();c.execute('UPDATE queue SET status=\'serving\',counter_id=?,called_at=? WHERE id=?',(free['id'],now,r['id']));c.execute('UPDATE counters SET current_token=? WHERE id=?',(r['token'],free['id']));c.commit();c.close();return jsonify(success=True,token=r['token'])
@app.get('/api/history')
def history():
    clean_history();c=db();r=[dict(x) for x in c.execute('SELECT * FROM history ORDER BY id DESC LIMIT 200').fetchall()];c.close();return jsonify(rows=r)
@app.get('/api/analytics')
def analytics_api():
    clean_history();c=db();live=c.execute("SELECT COUNT(*) n FROM queue WHERE status IN ('waiting','serving')").fetchone()['n'];done=c.execute('SELECT COUNT(*) n FROM history').fetchone()['n'];services={};
    for r in c.execute('SELECT service,COUNT(*) n FROM history GROUP BY service'):services[r['service']]=r['n']
    for r in c.execute("SELECT service,COUNT(*) n FROM queue WHERE status IN ('waiting','serving') GROUP BY service"):services[r['service']]=services.get(r['service'],0)+r['n']
    hourly=[0]*24
    for r in c.execute('SELECT joined_at FROM queue'):
      try:hourly[datetime.fromisoformat(r['joined_at']).hour]+=1
      except:pass
    c.close();return jsonify(live=live,completed=done,history=done,average_service_time=AVERAGE_SERVICE_TIME,hourly=hourly,services=services)
@app.get('/qr/token/<int:qid>')
def qr(qid):
    img=qrcode.make(request.host_url.rstrip('/')+f'/token/{qid}');out=io.BytesIO();img.save(out,format='PNG');out.seek(0);return send_file(out,mimetype='image/png')
@app.get('/health')
def health():return jsonify(status='healthy',app=APP_NAME)

if __name__=='__main__':
    init_db();clean_history();print(f'{APP_NAME} running on http://127.0.0.1:{PORT}');app.run(host='0.0.0.0',port=PORT,debug=False,threaded=True)
