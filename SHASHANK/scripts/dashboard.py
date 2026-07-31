import os, csv, base64
from pathlib import Path
from flask import Flask, render_template_string

HERE = Path(__file__).parent
CAPTURES = HERE / 'captures'
CSV_PATH = HERE / 'results.csv'

app = Flask(__name__)

HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plant Health</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f0f2f5;color:#222}
h1{font-size:18px}
.bar{background:#1a3a1a;color:#fff;padding:12px 20px;display:flex;align-items:center;gap:10px}
.bar span{font-size:20px}
.content{max-width:1100px;margin:0 auto;padding:16px 20px}
.empty{text-align:center;padding:60px 20px;color:#888}
.empty span{font-size:50px;display:block;margin-bottom:8px}
.imgbox{background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.06);margin-bottom:14px}
.imgwrap{text-align:center;padding:10px;background:#fafbfc}
.imgwrap img{max-width:100%;max-height:75vh;border-radius:6px}
.imgfoot{padding:10px 16px;display:flex;justify-content:space-between;align-items:center;border-top:1px solid #eee}
.crop{font-size:16px;font-weight:600;color:#1a3a1a}
.crop em{font-weight:400;color:#666;font-style:normal;margin-left:4px}
.ts{font-size:11px;color:#999}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;margin-bottom:16px}
.mc{background:#fff;border-radius:8px;padding:10px 12px;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.mc .l{font-size:9px;text-transform:uppercase;letter-spacing:.4px;color:#999;font-weight:500}
.mc .v{font-size:16px;font-weight:600;margin-top:1px;display:flex;align-items:center;gap:5px}
.mc .s{font-size:9px;color:#aaa;margin-top:1px}
.na{opacity:.5}
.b{display:inline-block;padding:1px 7px;border-radius:20px;font-size:9px;font-weight:600}
.bg{background:#e6f4ea;color:#1e7a34}
.bm{background:#fef7e0;color:#b7650a}
.bp{background:#fce8e6;color:#b31412}
.recents{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
.recents::-webkit-scrollbar{height:3px}
.recents::-webkit-scrollbar-thumb{background:#ccc;border-radius:3px}
.rec{flex-shrink:0;width:64px;cursor:pointer;border-radius:6px;overflow:hidden;border:2px solid transparent}
.rec.act{border-color:#2e7d32}
.rec img{width:64px;height:64px;object-fit:cover;display:block}
.rec .t{font-size:8px;text-align:center;color:#666;padding:1px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media(max-width:500px){.grid{grid-template-columns:1fr 1fr}.imgwrap img{max-height:50vh}}
</style>
</head>
<body>

<div class="bar"><span>&#127793;</span><h1>Plant Health Monitor</h1></div>

<div class="content">

{% if not data %}

<div class="empty">
<span>&#128247;</span>
<h2>No captures yet</h2>
<p>Run <code>python rpi_predict.py</code> on RPi, then refresh.</p>
</div>

{% else %}
{% set d = data[0] %}

<div class="imgbox">
<div class="imgwrap"><img src="data:image/jpeg;base64,{{ d.img }}" alt=""></div>
<div class="imgfoot">
<div><span class="crop" id="crop">{{ d.crop }} <em>&mdash; {{ d.cond }}</em></span></div>
<span class="ts" id="ts">{{ d.time }}</span>
</div>
</div>

<div class="grid" id="grid">
<div class="mc">
<div class="l">Confidence</div>
<div class="v">{{ d.conf }}</div>
<div class="s">{{ d.label_short }}</div>
</div>
<div class="mc{{ ' na' if d.m == 'N/A' else '' }}">
<div class="l">Soil Moisture</div>
<div class="v" id="vm">{{ d.m if d.m == 'N/A' else d.m + '% <span class=\"b bg\">' + d.sc + '</span>' }}</div>
<div class="s">sensor reading</div>
</div>
<div class="mc{{ ' na' if d.t == 'N/A' else '' }}">
<div class="l">Temperature</div>
<div class="v" id="vt">{{ d.t if d.t == 'N/A' else d.t + '&deg;C' }}</div>
<div class="s" id="vh">{% if d.h != 'N/A' %}Humidity: {{ d.h }}%{% else %}Humidity: N/A{% endif %}</div>
</div>
<div class="mc{{ ' na' if d.sm == 'N/A' else '' }}">
<div class="l">Soil Model</div>
<div class="v" id="vsm">{{ '<span class=\"b ' + d.smc + '\">' + d.sm + '</span>' if d.sm != 'N/A' else 'N/A' }}</div>
<div class="s">Random Forest</div>
</div>
<div class="mc">
<div class="l">Recommendation</div>
<div class="v" id="vdiag" style="font-size:13px;font-weight:500;color:#555">{{ d.diag }}</div>
<div class="s">plant care</div>
</div>
<div class="mc">
<div class="l">Overall Health</div>
<div class="v" id="vov">{{ '<span class=\"b ' + d.ovc + '\">' + d.ov + '</span>' }}</div>
<div class="s">plant + soil</div>
</div>
</div>

{% if data|length > 1 %}
<div style="margin-bottom:16px">
<strong style="font-size:12px;color:#666">&#128196; History ({{ data|length }})</strong>
<div class="recents">
{% for r in data %}
<div class="rec{{ ' act' if loop.first else '' }}" onclick="pick({{ loop.index0 }})">
{% if r.img %}<img src="data:image/jpeg;base64,{{ r.img }}" alt="">{% else %}<div style="width:64px;height:64px;background:#ddd;display:flex;align-items:center;justify-content:center;color:#888;font-size:24px">&#128247;</div>{% endif %}
<div class="t">{{ r.label_short }}</div>
</div>
{% endfor %}
</div>
</div>
{% endif %}

{% endif %}
</div>

<script>
const D = {{ js | safe }};
function pick(i){
  const d=D[i];if(!d)return;
  document.querySelector('.imgwrap img').src='data:image/jpeg;base64,'+d.img;
  document.getElementById('crop').innerHTML=d.crop+' <em>&mdash; '+d.cond+'</em>';
  document.getElementById('ts').textContent=d.time;
  document.querySelectorAll('.mc .v')[0].textContent=d.conf;
  document.querySelectorAll('.mc .s')[0].textContent=d.label_short;
  const vm=document.getElementById('vm');
  if(d.m!=='N/A'){vm.innerHTML=d.m+'% <span class="b bg">'+d.sc+'</span>';document.querySelectorAll('.mc')[1].classList.remove('na')}
  else{vm.textContent='N/A'}
  const vt=document.getElementById('vt');
  vt.innerHTML=d.t!=='N/A'?d.t+'&deg;C':'N/A';
  document.getElementById('vh').textContent=d.h!=='N/A'?'Humidity: '+d.h+'%':'Humidity: N/A';
  const vsm=document.getElementById('vsm');
  vsm.innerHTML=d.sm!=='N/A'?'<span class="b '+d.smc+'">'+d.sm+'</span>':'N/A';
  document.getElementById('vdiag').textContent=d.diag;
  document.getElementById('vov').innerHTML='<span class="b '+d.ovc+'">'+d.ov+'</span>';
  document.querySelectorAll('.rec').forEach((e,j)=>e.classList.toggle('act',j===i));
}
</script>
</body>
</html>'''


def badge_class(val):
    if val in ('Good', 'Excellent', 'Healthy'): return 'bg'
    if val in ('Moderate', 'Needs Attention', 'Fair'): return 'bm'
    return 'bp'


def overall(label, soil_cond, sm_cond):
    h = 'healthy' in label.lower()
    n = 0 if h else 1
    if soil_cond in ('Poor', 'N/A'): n += 1
    if sm_cond == 'Poor': n += 1
    elif sm_cond == 'Moderate': n += 0.5
    if n == 0: return 'Excellent', 'bg'
    if n <= 1: return 'Good', 'bg'
    if n <= 2: return 'Needs Attention', 'bm'
    return 'Critical', 'bp'


@app.route('/')
def index():
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row['label']
                parts = label.split('___') if '___' in label else [label, '']
                crop = parts[0]
                cond = parts[1] if len(parts) > 1 else ''
                soil_cond = row.get('soil_condition', 'N/A')
                sm_val = row.get('soil_model', 'N/A')
                ov, ovc = overall(label, soil_cond, sm_val)
                img_path = CAPTURES / row['filename']
                b64 = ''
                if img_path.exists():
                    with open(img_path, 'rb') as imgf:
                        b64 = base64.b64encode(imgf.read()).decode()
                rows.append(dict(
                    img=b64, crop=crop, cond=cond,
                    label_short=label[-20:] if len(label) > 22 else label,
                    conf=f"{float(row['confidence']):.1%}",
                    m=row.get('moisture', 'N/A'),
                    sc=soil_cond, sm=sm_val,
                    smc=badge_class(sm_val) if sm_val != 'N/A' else 'bm',
                    t=row.get('temperature', 'N/A'),
                    h=row.get('humidity', 'N/A'),
                    diag=row.get('diagnosis', ''),
                    time=row['timestamp'],
                    ov=ov, ovc=ovc,
                ))
    import json
    data = list(reversed(rows))
    return render_template_string(HTML, data=data, js=json.dumps(data))


if __name__ == '__main__':
    print('http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
