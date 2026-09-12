"""
Build a single self-contained interactive HTML viewer for every plot in
main/g_plots/, styled after experiment_artifacts/pa_ipip_full/pa_ipip_bestbar.html.

Each plot is one (model, steering direction, beta) grouped bar chart. We walk the
g_plots tree, base64-embed every PNG, and emit one portable HTML file with
model / direction / beta selectors.
"""
import os, re, json, base64

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "g_plots")
OUT  = os.path.join(HERE, "g_plots_steering.html")

png_re = re.compile(r"agent_(low|high)_beta([0-9.]+)\.png$")

# model_key -> {"family","variant","label","dirs":{LOW/HIGH:{beta:dataURI}}}
models = {}
for dirpath, _, files in os.walk(ROOT):
    for f in files:
        m = png_re.search(f)
        if not m:
            continue
        direction, beta = m.group(1).upper(), m.group(2)
        # model dir = the directory ABOVE AGENT_LOW/AGENT_HIGH
        agent_dir = os.path.dirname(os.path.join(dirpath, f))   # .../AGENT_HIGH
        model_dir = os.path.dirname(agent_dir)
        rel = os.path.relpath(model_dir, ROOT)                  # e.g. Vicuna/vicuna-7b-v1.5
        parts = rel.split(os.sep)
        family = parts[0]
        variant = parts[-1]
        label = variant if variant == family else f"{family} / {variant}"
        with open(os.path.join(dirpath, f), "rb") as fh:
            uri = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
        mk = models.setdefault(rel, {"family": family, "variant": variant,
                                     "label": label, "dirs": {}})
        mk["dirs"].setdefault(direction, {})[beta] = uri

# stable ordering: by family, then 7b before 13b
def sizenum(v):
    mm = re.search(r"(\d+)\s*b", v.lower())
    return int(mm.group(1)) if mm else 0
ordered = sorted(models.items(), key=lambda kv: (kv[1]["family"].lower(), sizenum(kv[1]["variant"])))
DATA = {k: v for k, v in ordered}
BETAS = ["0.01", "0.1", "0.5"]

print(f"{len(DATA)} models, "
      f"{sum(len(b) for v in DATA.values() for b in v['dirs'].values())} plots embedded")

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering open SFT models toward high or low Big-Five traits</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&family=IBM+Plex+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0a0a; --ink:#f0f0f0; --muted:#7a7a7a; --faint:#3a3a3a;
    --line:#1f1f1f; --base:#5a5a5a;
    --sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{background:var(--bg);color:var(--ink)}
  body{margin:0;font-family:var(--sans);font-weight:300;-webkit-font-smoothing:antialiased;letter-spacing:.01em}
  .wrap{max-width:1000px;margin:0 auto;padding:64px 40px 100px}
  .eyebrow{font-family:var(--mono);font-size:30px;font-weight:400;letter-spacing:.18em;text-transform:uppercase;color:var(--ink);margin:0 0 26px}
  h1{font-size:38px;font-weight:200;letter-spacing:-.015em;line-height:1.08;margin:0 0 18px}
  .sub{color:var(--muted);max-width:900px;margin:0 0 48px;font-size:21px;font-weight:300;line-height:1.7}
  .sub b{color:var(--ink);font-weight:500}
  .sub code{font-family:var(--mono);font-size:19px;color:var(--ink);font-weight:400}
  .rule{height:1px;background:var(--line);border:0;margin:0 0 40px}

  .explain{margin:0 0 52px;padding:36px 38px;border:1px solid var(--line);border-radius:14px;background:#0e0e0e}
  .explain h2{font-family:var(--mono);font-size:12px;font-weight:300;letter-spacing:.26em;text-transform:uppercase;color:var(--muted);margin:0 0 26px}
  .formula{font-family:var(--mono);font-weight:300;font-size:26px;line-height:1.2;letter-spacing:-.01em;color:var(--ink);margin:0}
  .formula .pos{color:#5b8def}.formula .neg{color:#e06}
  .explain .def{font-size:19px;line-height:1.65;color:#bdbdbd;font-weight:300;margin:24px 0 30px;max-width:790px}
  .explain .def code{font-family:var(--mono);font-size:16px;color:var(--ink);font-weight:400}
  .explain .def b{color:var(--ink);font-weight:500}
  .states{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}
  .state{padding:20px 22px;border:1px solid var(--line);border-radius:11px;display:flex;gap:14px;align-items:flex-start}
  .swatch{width:15px;height:15px;border-radius:3px;flex:0 0 auto;margin-top:4px}
  .state .lab{font-family:var(--mono);font-size:15px;font-weight:400;letter-spacing:-.01em}
  .state .txt{font-size:14.5px;color:var(--muted);margin-top:7px;line-height:1.55;font-weight:300}
  @media(max-width:720px){.states{grid-template-columns:1fr}.formula{font-size:20px}}

  .controls{display:flex;flex-wrap:wrap;gap:38px;align-items:flex-end;margin-bottom:46px}
  .control{display:flex;flex-direction:column;gap:10px}
  .control label{font-family:var(--mono);font-size:10.5px;font-weight:300;letter-spacing:.24em;text-transform:uppercase;color:var(--muted)}
  select{appearance:none;background:transparent;color:var(--ink);border:0;border-bottom:1px solid var(--ink);border-radius:0;
         padding:6px 28px 7px 2px;font-family:var(--sans);font-size:16px;font-weight:300;cursor:pointer;letter-spacing:.01em;
         background-image:linear-gradient(45deg,transparent 50%,var(--ink) 50%),linear-gradient(135deg,var(--ink) 50%,transparent 50%);
         background-position:right 8px center,right 3px center;background-size:5px 5px,5px 5px;background-repeat:no-repeat}
  select:focus{outline:none;border-bottom-width:2px}
  select option{color:#111;background:#fff}
  optgroup{color:#111}
  .seg{display:flex;border:1px solid var(--ink);border-radius:999px;overflow:hidden}
  .seg button{background:transparent;color:var(--ink);border:0;border-right:1px solid var(--line);
         padding:9px 20px;font-family:var(--mono);font-size:11px;font-weight:300;letter-spacing:.16em;
         text-transform:uppercase;cursor:pointer;transition:background .15s,color .15s}
  .seg button:last-child{border-right:0}
  .seg button.on{background:var(--ink);color:var(--bg)}
  button.pill{background:transparent;color:var(--ink);border:1px solid var(--ink);border-radius:999px;padding:9px 22px;
         font-family:var(--mono);font-size:11px;font-weight:300;letter-spacing:.18em;text-transform:uppercase;cursor:pointer;transition:background .15s,color .15s}
  button.pill:hover{background:var(--ink);color:var(--bg)}
  .slider-row{flex:1;min-width:280px}
  input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:1px;background:var(--faint);margin:18px 0 10px;cursor:pointer}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:13px;height:13px;border-radius:50%;background:var(--ink);border:0}
  input[type=range]::-moz-range-thumb{width:13px;height:13px;border-radius:50%;background:var(--ink);border:0}
  .nlabel{font-family:var(--mono);font-size:12px;font-weight:300;color:var(--muted);letter-spacing:.04em}
  .nlabel b{font-weight:400;color:var(--ink);font-size:13px}

  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-bottom:48px}
  .card{padding:22px 24px;border-right:1px solid var(--line)}
  .card:last-child{border-right:0}
  .card .k{font-family:var(--mono);font-size:9.5px;font-weight:300;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
  .card .v{font-family:var(--mono);font-size:24px;font-weight:300;letter-spacing:-.01em;margin-top:12px;line-height:1.15}
  .card .v small{font-size:13px;color:var(--muted)}

  .charthead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:20px;flex-wrap:wrap;gap:16px}
  .charthead .t{font-size:20px;font-weight:500;letter-spacing:.01em}
  .legend{display:flex;gap:24px;flex-wrap:wrap;font-family:var(--mono);font-size:14px;font-weight:300;letter-spacing:.04em;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:9px}
  .gd{width:13px;height:13px;border-radius:3px;display:inline-block}
  .gd.grey{background:#7f7f7f}.gd.blue{background:#4C72B0}.gd.green{background:#55A868}.gd.red{background:#C44E52}
  .plot{background:#ffffff;border-radius:6px;padding:14px;min-height:200px}
  .plot img{width:100%;height:auto;display:block}
  .foot{color:var(--muted);font-size:21px;font-weight:300;line-height:1.7;margin-top:44px;padding-top:24px;border-top:1px solid var(--line);max-width:900px}
  .foot code{font-family:var(--mono);font-size:19px;color:var(--ink)}
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">DPO &middot; OCEAN steering &middot; open SFT models</p>
  <h1>Can DPO push a model toward a high or low Big-Five profile?</h1>
  <p class="sub" id="sub">
    Each open <b>SFT-only</b> model is fine-tuned with <b>QLoRA-DPO</b> on a set of OCEAN
    preference pairs that push it either <b>toward high</b> or <b>toward low</b> on the Big-Five
    traits, at three DPO strengths <code>&beta;</code>. We then re-measure its personality with
    <b>three held-out questionnaires</b> &mdash; <b style="color:#7aa0d8">BFI-2</b>,
    <b style="color:#7fc99a">Goldberg-100</b>, and <b style="color:#e08a8f">FFPI</b> &mdash; and compare
    against the untuned <b style="color:#b6b6b6">baseline</b>. Each group of bars is one Big-Five trait;
    error bars are 95% intervals. Pick a model, flip the steering direction, and slide
    <code>&beta;</code> to watch the colored bars move away from grey.
  </p>
  <hr class="rule">

  <section class="explain">
    <h2>What you are looking at</h2>
    <div class="formula">baseline (SFT) &nbsp;<span style="color:#7a7a7a">+ DPO toward</span>&nbsp; <span class="pos">HIGH</span> / <span class="neg">LOW</span> &nbsp;<span style="color:#7a7a7a">at strength</span>&nbsp; &beta;</div>
    <p class="def">
      The DPO preference pairs come from OCEAN agent answers; eval uses three <b>different</b>
      inventories than the training source, so the score shift is not just memorizing the test.
      A bigger <code>&beta;</code> means a stronger pull away from the base model. If steering works,
      the three colored bars should rise above grey in the <b>HIGH</b> direction and drop below grey in
      the <b>LOW</b> direction &mdash; and roughly agree across the three questionnaires.
    </p>
    <div class="states">
      <div class="state"><span class="swatch" style="background:#7f7f7f"></span><div>
        <div class="lab" style="color:#b6b6b6">Baseline (SFT)</div>
        <div class="txt">The untuned model, averaged over the three questionnaires. The same reference in every panel.</div></div>
      </div>
      <div class="state"><span class="swatch" style="background:#4C72B0"></span><div>
        <div class="lab" style="color:#7aa0d8">BFI-2</div>
        <div class="txt">Big Five Inventory&ndash;2. One held-out measure of where the steered model lands.</div></div>
      </div>
      <div class="state"><span class="swatch" style="background:#55A868"></span><div>
        <div class="lab" style="color:#7fc99a">Goldberg-100</div>
        <div class="txt">Goldberg's 100-item IPIP markers. A second independent read on the same traits.</div></div>
      </div>
      <div class="state"><span class="swatch" style="background:#C44E52"></span><div>
        <div class="lab" style="color:#e08a8f">FFPI</div>
        <div class="txt">Five-Factor Personality Inventory. A third instrument &mdash; agreement across all three is the real signal.</div></div>
      </div>
    </div>
  </section>

  <div class="controls">
    <div class="control">
      <label for="model">Model</label>
      <select id="model"></select>
    </div>
    <div class="control">
      <label>Steering direction</label>
      <div class="seg" id="dir">
        <button data-d="LOW">&#9660;&ensp;Low</button>
        <button data-d="HIGH">&#9650;&ensp;High</button>
      </div>
    </div>
    <div class="control"><button class="pill" id="play">&#9654;&ensp;Sweep &beta;</button></div>
    <div class="control slider-row">
      <label for="beta">DPO strength &beta;</label>
      <input type="range" id="beta" min="0" max="2" value="1" step="1">
      <div class="nlabel">&beta;&ensp;=&ensp;<b id="betaval">0.1</b><span id="betanote"></span></div>
    </div>
  </div>

  <div class="cards">
    <div class="card"><div class="k">Model</div><div class="v" id="c-model"></div></div>
    <div class="card"><div class="k">Family</div><div class="v" id="c-family"></div></div>
    <div class="card"><div class="k">Steering</div><div class="v" id="c-dir"></div></div>
    <div class="card"><div class="k">DPO strength</div><div class="v" id="c-beta"></div></div>
  </div>

  <div class="charthead">
    <div class="t" id="chart-title"></div>
    <div class="legend">
      <span><i class="gd grey"></i>baseline (SFT)</span>
      <span><i class="gd blue"></i>BFI-2</span>
      <span><i class="gd green"></i>Goldberg-100</span>
      <span><i class="gd red"></i>FFPI</span>
    </div>
  </div>
  <div class="plot"><img id="plot" alt=""></div>

  <p class="foot">
    Each model is QLoRA-DPO fine-tuned on OCEAN preference pairs (<code>agent_low</code> /
    <code>agent_high</code>) at <code>&beta; &isin; {0.01, 0.1, 0.5}</code>, then evaluated on
    <b>BFI-2</b>, <b>Goldberg-100</b>, and <b>FFPI</b> &mdash; three inventories held out from the
    DPO source (IPIP-300). Bars are mean trait scores (1&ndash;5) with 95% error bars; grey is the
    untuned baseline averaged over the three inventories. Models shown:
    <span id="foot-models"></span>.
  </p>
</div>

<script>
const DATA = __DATA__;
const BETAS = __BETAS__;
const $ = id => document.getElementById(id);
const modelSel=$("model"), betaSlider=$("beta");
const KEYS = Object.keys(DATA);

// group dropdown options by family
const byFamily = {};
KEYS.forEach(k=>{ (byFamily[DATA[k].family] ||= []).push(k); });
Object.keys(byFamily).forEach(fam=>{
  const og=document.createElement("optgroup"); og.label=fam;
  byFamily[fam].forEach(k=>{ const o=document.createElement("option");
    o.value=k; o.textContent=DATA[k].variant; og.appendChild(o); });
  modelSel.appendChild(og);
});
$("foot-models").textContent = KEYS.map(k=>DATA[k].label).join(", ");

let model=KEYS[0], dir="HIGH", bi=1;

function availDirs(){ return Object.keys(DATA[model].dirs); }
function dirHas(d){ return availDirs().includes(d); }

function syncDirButtons(){
  document.querySelectorAll("#dir button").forEach(b=>{
    const d=b.getAttribute("data-d");
    b.classList.toggle("on", d===dir);
    b.disabled = !dirHas(d);
    b.style.opacity = dirHas(d) ? 1 : .3;
  });
}

function draw(){
  if(!dirHas(dir)) dir = availDirs()[0];
  const beta = BETAS[bi];
  const m = DATA[model];
  $("plot").src = m.dirs[dir][beta];
  $("plot").alt = m.label+" — steer "+dir+" — beta "+beta;
  $("chart-title").textContent = "OCEAN trait scores  —  "+m.label+"  ("+(dir==="HIGH"?"steer high":"steer low")+", β = "+beta+")";
  $("betaval").textContent = beta;
  $("c-model").textContent = m.variant;
  $("c-family").textContent = m.family;
  $("c-dir").innerHTML = dir==="HIGH" ? "High <small>▲</small>" : "Low <small>▼</small>";
  $("c-beta").innerHTML = "β = "+beta+" <small>"+(beta==="0.01"?"(gentle)":beta==="0.5"?"(strong)":"(mid)")+"</small>";
  betaSlider.value = bi;
  $("betanote").innerHTML = "&emsp;/&emsp;"+BETAS.join(" · ");
  syncDirButtons();
}

modelSel.addEventListener("change",e=>{ model=e.target.value; draw(); });
document.querySelectorAll("#dir button").forEach(b=>{
  b.addEventListener("click",()=>{ if(b.disabled) return; dir=b.getAttribute("data-d"); draw(); });
});
betaSlider.addEventListener("input",e=>{ bi=+e.target.value; draw(); });

let timer=null;
$("play").addEventListener("click",()=>{
  if(timer){clearInterval(timer);timer=null;$("play").innerHTML="&#9654;&ensp;Sweep β";return;}
  $("play").innerHTML="&#10074;&#10074;&ensp;Pause";
  bi=0; draw();
  timer=setInterval(()=>{
    if(bi>=BETAS.length-1){clearInterval(timer);timer=null;$("play").innerHTML="&#9654;&ensp;Sweep β";return;}
    bi++; draw();
  },700);
});

draw();
</script>
</body>
</html>
"""

html = (HTML
        .replace("__DATA__", json.dumps(DATA, separators=(",", ":")))
        .replace("__BETAS__", json.dumps(BETAS)))
with open(OUT, "w") as f:
    f.write(html)
print("wrote", OUT, f"({os.path.getsize(OUT)/1024/1024:.2f} MB)")
