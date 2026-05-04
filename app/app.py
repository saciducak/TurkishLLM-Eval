"""
TurkishLLM-Eval Leaderboard — Ultra Premium UI
Port: 7847
"""

import json
from pathlib import Path
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SAMPLE_DATA = Path(__file__).parent / "sample_results.json"
PORT = 7847
BENCHMARKS = ["truthfulqa_tr", "mmlu_tr", "hallucination_tr", "bias_tr"]
BENCH_LABELS = {"truthfulqa_tr": "TruthfulQA-TR", "mmlu_tr": "MMLU-TR",
                "hallucination_tr": "Hallucination", "bias_tr": "Bias Detection"}

# Ultra-premium enterprise aesthetic palette (Apple/Vercel inspired, monochromatic & soft accents)
PALETTE = ["#E2E8F0", "#94A3B8", "#64748B", "#475569", "#334155", "#1E293B", "#0F172A"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-app: #000000;
  --bg-card: #09090b;
  --bg-card-hover: #18181b;
  --border-light: #27272a;
  --text-main: #fafafa;
  --text-muted: #a1a1aa;
  --accent-cyan: #e4e4e7; /* Replaced neon with crisp metallic gray/white */
  --accent-teal: #d4d4d8;
}

* { font-family: 'Plus Jakarta Sans', sans-serif !important; box-sizing: border-box; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

body, .gradio-container { background-color: var(--bg-app) !important; color: var(--text-main) !important; }
.gradio-container { max-width: 1400px !important; padding: 0 32px !important; margin: 0 auto !important; }

/* ── Hero Section ── */
.hero {
    padding: 60px 0 40px;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 40px;
    background: transparent;
}
.hero h1 {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 12px;
    color: var(--text-main);
}
.hero p {
    font-size: 16px;
    color: var(--text-muted);
    max-width: 600px;
    line-height: 1.6;
    margin: 0 0 24px;
}
.badges { display: flex; gap: 12px; flex-wrap: wrap; }
.badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid;
}
.badge-teal { background: rgba(255, 255, 255, 0.05); color: var(--text-muted); border-color: rgba(255, 255, 255, 0.1); }
.badge-cyan { background: rgba(255, 255, 255, 0.08); color: var(--text-main); border-color: rgba(255, 255, 255, 0.2); }

/* ── Metrics Row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin-bottom: 40px;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 24px;
    transition: all 0.2s ease;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    border-color: #3f3f46;
}
.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: var(--text-main);
    line-height: 1;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 13px;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Custom Leaderboard Table ── */
.custom-table-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 30px;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
}
.custom-table th {
    background: rgba(0,0,0,0.2);
    padding: 16px 20px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border-light);
}
.custom-table td {
    padding: 16px 20px;
    font-size: 14px;
    border-bottom: 1px solid var(--border-light);
    color: var(--text-main);
    vertical-align: middle;
}
.custom-table tr:last-child td { border-bottom: none; }
.custom-table tr:hover { background: var(--bg-card-hover); }
.custom-table .rank { font-weight: 700; color: var(--text-muted); width: 50px; }
.custom-table .model-name { font-weight: 600; color: var(--text-main); font-size: 15px; }
.custom-table .score-main { font-weight: 700; color: #18181b; background: var(--text-main); padding: 4px 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); }

/* ── Gradio Tabs Overrides ── */
.tabs { background: transparent !important; border: none !important; }
.tab-nav { border-bottom: 1px solid var(--border-light) !important; background: transparent !important; margin-bottom: 24px !important; padding: 0 !important;}
.tab-nav button { background: transparent !important; border: none !important; color: var(--text-muted) !important; font-weight: 600 !important; font-size: 15px !important; padding: 12px 24px !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important;}
.tab-nav button.selected { color: var(--text-main) !important; border-bottom-color: var(--text-main) !important; }
.tab-nav button:hover { color: var(--text-main) !important; }

/* ── Plots & Container ── */
.plot-container { background: var(--bg-card) !important; border: 1px solid var(--border-light) !important; border-radius: 16px !important; padding: 16px !important;}

/* ── Dropdowns & Buttons ── */
.wrap { background: var(--bg-card) !important; border: 1px solid var(--border-light) !important; border-radius: 8px !important; }
input, select, textarea { background: var(--bg-card) !important; color: var(--text-main) !important; border: none !important; font-size: 14px !important; }
.primary { background: var(--text-main) !important; color: #000 !important; font-weight: 600 !important; border-radius: 8px !important; transition: all 0.2s !important; }
.primary:hover { opacity: 0.9 !important; transform: scale(1.01); }

/* ── Markdown ── */
.prose { color: var(--text-muted) !important; font-size: 15px !important; line-height: 1.7 !important; }
.prose h2, .prose h3 { color: var(--text-main) !important; font-weight: 700 !important; margin-top: 32px !important; }
.prose strong { color: var(--text-main) !important; }
.prose table { width: 100%; border-collapse: collapse; margin: 20px 0; border: 1px solid var(--border-light); border-radius: 12px; overflow: hidden; }
.prose th { background: rgba(0,0,0,0.2); padding: 12px; text-align: left; font-size: 13px; color: var(--text-muted); }
.prose td { padding: 12px; border-top: 1px solid var(--border-light); }

.footer { text-align: center; margin-top: 60px; padding: 30px 0; border-top: 1px solid var(--border-light); color: var(--text-muted); font-size: 13px; }
.footer a { color: var(--accent-teal); text-decoration: none; }
.footer a:hover { color: var(--accent-cyan); text-decoration: underline; }
"""

# Plotly Base Theme Configuration for dark premium aesthetic
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#a1a1aa", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#fafafa", size=12), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=40, l=40, r=20)
)

def load_data():
    if SAMPLE_DATA.exists():
        with open(SAMPLE_DATA, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def generate_leaderboard_html(data):
    sd = sorted(data, key=lambda x: x["turkeval_score"], reverse=True)
    html = '<div class="custom-table-wrapper"><table class="custom-table">'
    html += '<thead><tr><th class="rank">#</th><th>Model</th><th>Developer</th><th>Params</th><th>TurkEval™</th><th>TruthfulQA</th><th>MMLU</th><th>Halluc.</th><th>Bias</th></tr></thead>'
    html += '<tbody>'
    for i, m in enumerate(sd, 1):
        html += f'''<tr>
            <td class="rank">{i}</td>
            <td class="model-name">{m["model_name"]}</td>
            <td style="color: var(--text-muted);">{m.get("developer", "-")}</td>
            <td style="color: var(--text-muted);">{m.get("parameters", "-")}</td>
            <td><span class="score-main">{m["turkeval_score"]:.1f}</span></td>
            <td>{m.get("truthfulqa_tr", 0):.1f}</td>
            <td>{m.get("mmlu_tr", 0):.1f}</td>
            <td>{m.get("hallucination_tr", 0):.1f}</td>
            <td>{m.get("bias_tr", 0):.1f}</td>
        </tr>'''
    html += '</tbody></table></div>'
    return html

def ranking_chart(data):
    sd = sorted(data, key=lambda x: x["turkeval_score"], reverse=True)
    colors = [PALETTE[0] if s["turkeval_score"] >= 80 else PALETTE[1] if s["turkeval_score"] >= 65 else PALETTE[3] for s in sd]
    
    fig = go.Figure(go.Bar(
        x=[d["turkeval_score"] for d in sd], 
        y=[d["model_name"] for d in sd],
        orientation="h", 
        marker=dict(color=colors, cornerradius=6),
        text=[f'{d["turkeval_score"]:.1f}' for d in sd], 
        textposition="outside",
        textfont=dict(color="#fafafa", size=13, family="Plus Jakarta Sans"),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        **PL, 
        height=400, 
        title=dict(text="Score Distribution", font=dict(size=18, color="#fafafa", weight="bold"), x=0),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[0, max(d["turkeval_score"] for d in sd) * 1.15], gridcolor="#27272a")
    )
    return fig

def radar_chart(data):
    fig = go.Figure()
    top4 = sorted(data, key=lambda x: x["turkeval_score"], reverse=True)[:4]
    labels = ["TruthfulQA", "MMLU", "Hallucination", "Bias"]
    
    for i, m in enumerate(top4):
        vals = [m.get(b, 0) for b in BENCHMARKS] + [m.get(BENCHMARKS[0], 0)]
        c = PALETTE[i % len(PALETTE)]
        
        h = c.lstrip('#')
        rgba_fill = f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, 0.15)"
        
        fig.add_trace(go.Scatterpolar(
            r=vals, 
            theta=labels + [labels[0]], 
            fill="toself", 
            name=m["model_name"],
            line=dict(color=c, width=3),
            fillcolor=rgba_fill,
            marker=dict(size=8, color=c)
        ))
        
    fig.update_layout(
        **PL, 
        height=500, 
        title=dict(text="Multidimensional Capability Profile (Top 4)", font=dict(size=18, color="#fafafa", weight="bold"), x=0),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#27272a", linecolor="#27272a", tickfont=dict(color="#a1a1aa")),
            angularaxis=dict(gridcolor="#27272a", linecolor="#27272a", tickfont=dict(color="#fafafa", size=13))
        )
    )
    return fig

def bar_chart(data):
    rows = []
    for m in data:
        for b in BENCHMARKS:
            rows.append({"Model": m["model_name"], "Benchmark": BENCH_LABELS[b], "Score": m.get(b, 0)})
    
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Model", y="Score", color="Benchmark", barmode="group",
        color_discrete_sequence=PALETTE[:4], 
        height=450
    )
    
    fig.update_layout(
        **PL, 
        title=dict(text="Detailed Benchmark Breakdown", font=dict(size=18, color="#fafafa", weight="bold"), x=0),
        xaxis_tickangle=-30, 
        bargap=0.2, 
        bargroupgap=0.1
    )
    fig.update_traces(marker_cornerradius=4)
    return fig

def heatmap_chart(data):
    sd = sorted(data, key=lambda x: x["turkeval_score"], reverse=True)
    z = [[d.get(b, 0) for b in BENCHMARKS] for d in sd]
    
    colorscale = [
        [0, "#000000"], 
        [0.4, "#09090b"], 
        [0.7, "#27272a"], 
        [1, "#fafafa"]
    ]
    
    fig = go.Figure(go.Heatmap(
        z=z, 
        x=list(BENCH_LABELS.values()), 
        y=[d["model_name"] for d in sd],
        colorscale=colorscale,
        text=[[f"{v:.1f}" for v in row] for row in z], 
        texttemplate="%{text}",
        textfont=dict(size=13, color="#FFFFFF", family="Plus Jakarta Sans"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
        showscale=False
    ))
    
    fig.update_layout(
        **PL, 
        height=400, 
        title=dict(text="Density Matrix", font=dict(size=18, color="#fafafa", weight="bold"), x=0),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=13)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=13))
    )
    return fig

def compare(m1, m2, data):
    d1 = next((d for d in data if d["model_name"] == m1), None)
    d2 = next((d for d in data if d["model_name"] == m2), None)
    if not d1 or not d2:
        return go.Figure(), "Please select two distinct models for comparison."
        
    v1, v2 = [d1.get(b, 0) for b in BENCHMARKS], [d2.get(b, 0) for b in BENCHMARKS]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name=m1, x=list(BENCH_LABELS.values()), y=v1, marker_color=PALETTE[0], marker_cornerradius=4))
    fig.add_trace(go.Bar(name=m2, x=list(BENCH_LABELS.values()), y=v2, marker_color=PALETTE[3], marker_cornerradius=4))
    
    fig.update_layout(
        **PL, 
        barmode="group", 
        height=450,
        title=dict(text=f"Head-to-Head: {m1} vs {m2}", font=dict(size=18, color="#fafafa", weight="bold"), x=0),
        bargroupgap=0.1
    )
    
    diff = d1["turkeval_score"] - d2["turkeval_score"]
    winner = m1 if diff > 0 else m2 if diff < 0 else "Tie"
    
    md = f"### Ultimate Winner: **{winner}** &nbsp;(Margin: {abs(diff):.1f} pts)\n\n"
    md += f"| Evaluation Axis | {m1} | {m2} | Delta |\n|:---|:---|:---|:---|\n"
    for b in BENCHMARKS:
        s1, s2 = d1.get(b, 0), d2.get(b, 0)
        d = s1 - s2
        arrow = '🟢 ↑' if d>0 else '🔴 ↓' if d<0 else '⚪ ='
        md += f"| **{BENCH_LABELS[b]}** | {s1:.1f} | {s2:.1f} | {arrow} {abs(d):.1f} |\n"
    md += f"| **TurkEval™ Composite** | **{d1['turkeval_score']:.1f}** | **{d2['turkeval_score']:.1f}** | **{diff:+.1f}** |"
    
    return fig, md

def build():
    data = load_data()
    names = [d["model_name"] for d in sorted(data, key=lambda x: x["turkeval_score"], reverse=True)]
    leaderboard_html = generate_leaderboard_html(data)

    with gr.Blocks(title="TurkishLLM-Eval Enterprise") as app:
        
        # Hero Section
        gr.HTML("""
        <div class="hero">
            <h1>TurkishLLM-Eval</h1>
            <p>The definitive benchmark platform for Turkish Large Language Models. Evaluates truthfulness, reasoning, hallucination rates, and cultural bias using a rigorous multi-judge ensemble.</p>
            <div class="badges">
                <span class="badge badge-cyan">v0.1.0-beta</span>
                <span class="badge badge-teal">GPT-4o + Claude 3.5 Judges</span>
                <span class="badge badge-teal">80+ Curated Scenarios</span>
            </div>
        </div>
        """)
        
        # KPI Row
        gr.HTML(f"""
        <div class="metrics-row">
            <div class="metric-card"><div class="metric-value">{len(data)}</div><div class="metric-label">Models Evaluated</div></div>
            <div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Core Dimensions</div></div>
            <div class="metric-card"><div class="metric-value">160+</div><div class="metric-label">LLM Judgments</div></div>
            <div class="metric-card"><div class="metric-value">1</div><div class="metric-label">Standardized Score</div></div>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Leaderboard"):
                gr.HTML(leaderboard_html)
                gr.Plot(value=ranking_chart(data))

            with gr.Tab("Deep Analysis"):
                with gr.Row():
                    gr.Plot(value=radar_chart(data))
                gr.Plot(value=bar_chart(data))
                gr.Plot(value=heatmap_chart(data))

            with gr.Tab("Head-to-Head"):
                with gr.Row():
                    dd1 = gr.Dropdown(choices=names, label="Primary Model", value=names[0] if names else None)
                    dd2 = gr.Dropdown(choices=names, label="Comparison Model", value=names[1] if len(names) > 1 else None)
                btn = gr.Button("Execute Comparison", variant="primary")
                plot = gr.Plot()
                md = gr.Markdown()
                btn.click(fn=lambda a, b: compare(a, b, data), inputs=[dd1, dd2], outputs=[plot, md])

            with gr.Tab("Methodology"):
                gr.Markdown("""
## Evaluation Architecture

The **TurkEval™** composite score is a weighted aggregation designed specifically for enterprise and production readiness in the Turkish language.

**TurkEval™ = (0.30 × TruthfulQA) + (0.25 × MMLU) + (0.25 × Anti-Hallucination) + (0.20 × Anti-Bias)**

### Confidence Intervals & Grading

| Grade | Threshold | Enterprise Implication |
|:------|:----------|:-----------------------|
| **A+** | ≥ 90 | Production-ready for critical Turkish deployments. |
| **A** | ≥ 80 | Highly reliable with minor, predictable edge cases. |
| **B** | ≥ 70 | Usable in human-in-the-loop workflows. |
| **C** | ≥ 60 | Significant limitations; high risk of hallucination. |
| **F** | < 50 | Not recommended for Turkish language tasks. |

### Multi-Judge Ensemble

To mitigate single-model bias (such as GPT-4 preferring its own outputs), we utilize a weighted ensemble:
- **Primary Judge:** GPT-4o (Weight: 0.55)
- **Secondary Judge:** Claude 3.5 Sonnet (Weight: 0.45)

*Inter-judge agreement is strictly monitored via Cohen's Kappa (κ).*

### Turkey-Specific Bias Taxonomy

Evaluations are rooted in the cultural context of Turkey, covering:
- **Gender Dynamics:** Professional stereotypes, familial roles in Turkish society.
- **Ethnic & Regional:** Perceptions of demographic groups (e.g., East vs. West dynamics).
- **Sectarian & Socioeconomic:** Class, education, and religious biases.
                """)

        gr.HTML('''
        <div class="footer">
            Built by <a href="https://github.com/saciducak" target="_blank">Muhammed Sacid Ucak</a> 
            <span style="margin: 0 12px; color: var(--border-light);">|</span> 
            <a href="https://github.com/saciducak/TurkishLLM-Eval" target="_blank">GitHub Repository</a>
            <span style="margin: 0 12px; color: var(--border-light);">|</span> 
            Apache 2.0 License
        </div>
        ''')
        
    return app

if __name__ == "__main__":
    app = build()
    app.launch(server_port=PORT, server_name="0.0.0.0", share=False, css=CSS)

