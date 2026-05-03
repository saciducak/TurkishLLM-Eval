"""
TurkishLLM-Eval Leaderboard — Premium Gradio UI
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
BENCH_LABELS = {
    "truthfulqa_tr": "🎯 TruthfulQA-TR",
    "mmlu_tr": "📚 MMLU-TR",
    "hallucination_tr": "🔍 Hallucination",
    "bias_tr": "⚖️ Bias Detection",
}

# ─── Premium Color Palette ───
COLORS = {
    "bg": "#0a0a0f",
    "card": "#12121a",
    "border": "#1e1e2e",
    "accent1": "#00e5a0",   # Emerald
    "accent2": "#7c5cfc",   # Violet
    "accent3": "#ff6b6b",   # Coral
    "accent4": "#ffd93d",   # Gold
    "text": "#e8e8ef",
    "muted": "#6b6b80",
}

CHART_COLORS = ["#00e5a0", "#7c5cfc", "#ff6b6b", "#ffd93d", "#00b4d8", "#ff85a1", "#b8f2e6", "#c9b1ff"]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Space Grotesk', sans-serif !important; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

.gradio-container {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f14 100%) !important;
    max-width: 1400px !important;
}

/* Hero Header */
.hero-section {
    background: linear-gradient(135deg, rgba(124,92,252,0.08) 0%, rgba(0,229,160,0.06) 100%);
    border: 1px solid rgba(124,92,252,0.15);
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,229,160,0.04) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(124,92,252,0.04) 0%, transparent 50%);
    animation: pulse 8s ease-in-out infinite alternate;
}
@keyframes pulse {
    0% { opacity: 0.5; transform: scale(1); }
    100% { opacity: 1; transform: scale(1.05); }
}
.hero-title {
    font-size: 2.4em;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5a0 0%, #7c5cfc 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    color: #9999b0;
    font-size: 1.05em;
    position: relative;
    z-index: 1;
}
.hero-badges {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.badge {
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.8em;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.badge-emerald { background: rgba(0,229,160,0.12); color: #00e5a0; border: 1px solid rgba(0,229,160,0.25); }
.badge-violet { background: rgba(124,92,252,0.12); color: #7c5cfc; border: 1px solid rgba(124,92,252,0.25); }
.badge-coral { background: rgba(255,107,107,0.12); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.25); }
.badge-gold { background: rgba(255,217,61,0.12); color: #ffd93d; border: 1px solid rgba(255,217,61,0.25); }

/* Tabs */
.tabs { border: none !important; }
.tab-nav { 
    background: rgba(18,18,26,0.6) !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 4px !important;
}
.tab-nav button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    color: #6b6b80 !important;
    border: none !important;
    transition: all 0.3s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, rgba(0,229,160,0.15), rgba(124,92,252,0.15)) !important;
    color: #e8e8ef !important;
    border: 1px solid rgba(0,229,160,0.3) !important;
}

/* DataFrames */
.dataframe {
    border: 1px solid #1e1e2e !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
table { background: #12121a !important; }
th {
    background: linear-gradient(135deg, rgba(124,92,252,0.1), rgba(0,229,160,0.05)) !important;
    color: #b0b0c8 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75em !important;
    letter-spacing: 0.08em !important;
    padding: 14px 16px !important;
    border-bottom: 1px solid #1e1e2e !important;
}
td {
    color: #d0d0e0 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(30,30,46,0.5) !important;
    font-size: 0.9em !important;
}
tr:hover td { background: rgba(0,229,160,0.03) !important; }

/* Buttons */
.primary {
    background: linear-gradient(135deg, #00e5a0 0%, #00c48c 100%) !important;
    color: #0a0a0f !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
}
.primary:hover {
    box-shadow: 0 0 30px rgba(0,229,160,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Dropdowns */
.wrap { border: 1px solid #1e1e2e !important; border-radius: 12px !important; background: #12121a !important; }
select, input { color: #e8e8ef !important; background: #12121a !important; }

/* Plots */
.plot-container { border: 1px solid #1e1e2e !important; border-radius: 16px !important; overflow: hidden !important; }

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #12121a, #161625);
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.stat-value {
    font-size: 2em;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5a0, #7c5cfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { color: #6b6b80; font-size: 0.85em; margin-top: 4px; }

/* Footer */
.footer-text {
    text-align: center;
    color: #3a3a50;
    font-size: 0.8em;
    padding: 24px;
    border-top: 1px solid #1e1e2e;
    margin-top: 32px;
}
"""

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(18,18,26,0)",
    plot_bgcolor="rgba(18,18,26,0.3)",
    font=dict(family="Space Grotesk, sans-serif", color="#b0b0c8", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9999b0")),
    margin=dict(t=60, b=40, l=50, r=20),
)


def load_data():
    if SAMPLE_DATA.exists():
        with open(SAMPLE_DATA, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def create_leaderboard_df(data):
    df = pd.DataFrame(data)
    df = df.sort_values("turkeval_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "#"
    df = df.rename(columns={
        "model_name": "Model", "developer": "Developer", "parameters": "Params",
        "turkeval_score": "TurkEval™", "truthfulqa_tr": "TruthfulQA",
        "mmlu_tr": "MMLU", "hallucination_tr": "Halluc.", "bias_tr": "Bias",
    })
    return df[["Model", "Developer", "Params", "TurkEval™", "TruthfulQA", "MMLU", "Halluc.", "Bias"]]


def make_ranking_chart(data):
    sd = sorted(data, key=lambda x: x["turkeval_score"], reverse=True)
    names = [d["model_name"] for d in sd]
    scores = [d["turkeval_score"] for d in sd]
    
    colors = []
    for s in scores:
        if s >= 85: colors.append("#00e5a0")
        elif s >= 70: colors.append("#7c5cfc")
        elif s >= 60: colors.append("#ffd93d")
        else: colors.append("#ff6b6b")

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h", marker=dict(
            color=colors, line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"<b>{s:.1f}</b>" for s in scores], textposition="outside",
        textfont=dict(color="#e8e8ef", size=13, family="Space Grotesk"),
        hovertemplate="<b>%{y}</b><br>TurkEval™: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="<b>⚡ TurkEval™ Score Rankings</b>", x=0.02),
        height=380, yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, max(scores) * 1.15], title=""),
    )
    return fig


def make_radar(data):
    fig = go.Figure()
    cats = BENCHMARKS
    labels = ["TruthfulQA", "MMLU", "Hallucination", "Bias"]

    for i, m in enumerate(sorted(data, key=lambda x: x["turkeval_score"], reverse=True)[:5]):
        vals = [m.get(c, 0) for c in cats] + [m.get(cats[0], 0)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]], fill="toself",
            name=m["model_name"], line_color=CHART_COLORS[i],
            fillcolor=f"rgba({int(CHART_COLORS[i][1:3],16)},{int(CHART_COLORS[i][3:5],16)},{int(CHART_COLORS[i][5:7],16)},0.08)",
            line_width=2.5,
        ))

    fig.update_layout(
        **PLOTLY_BASE,
        polar=dict(
            bgcolor="rgba(18,18,26,0.3)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(30,30,46,0.6)", linecolor="#1e1e2e"),
            angularaxis=dict(gridcolor="rgba(30,30,46,0.4)", linecolor="#1e1e2e"),
        ),
        title=dict(text="<b>🕸️ Multi-Dimensional Comparison</b>", x=0.02),
        height=500, showlegend=True,
    )
    return fig


def make_grouped_bar(data):
    rows = []
    for m in data:
        for b in BENCHMARKS:
            rows.append({"Model": m["model_name"], "Benchmark": BENCH_LABELS.get(b, b), "Score": m.get(b, 0)})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Model", y="Score", color="Benchmark", barmode="group",
                 color_discrete_sequence=CHART_COLORS[:4], height=450)
    fig.update_layout(**PLOTLY_BASE,
                      title=dict(text="<b>📊 Per-Benchmark Breakdown</b>", x=0.02),
                      xaxis_tickangle=-25, bargap=0.2, bargroupgap=0.05)
    fig.update_traces(marker_cornerradius=4)
    return fig


def make_heatmap(data):
    models = [d["model_name"] for d in sorted(data, key=lambda x: x["turkeval_score"], reverse=True)]
    z = [[d.get(b, 0) for b in BENCHMARKS] for d in sorted(data, key=lambda x: x["turkeval_score"], reverse=True)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["TruthfulQA", "MMLU", "Hallucination", "Bias"], y=models,
        colorscale=[[0, "#1a0a2e"], [0.3, "#7c5cfc"], [0.6, "#00e5a0"], [1, "#ffd93d"]],
        text=[[f"{v:.1f}" for v in row] for row in z], texttemplate="%{text}",
        textfont={"size": 12, "color": "#e8e8ef"},
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_BASE,
                      title=dict(text="<b>🌡️ Performance Heatmap</b>", x=0.02),
                      height=400)
    return fig


def compare_models(m1, m2, data):
    d1 = next((d for d in data if d["model_name"] == m1), None)
    d2 = next((d for d in data if d["model_name"] == m2), None)
    if not d1 or not d2:
        return go.Figure(), "Model bulunamadı"

    labels = list(BENCH_LABELS.values())
    v1 = [d1.get(c, 0) for c in BENCHMARKS]
    v2 = [d2.get(c, 0) for c in BENCHMARKS]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=m1, x=labels, y=v1, marker_color="#00e5a0", marker_cornerradius=6))
    fig.add_trace(go.Bar(name=m2, x=labels, y=v2, marker_color="#7c5cfc", marker_cornerradius=6))
    fig.update_layout(**PLOTLY_BASE, barmode="group",
                      title=dict(text=f"<b>⚔️ {m1} vs {m2}</b>", x=0.02), height=400)

    diff = d1["turkeval_score"] - d2["turkeval_score"]
    winner = m1 if diff > 0 else m2
    md = f"### 🏆 Kazanan: **{winner}**\nFark: **{abs(diff):.1f}** puan\n\n| Benchmark | {m1} | {m2} | Δ |\n|---|---|---|---|\n"
    for i, c in enumerate(BENCHMARKS):
        s1, s2 = d1.get(c, 0), d2.get(c, 0)
        delta = s1 - s2
        icon = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
        md += f"| {BENCH_LABELS[c]} | {s1:.1f} | {s2:.1f} | {icon} {delta:+.1f} |\n"
    md += f"\n| **TurkEval™** | **{d1['turkeval_score']:.1f}** | **{d2['turkeval_score']:.1f}** | **{diff:+.1f}** |"
    return fig, md


def build_app():
    data = load_data()
    df = create_leaderboard_df(data)
    names = [d["model_name"] for d in sorted(data, key=lambda x: x["turkeval_score"], reverse=True)]
    top = data[0] if data else {}

    with gr.Blocks(css=CUSTOM_CSS, title="TurkishLLM-Eval") as app:

        # ── Hero ──
        gr.HTML(f"""
        <div class="hero-section">
            <div class="hero-title">🇹🇷 TurkishLLM-Eval</div>
            <div class="hero-subtitle">
                Türkçe LLM'ler için kapsamlı Halüsinasyon, Doğruluk ve Önyargı Benchmark Platformu
            </div>
            <div class="hero-badges">
                <span class="badge badge-emerald">GPT-4o + Claude Judge Ensemble</span>
                <span class="badge badge-violet">4 Benchmark Suite</span>
                <span class="badge badge-coral">80+ Test Case</span>
                <span class="badge badge-gold">TurkEval™ Composite Score</span>
            </div>
        </div>
        """)

        # ── Stats Row ──
        with gr.Row():
            gr.HTML(f'<div class="stat-card"><div class="stat-value">{len(data)}</div><div class="stat-label">Models Evaluated</div></div>')
            gr.HTML(f'<div class="stat-card"><div class="stat-value">4</div><div class="stat-label">Benchmark Suites</div></div>')
            gr.HTML(f'<div class="stat-card"><div class="stat-value">80+</div><div class="stat-label">Test Questions</div></div>')
            gr.HTML(f'<div class="stat-card"><div class="stat-value">2</div><div class="stat-label">LLM Judges</div></div>')

        # ── Tabs ──
        with gr.Tabs():
            with gr.Tab("🏆 Leaderboard"):
                gr.Dataframe(value=df, interactive=False, wrap=True)
                gr.Plot(value=make_ranking_chart(data))

            with gr.Tab("📊 Deep Analysis"):
                with gr.Row():
                    gr.Plot(value=make_radar(data))
                gr.Plot(value=make_grouped_bar(data))
                gr.Plot(value=make_heatmap(data))

            with gr.Tab("⚔️ Head-to-Head"):
                gr.Markdown("### İki modeli doğrudan karşılaştır")
                with gr.Row():
                    dd1 = gr.Dropdown(choices=names, label="Model 1", value=names[0] if names else None)
                    dd2 = gr.Dropdown(choices=names, label="Model 2", value=names[1] if len(names) > 1 else None)
                btn = gr.Button("⚡ Karşılaştır", variant="primary", size="lg")
                comp_plot = gr.Plot()
                comp_md = gr.Markdown()
                btn.click(fn=lambda a, b: compare_models(a, b, data), inputs=[dd1, dd2], outputs=[comp_plot, comp_md])

            with gr.Tab("📖 Methodology"):
                gr.Markdown("""
## TurkEval™ Composite Score

```
TurkEval™ = 0.30 × TruthfulQA-TR + 0.25 × MMLU-TR + 0.25 × Anti-Hallucination + 0.20 × Anti-Bias
```

| Grade | Score Range | Meaning |
|-------|------------|---------|
| **A+** | ≥ 90 | Exceptional — production-ready |
| **A** | ≥ 80 | Strong — reliable |
| **B** | ≥ 70 | Good — usable with supervision |
| **C** | ≥ 60 | Fair — significant limitations |
| **F** | < 50 | Failing — not recommended |

---

### Judge Pipeline

| Judge | Weight | Role |
|-------|--------|------|
| **GPT-4o** | 0.55 | Primary evaluator |
| **Claude 3.5 Sonnet** | 0.45 | Cross-validator |

Inter-judge agreement measured via **Cohen's κ** coefficient.

### Bias Taxonomy (Turkey-Specific)

| Category | Turkish | Focus |
|----------|---------|-------|
| Gender | Cinsiyet | Professional stereotypes, family roles |
| Ethnic | Etnik | Kurdish, Arab, Laz, Roma stereotypes |
| Sectarian | Mezhepsel | Sunni/Alevi dynamics |
| Regional | Bölgesel | East-West, urban-rural divide |
| Socioeconomic | Sosyoekonomik | Class & education bias |

---

**Source Code:** [github.com/saciducak/TurkishLLM-Eval](https://github.com/saciducak/TurkishLLM-Eval)  
**License:** Apache 2.0
                """)

        gr.HTML('<div class="footer-text">Built with 🤍 by Muhammed Sacid Ucak · Apache 2.0 · TurkishLLM-Eval v0.1.0</div>')

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_port=PORT, server_name="0.0.0.0", share=False)
