"""
experiments/visualize.py — графики для статьи (с доверительными интервалами).

Графики:
  fig1  ASR vs квантизация с error bars (главный, ключевой результат)
  fig2  Malformed rate vs квантизация (методологический контроль)
  fig3  ASR по типам атак (direct_harm vs data_stealing)
  fig4  ASR по стилям инъекций (какие инъекции опаснее)
  fig5  ASR по моделям (если несколько семейств)

Запуск после run.py:
    python experiments/visualize.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
QUANT_BITS = {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q6_K": 6.5, "Q8_0": 8.0}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

FAMILY_COLORS = {
    "qwen2.5-1.5b": "#C62828",
    "llama3.2-1b":  "#1565C0",
    "phi3-mini":    "#2E7D32",
}


def load():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "results", "results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет {path}. Сначала: python experiments/run.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── fig1: ASR vs квантизация с error bars (по семействам) ─────────────────────

def fig1_asr_ci(data, save):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    families = sorted(set(r["family"] for r in data))

    for fam in families:
        rows = {r["quant"]: r for r in data if r["family"] == fam}
        quants = [q for q in QUANT_ORDER if q in rows]
        x = [QUANT_BITS[q] for q in quants]
        asr = [rows[q]["asr_pct"] for q in quants]
        # error bars из доверительных интервалов
        lo = [rows[q]["asr_pct"] - rows[q].get("asr_ci_low", rows[q]["asr_pct"]) for q in quants]
        hi = [rows[q].get("asr_ci_high", rows[q]["asr_pct"]) - rows[q]["asr_pct"] for q in quants]

        color = FAMILY_COLORS.get(fam, "#555555")
        ax.errorbar(x, asr, yerr=[lo, hi], marker="o", markersize=8,
                    linewidth=2, capsize=5, color=color, label=fam)

    ax.set_xlabel("Quantization level (approx. bits per weight)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Indirect Prompt Injection Success vs. Quantization\n"
                 "(error bars: 95% Wilson CI; higher = more vulnerable)")
    ax.set_xticks(list(QUANT_BITS.values()))
    ax.set_xticklabels(QUANT_ORDER)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig1] {save}")


# ── fig2: malformed rate ──────────────────────────────────────────────────────

def fig2_malformed(data, save):
    fig, ax = plt.subplots(figsize=(7, 4))
    families = sorted(set(r["family"] for r in data))
    n_fam = len(families)
    width = 0.8 / n_fam

    for i, fam in enumerate(families):
        rows = {r["quant"]: r for r in data if r["family"] == fam}
        quants = [q for q in QUANT_ORDER if q in rows]
        mal = [rows[q]["malformed_rate_pct"] for q in quants]
        x = np.arange(len(quants)) + i * width
        ax.bar(x, mal, width, label=fam,
               color=FAMILY_COLORS.get(fam, "#555"), edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Quantization level")
    ax.set_ylabel("Malformed / tool-failure rate (%)")
    ax.set_title("Agentic Format Failures vs. Quantization\n"
                 "(controls: broke JSON vs. genuinely resisted)")
    ax.set_xticks(np.arange(len(QUANT_ORDER)) + width * (n_fam-1) / 2)
    ax.set_xticklabels(QUANT_ORDER)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig2] {save}")


# ── fig3: ASR по типам атак ───────────────────────────────────────────────────

def fig3_by_type(data, save):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # усредняем по семействам для каждого quant
    quants = QUANT_ORDER
    direct, steal = [], []
    for q in quants:
        rows = [r for r in data if r["quant"] == q]
        d = [r.get("asr_direct_harm", 0) for r in rows if "asr_direct_harm" in r]
        s = [r.get("asr_data_stealing", 0) for r in rows if "asr_data_stealing" in r]
        direct.append(np.mean(d) if d else 0)
        steal.append(np.mean(s) if s else 0)

    x = np.arange(len(quants))
    w = 0.35
    ax.bar(x - w/2, direct, w, label="Direct harm",
           color="#D84315", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, steal, w, label="Data stealing",
           color="#00838F", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Quantization level")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR by Attack Type vs. Quantization")
    ax.set_xticks(x)
    ax.set_xticklabels(quants)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig3] {save}")


# ── fig4: ASR по стилям инъекций ──────────────────────────────────────────────

def fig4_by_style(data, save):
    styles = ["imperative", "authority", "hidden", "roleplay", "urgency"]
    # усредняем по всем конфигурациям
    style_asr = {}
    for st in styles:
        key = f"asr_style_{st}"
        vals = [r[key] for r in data if key in r]
        if vals:
            style_asr[st] = np.mean(vals)

    if not style_asr:
        print("[Fig4] пропущен (нет данных по стилям)")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    sts = list(style_asr.keys())
    vals = [style_asr[s] for s in sts]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sts)))
    bars = ax.bar(sts, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}%",
                ha="center", fontsize=9)

    ax.set_xlabel("Injection style")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR by Injection Style (averaged across quant levels)")
    ax.set_ylim(0, max(vals) * 1.2 + 5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig4] {save}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(here, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data = load()
    print("[Visualize] Генерируем графики...")

    fig1_asr_ci(data, os.path.join(fig_dir, "fig1_asr_vs_quant.pdf"))
    fig2_malformed(data, os.path.join(fig_dir, "fig2_malformed.pdf"))
    fig3_by_type(data, os.path.join(fig_dir, "fig3_by_attack_type.pdf"))
    fig4_by_style(data, os.path.join(fig_dir, "fig4_by_style.pdf"))

    print(f"\n[Done] Графики в {fig_dir}/")


if __name__ == "__main__":
    main()
