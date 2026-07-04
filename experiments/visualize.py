"""
experiments/visualize.py — графики для статьи (финальная версия).

Главный фокус СМЕЩЁН на стили инъекций (это ключевой результат).

Графики:
  fig1  ASR по стилям инъекций с CI — ГЛАВНЫЙ (H2)
  fig2  Heatmap: стиль × модель — показывает универсальность паттерна
  fig3  ASR vs квантизация — негативный контроль (H1)
  fig4  ASR vs размер модели (H3)
  fig5  Malformed rate по моделям (методологический контроль)

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
STYLE_ORDER = ["hidden", "imperative", "roleplay", "urgency", "authority"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ARCH_COLORS = {"qwen": "#C62828", "llama": "#1565C0",
               "gemma": "#2E7D32", "phi": "#F57F17"}


def load_summary():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "results", "results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет {path}. Сначала: python experiments/run.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "results", "raw_runs.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── fig1: ASR по стилям (ГЛАВНЫЙ) ─────────────────────────────────────────────

def fig1_styles(raw, save):
    """ASR по каждому стилю инъекции, усреднённо по всем моделям, с CI."""
    from src.stats import wilson_ci
    from collections import defaultdict

    groups = defaultdict(lambda: {"succ": 0, "n": 0})
    for r in raw:
        if r["attack_type"] == "baseline":
            continue
        if r["outcome"] in ("attack_success", "defense_hold", "refused"):
            st = r["injection_style"]
            groups[st]["n"] += 1
            if r["outcome"] == "attack_success":
                groups[st]["succ"] += 1

    styles = [s for s in STYLE_ORDER if s in groups]
    asr = [100*groups[s]["succ"]/groups[s]["n"] for s in styles]
    cis = [wilson_ci(groups[s]["succ"], groups[s]["n"]) for s in styles]
    err_lo = [asr[i]-cis[i][0] for i in range(len(styles))]
    err_hi = [cis[i][1]-asr[i] for i in range(len(styles))]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(styles)))
    bars = ax.bar(range(len(styles)), asr, yerr=[err_lo, err_hi],
                  capsize=5, color=colors, edgecolor="black", linewidth=0.7)
    for i, v in enumerate(asr):
        ax.text(i, v + max(err_hi)*0.3 + 2, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Injection style")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate by Injection Style\n"
                 "(averaged across all models & quantizations; 95% Wilson CI)")
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels(styles)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig1] {save}")


# ── fig2: heatmap стиль × модель ──────────────────────────────────────────────

def fig2_heatmap(raw, save):
    from collections import defaultdict

    # ASR[model][style]
    data = defaultdict(lambda: defaultdict(lambda: {"succ": 0, "n": 0}))
    models_set = set()
    for r in raw:
        if r["attack_type"] == "baseline":
            continue
        if r["outcome"] in ("attack_success", "defense_hold", "refused"):
            m = r["family"]
            models_set.add(m)
            st = r["injection_style"]
            data[m][st]["n"] += 1
            if r["outcome"] == "attack_success":
                data[m][st]["succ"] += 1

    models = sorted(models_set)
    styles = [s for s in STYLE_ORDER]
    matrix = np.zeros((len(models), len(styles)))
    for i, m in enumerate(models):
        for j, st in enumerate(styles):
            g = data[m][st]
            matrix[i, j] = 100*g["succ"]/g["n"] if g["n"] else np.nan

    fig, ax = plt.subplots(figsize=(8, max(4, len(models)*0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels(styles)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(styles)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                        fontsize=9, color="black")

    ax.set_title("Attack Success Rate (%): Model × Injection Style")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ASR (%)")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig2] {save}")


# ── fig3: ASR vs квантизация (негативный контроль) ────────────────────────────

def fig3_quant(summary, save):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    families = sorted(set(r["family"] for r in summary))

    for fam in families:
        rows = {r["quant"]: r for r in summary if r["family"] == fam}
        quants = [q for q in QUANT_ORDER if q in rows]
        if not quants:
            continue
        x = [QUANT_BITS[q] for q in quants]
        asr = [rows[q]["asr_mean"] for q in quants]
        lo = [rows[q]["asr_mean"] - rows[q].get("asr_ci_low", rows[q]["asr_mean"]) for q in quants]
        hi = [rows[q].get("asr_ci_high", rows[q]["asr_mean"]) - rows[q]["asr_mean"] for q in quants]
        arch = rows[quants[0]].get("arch", "qwen")
        ax.errorbar(x, asr, yerr=[lo, hi], marker="o", markersize=6,
                    linewidth=1.5, capsize=4,
                    color=ARCH_COLORS.get(arch, "#555"), label=fam, alpha=0.8)

    ax.set_xlabel("Quantization level (approx. bits per weight)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR vs. Quantization — Negative Control\n"
                 "(no systematic trend; error bars: 95% CI over seeds)")
    ax.set_xticks(list(QUANT_BITS.values()))
    ax.set_xticklabels(QUANT_ORDER)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig3] {save}")


# ── fig4: ASR vs размер модели (H3) ───────────────────────────────────────────

def fig4_size(summary, save):
    # только Qwen (единое семейство, чистое сравнение размеров)
    qwen = [r for r in summary if r.get("arch") == "qwen"]
    if not qwen:
        print("[Fig4] пропущен (нет Qwen size-данных)")
        return

    sizes = sorted(set(r["size_b"] for r in qwen))
    # усредняем по квантам для каждого размера
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, stds = [], []
    for s in sizes:
        vals = [r["asr_mean"] for r in qwen if r["size_b"] == s]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    ax.errorbar(sizes, means, yerr=stds, marker="s", markersize=9,
                linewidth=2, capsize=5, color="#C62828")
    for x, y in zip(sizes, means):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Model size (billions of parameters)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR vs. Model Size (Qwen2.5 family)\n"
                 "(averaged across quantizations)")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig4] {save}")


# ── fig5: malformed rate ──────────────────────────────────────────────────────

def fig5_malformed(summary, save):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['family']}\n{r['quant']}" for r in summary]
    mal = [r["malformed_rate_pct"] for r in summary]
    colors = [ARCH_COLORS.get(r.get("arch","qwen"), "#555") for r in summary]

    ax.bar(range(len(labels)), mal, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Malformed / tool-failure rate (%)")
    ax.set_title("Format Failures Across Configurations\n"
                 "(controls: broke JSON vs. genuinely resisted)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig5] {save}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(here, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    summary = load_summary()
    raw = load_raw()
    print("[Visualize] Генерируем графики...")

    fig1_styles(raw, os.path.join(fig_dir, "fig1_styles.pdf"))
    fig2_heatmap(raw, os.path.join(fig_dir, "fig2_heatmap.pdf"))
    fig3_quant(summary, os.path.join(fig_dir, "fig3_quant_control.pdf"))
    fig4_size(summary, os.path.join(fig_dir, "fig4_size.pdf"))
    fig5_malformed(summary, os.path.join(fig_dir, "fig5_malformed.pdf"))

    print(f"\n[Done] Графики в {fig_dir}/")


if __name__ == "__main__":
    main()
