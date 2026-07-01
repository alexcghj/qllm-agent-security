"""
experiments/visualize.py — графики для статьи.

Главный график:  ASR vs уровень квантизации (ключевой результат)
Дополнительные:   malformed rate, разбивка по типам атак, latency

Запуск после run.py:
    python experiments/visualize.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
# числовая ось: примерные биты на вес (для оси X)
QUANT_BITS = {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q6_K": 6.5, "Q8_0": 8.0}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "results", "results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Нет {path}. Сначала запусти: python experiments/run.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered(data, key):
    """Возвращает значения метрики в порядке QUANT_ORDER."""
    by_q = {r["quant"]: r for r in data}
    xs, ys = [], []
    for q in QUANT_ORDER:
        if q in by_q:
            xs.append(q)
            ys.append(by_q[q].get(key, 0))
    return xs, ys


# ── главный график: ASR vs квантизация ────────────────────────────────────────

def fig_asr(data, save):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    quants, asr = _ordered(data, "asr_pct")
    x = [QUANT_BITS[q] for q in quants]

    ax.plot(x, asr, marker="o", markersize=9, linewidth=2.2,
            color="#C62828", label="Attack Success Rate")
    for xi, yi, q in zip(x, asr, quants):
        ax.annotate(f"{yi:.0f}%", (xi, yi),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Quantization level (approx. bits per weight)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Indirect Prompt Injection Success vs. Quantization\n"
                 "(higher = more vulnerable)")
    ax.set_xticks(x)
    ax.set_xticklabels(quants)
    ax.set_ylim(0, max(asr) * 1.25 + 5 if asr else 100)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # слева сильное сжатие (Q4), справа слабое (Q8)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig] {save}")


# ── malformed rate (методологическая честность) ───────────────────────────────

def fig_malformed(data, save):
    fig, ax = plt.subplots(figsize=(6.5, 4))

    quants, mal = _ordered(data, "malformed_rate_pct")
    x = [QUANT_BITS[q] for q in quants]

    ax.bar(range(len(quants)), mal, color="#6A1B9A",
           edgecolor="black", linewidth=0.6, width=0.5)
    for i, v in enumerate(mal):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=10)

    ax.set_xlabel("Quantization level")
    ax.set_ylabel("Malformed / tool-failure rate (%)")
    ax.set_title("Agentic Format Failures vs. Quantization\n"
                 "(controls for: did the model break JSON vs. resist?)")
    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels(quants)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig] {save}")


# ── ASR по типам атак ─────────────────────────────────────────────────────────

def fig_by_type(data, save):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    by_q = {r["quant"]: r for r in data}
    quants = [q for q in QUANT_ORDER if q in by_q]
    x = list(range(len(quants)))

    direct = [by_q[q].get("asr_direct_harm", 0) for q in quants]
    steal  = [by_q[q].get("asr_data_stealing", 0) for q in quants]

    w = 0.35
    ax.bar([i - w/2 for i in x], direct, w, label="Direct harm",
           color="#D84315", edgecolor="black", linewidth=0.6)
    ax.bar([i + w/2 for i in x], steal, w, label="Data stealing",
           color="#00838F", edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Quantization level")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR by Attack Type vs. Quantization")
    ax.set_xticks(x)
    ax.set_xticklabels(quants)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"[Fig] {save}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(here, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data = load()
    print("[Visualize] Генерируем графики...")

    fig_asr(data, os.path.join(fig_dir, "fig1_asr_vs_quant.pdf"))
    fig_malformed(data, os.path.join(fig_dir, "fig2_malformed.pdf"))
    fig_by_type(data, os.path.join(fig_dir, "fig3_by_attack_type.pdf"))

    print(f"\n[Done] Графики в {fig_dir}/")


if __name__ == "__main__":
    main()
