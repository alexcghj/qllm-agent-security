# What Makes Small LLM Agents Vulnerable to Indirect Prompt Injection?

**A systematic study of injection style vs. quantization level in small tool-using agents.**

This project measures what actually drives an agent's susceptibility to **indirect prompt
injection**: the *style* of the injected instruction, the *quantization level* of the model,
the model *size*, and the model *architecture*.

> ⚠️ Research / educational project. All attacks run against **local** models in a sandboxed
> setting to study defensive robustness. Fictional targets only; nothing harmful is executed.

---

## Key finding (preview)

Injection **style** dominates; quantization does **not**.

- Instructions hidden in markup (HTML comments) succeed almost always, regardless of quantization.
- Authority-framed instructions are resisted far more often.
- Compressing a model from Q8 down to Q4 does **not** systematically change its resistance.

## Hypotheses tested

```
H1  Quantization (Q4–Q8) does NOT systematically affect injection resistance.  (negative control)
H2  Injection style is the dominant factor in attack success.                  (main result)
H3  Resistance depends on model size.                                          (Qwen 0.5B/1.5B/3B)
H4  Different model families have different weaknesses.                        (Qwen vs Llama vs …)
```

## Method

| Component | Choice |
|---|---|
| Models | Qwen2.5 (0.5B / 1.5B / 3B), Llama-3.2-1B (+ optional Gemma2-2B, Phi-3-mini) |
| Quantization | Q4_K_M, Q5_K_M, Q6_K, Q8_0 (GGUF via Ollama) |
| Attack | Indirect prompt injection in a simulated tool-use loop |
| Scenarios | 60 attack scenarios (2 types × 5 injection styles × 6 domains) + 5 baselines |
| Injection styles | imperative, authority, hidden (markup), roleplay, urgency |
| Sampling | temperature = 0.7, seeds = {42, 123, 456} (realistic; reproducibility over seeds) |
| Statistics | Wilson & bootstrap CI, two-proportion z-test, χ², Cohen's *h* effect size, Holm correction |
| Metrics | Attack Success Rate (ASR), malformed rate, baseline completion, latency |
| Runtime | Ollama on CPU (no GPU required) |

A four-outcome classifier (attack_success / defense_hold / refused / malformed) separates genuine
resistance from technical format failures, so a broken JSON response is never mistaken for a
successful defense.

---

## Installation

Requires [Ollama](https://ollama.com) installed and running.

```bash
# Pull models (Qwen size ladder + Llama)
ollama pull qwen2.5:0.5b-instruct-q4_K_M
ollama pull qwen2.5:0.5b-instruct-q5_K_M
ollama pull qwen2.5:0.5b-instruct-q6_K
ollama pull qwen2.5:0.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q5_K_M
ollama pull qwen2.5:1.5b-instruct-q6_K
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q5_K_M
ollama pull qwen2.5:3b-instruct-q6_K
ollama pull qwen2.5:3b-instruct-q8_0
ollama pull llama3.2:1b-instruct-q4_K_M
ollama pull llama3.2:1b-instruct-q5_K_M
ollama pull llama3.2:1b-instruct-q6_K
ollama pull llama3.2:1b-instruct-q8_0

pip install -r requirements.txt
```

## Run

```bash
# 1. Preflight: verify every model responds (~2 min) BEFORE the long run
python experiments/preflight.py

# 2. Full experiment (long — run overnight; keep laptop awake & plugged in)
python experiments/run.py

# 3. Figures
python experiments/visualize.py
```

Outputs: `results/results.csv`, `results/analysis.json` (hypothesis tests),
`results/raw_runs.json`, and `results/figures/`.

---

## Repository structure

```
qllm-agent-security/
├── src/
│   ├── ollama_client.py   # Ollama REST API wrapper (temp, seed)
│   ├── agent.py           # tool-using agent + 4-outcome classifier
│   ├── attacks.py         # scenario generator (types × styles × domains)
│   ├── metrics.py         # ASR, malformed, baseline completion + CIs
│   └── stats.py           # Wilson/bootstrap CI, z-test, χ², Cohen's h, Holm
├── experiments/
│   ├── preflight.py       # readiness check before the long run
│   ├── run.py             # full experiment: matrix, seeds, hypothesis analysis
│   └── visualize.py       # figures: styles / heatmap / quant control / size
└── results/
    ├── results.csv
    ├── analysis.json      # H1–H4 statistical results
    └── figures/
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

A paper is in preparation; citation details will be added here.
