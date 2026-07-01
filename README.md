# Quantization vs. Agent Security

**Does aggressive quantization weaken a small LLM agent's resistance to indirect prompt injection?**

This project systematically measures how the quantization level (Q4 → Q5 → Q6 → Q8) of small
language models affects their vulnerability to **indirect prompt injection** in tool-using
(agentic) scenarios.

> ⚠️ Research / educational project. All attacks are run against **local** models in a
> sandboxed setting to study defensive robustness. Do not use against systems you do not own.

---

## Motivation

Small LLMs (1–3B) are the models people actually quantize and deploy on edge devices, phones,
and local agents. Their safety "margin" is already thin compared to frontier models. **What
happens to that thin margin under aggressive quantization?** Prior work studies prompt injection
and quantization separately, but the intersection — *quantization × indirect injection × small
agents* — is largely unexplored.

## Research question

```
Take ONE model (Qwen2.5-1.5B-Instruct)
   → quantize it at 4 levels: Q4_K_M / Q5_K_M / Q6_K / Q8_0
   → attack each version with the SAME indirect prompt injection
   → measure: does a more compressed version get compromised more often?
```

If the Q4 version lets more attacks through than Q8, that is direct evidence that aggressive
quantization degrades safety alignment in agentic settings — a real risk for edge deployment.

## Method (at a glance)

| Component | Choice |
|---|---|
| Models | Qwen2.5-1.5B-Instruct (+ Llama-3.2-1B as a second model) |
| Quantization | Q4_K_M, Q5_K_M, Q6_K, Q8_0 (GGUF via Ollama) |
| Attack | Indirect prompt injection in a simulated tool-use loop |
| Scenarios | 60 attack scenarios (2 attack types × 5 injection styles × 6 domains) + 5 benign baselines |
| Injection styles | imperative, authority, hidden (markup), roleplay, urgency |
| Repeats | 3 per scenario (averages sampling noise) |
| Statistics | 95% Wilson confidence intervals, bootstrap CI, two-proportion z-test, χ² |
| Metrics | Attack Success Rate (ASR), malformed rate, baseline completion, latency |
| Runtime | Ollama on CPU (no GPU required) |

The four-outcome classifier (attack_success / defense_hold / refused / malformed)
separates genuine resistance from technical format failures, so a broken JSON
response is never mistaken for a successful defense.

## Status

🚧 Work in progress. This repository accompanies an in-progress research paper.

---

## Installation

Requires [Ollama](https://ollama.com) installed and running.

```bash
# 1. Pull the four quantization levels
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q5_K_M
ollama pull qwen2.5:1.5b-instruct-q6_K
ollama pull qwen2.5:1.5b-instruct-q8_0

# 2. Install Python dependencies
pip install -r requirements.txt
```

## Reproduce

```bash
# Run the full experiment (all models × all quant levels × all scenarios)
python experiments/run.py

# Generate figures
python experiments/visualize.py
```

Results are written to `results/results.csv` and figures to `results/figures/`.

---

## Repository structure

```
qllm-agent-security/
├── src/
│   ├── ollama_client.py   # thin wrapper over the Ollama REST API
│   ├── agent.py           # simulated tool-using agent + 4-outcome classifier
│   ├── attacks.py         # scenario generator (attack types × injection styles)
│   ├── metrics.py         # ASR, malformed rate, baseline completion + CIs
│   └── stats.py           # Wilson CI, bootstrap, z-test, χ² (pure numpy)
├── experiments/
│   ├── run.py             # main driver: repeats, multi-model, significance tests
│   └── visualize.py       # plots with error bars: ASR / malformed / by type / by style
└── results/
    ├── results.csv
    ├── stats.json         # significance tests (Q4 vs Q8 per model)
    └── figures/
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this work is useful, a paper is in preparation; citation details will be added here.
