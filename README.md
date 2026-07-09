[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21268938-blue)](https://doi.org/10.5281/zenodo.21268938)

# Quantization and the Security of Small LLM Agents

Systematic study of what determines the vulnerability of small (0.5–4B) LLM
agents to indirect prompt injection: **quantization, model size,
architecture, and attack style**.

**Author:** Aleksandr Kuleshov (Peter the Great St. Petersburg Polytechnic
University) · ORCID 0009-0001-7197-8001

---

## TL;DR — key findings

Across **14 model configurations** and **>14,000 agent rollouts** (10 seeds),
cross-checked on the external **InjecAgent** benchmark:

1. **Quantization does not matter (practically).** From Q8 down to Q4, the
   effect on injection resistance is negligible for every model
   (Cohen's *h* < 0.16). The one statistically significant case is a
   large-sample detection of a 7-point gap that vanishes under greedy
   decoding — not a meaningful difference.
2. **A standard defense can backfire.** Delimiter-based *spotlighting*
   reduces attack success on Llama-3.2 by 20–25 points but **raises** it on
   Qwen2.5 by 14–18 points (*h* up to 0.53). Prompt-level defenses do not
   transfer between architectures.
3. **Sophistication backfires too.** Plainly-phrased injections dominate
   (hidden-in-markup 85%), while "sophisticated" code/config disguises are
   the *weakest* (code-block 16%) — small models don't parse the ornate
   framing as a command.
4. **Capability floor.** Below ~1B parameters, apparent "vulnerability" is an
   artifact of incapacity, not a defeated safeguard. A four-outcome scorer +
   benign-baseline usability criterion separates the two.

---

## Repository layout

```
qllm-agent-security/
├── src/
│   ├── ollama_client.py       # Ollama REST wrapper (temperature, seed, num_predict)
│   ├── agent.py               # single-step tool-use agent; 4-outcome classifier;
│   │                          #   weak + hardened (spotlighting) defenses; strict_format
│   ├── attacks.py             # scenario generator; 8 injection styles (5 plain + 3 elaborate)
│   ├── metrics.py             # ASR, Wilson CI, bootstrap, baseline completion
│   ├── stats.py               # z-test, chi-square, Cohen's h, Holm, TOST, power analysis
│   └── injecagent_adapter.py  # converts InjecAgent test cases into our scenario format
├── experiments/
│   ├── preflight.py           # checks all models respond before a long run
│   ├── run.py                 # MAIN experiment: 14 configs × 96 scenarios × 10 seeds
│   ├── temp_sweep.py          # H1 robustness across temperatures {0.0, 0.7, 1.0}
│   ├── run_injecagent.py      # external-benchmark cross-check (InjecAgent subset)
│   ├── run_defense.py         # weak vs hardened spotlighting comparison
│   └── visualize.py           # generates the 6 figures
├── paper/
│   ├── main.tex               # the paper (LaTeX)
│   └── figures/               # fig1–fig6 (PDF)
├── results/                   # generated: results.json, analysis.json, raw_runs.json, *.csv
└── README.md
```

---

## Requirements

- **Python 3.10+**
- **Ollama** (https://ollama.com) installed and running
- Python packages: `pip install requests tqdm numpy matplotlib`
- CPU is sufficient — no GPU required. (Runs were done on an 8-core laptop.)

### Models used (pull with Ollama)

```bash
ollama pull qwen2.5:0.5b-instruct-q4_K_M
ollama pull qwen2.5:0.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q5_K_M
ollama pull qwen2.5:1.5b-instruct-q6_K
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q8_0
ollama pull llama3.2:1b-instruct-q4_K_M
ollama pull llama3.2:1b-instruct-q5_K_M
ollama pull llama3.2:1b-instruct-q6_K
ollama pull llama3.2:1b-instruct-q8_0
ollama pull gemma2:2b-instruct-q4_K_M
ollama pull phi3:mini
```

---

## Reproducing the results

```bash
# 0. sanity check — all models respond and parse
python experiments/preflight.py

# 1. main experiment (longest; run overnight)
python experiments/run.py

# 2. external-benchmark cross-check
#    first: git clone https://github.com/uiuc-kang-lab/InjecAgent.git  (next to this repo)
python experiments/run_injecagent.py

# 3. temperature robustness
python experiments/temp_sweep.py

# 4. defense comparison (weak vs hardened spotlighting)
python experiments/run_defense.py

# 5. figures
python experiments/visualize.py
```

Each script writes JSON to `results/`. `run.py` also prints the H1–H4
analysis (quantization, style, size, architecture) to the console.

---

## Method in brief

- **Threat model.** Attacker controls only tool-returned content (email
  body, file contents, search text) and hides an instruction in it. They do
  not control the system prompt, the user instruction, or the model weights,
  and do not know the quantization level.
- **Harness.** A single tool-use turn: the agent is given a benign task, its
  first tool call is prefilled, and the (injected) tool result is returned;
  we observe the next action. Prefilling isolates the injection decision from
  multi-step confounds (see Limitations in the paper).
- **Scoring.** Every rollout is one of four outcomes — attack success,
  defense hold, refused, malformed. ASR is computed only over well-formed
  rollouts; the malformed rate is reported separately so that an incoherent
  model is not mistaken for a secure one.
- **Statistics.** 10 seeds; Wilson CIs; two-proportion z-test and chi-square;
  Cohen's *h* for effect size; Holm correction for the style comparisons;
  TOST + power analysis for the (negative) quantization result.

---

## Data / artifacts

`results/` contains, after a run:
- `results.json` — per-configuration summary (ASR, CIs, baseline, malformed, per-style)
- `analysis.json` — H1–H4 hypothesis tests
- `raw_runs.json` — every individual rollout (outcome, latency, raw response snippet)
- `results.csv` — flat table
- `temp_sweep.json`, `injecagent_results.json`, `defense_results.json` — focused studies

---

## Paper

The write-up is in `paper/main.tex` (compiles with pdflatex; also
Overleaf-ready). It reports all of the above with figures and full
statistics.

---

## License & citation

Code released for reproducibility. If you use it, please cite the paper
(preprint link to be added on arXiv release).

Reproducibility artifacts are archived on Zenodo:

DOI: https://doi.org/10.5281/zenodo.21268938

---

## Acknowledgements

Built on Ollama (https://ollama.com), the InjecAgent benchmark
(https://github.com/uiuc-kang-lab/InjecAgent), and the open-weight Qwen2.5,
Llama-3.2, Gemma-2, and Phi-3 model families.
