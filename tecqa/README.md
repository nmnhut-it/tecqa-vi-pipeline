# TECQA Pipeline

This folder contains the core pipeline for the Temporal Evidence Chain-based Question Answering (TECQA) system over Knowledge Graphs.

## 1. Environment Setup

Before running the code, ensure you have the required packages installed (e.g., `sentence-transformers`, `numpy`, `python-dotenv`, `requests`, etc.).

You must create a `.env` file in the `tecqa/` folder (you can copy from `.env.example`) and provide your API key:
```env
OPENROUTER_API_KEY=your_api_key_here
```

## 2. Data Structure Requirements

You need to place your Knowledge Graph and Question files inside the `tecqa/data/` directory.

### Folder Structure
Your workspace should look like this:

```text
tecqa/
├── data/
│   ├── kg/
│   │   └── full.txt                 # The full English Knowledge Graph (461K facts)
│   ├── questions/
│   │   ├── test_en.json             # English test questions
│   │   └── test_vi.json             # Vietnamese translated test questions
├── .env                             # API Keys
├── run_eval_parallel.py
├── smoke_test.py
└── ...
```

**Note on Embeddings:**
The first time you run the code, `sentence-transformers` (using `BAAI/bge-base-en-v1.5`) will automatically generate semantic embeddings for the entities and relations in the Knowledge Graph. These will be cached as `data/entity_embeddings_en.json` and `data/relation_embeddings_en.json` inside the `tecqa/data/` folder so subsequent runs will be significantly faster.

---

## 3. Running a Few Samples (Smoke Test)

If you just want to run a quick test on a few samples to see the step-by-step reasoning (Stage 1 to Stage 3 debug prints), use `smoke_test.py`.

```bash
cd tecqa
python smoke_test.py --n 3 --kg-path data/kg/full.txt --data-path data/questions/test_en.json
```

---

## 4. Running the Full Evaluation

To run parallel evaluation on the dataset and measure Hits@1, use `run_eval_parallel.py`. You can configure the number of samples (`--n`), workers, and the specific paths to your data.

### Evaluate English Dataset:
```bash
cd tecqa
python run_eval_parallel.py \
    --language en \
    --n 100 \
    --workers 16 \
    --kg-path data/kg/full.txt \
    --data-path data/questions/test_en.json
```
*Outputs are saved to `eval_results_{n}_en.json`.*

### Evaluate Vietnamese Dataset:
```bash
cd tecqa
python run_eval_parallel.py \
    --language vi \
    --n 100 \
    --workers 16 \
    --kg-path data/kg/full.txt \
    --data-path data/questions/test_vi.json
```
*Outputs are saved to `eval_results_{n}_vi.json`.*

---

## 5. Full Evaluation Harness (`tecqa/eval/`)

`run_eval_parallel.py` above answers one question: Hits@1 on N random questions.
The harness in `tecqa/eval/` runs the paper's whole Section 5 — ablations, the
K/N sweep, the backbone sweep, the Appendix D recalls, and the error-analysis
diagnostics — from a single command, and writes into `results/`, which is what
`scripts/make_tables.py` turns into the paper's numbers.

```bash
# Plan and cost estimate. Touches no API, needs no key.
python -m tecqa.eval.run_eval --dry-run --n 200

# Freeze a sample so every run and re-run scores the same questions.
python -m tecqa.eval.run_eval --freeze-sample --n 200 --seed 42

# The main Vietnamese run, capped at $5, six workers.
python -m tecqa.eval.run_eval --lang vi --sample multitq_n200_seed42 --max-usd 5 --parallel 6

# The paired English condition — same questions, same models, same K/N.
python -m tecqa.eval.run_eval --lang en --sample multitq_n200_seed42

# Ablations (paper Table 2) and sweeps (Figure 4).
python -m tecqa.eval.run_eval --ablation no_sg   --sample multitq_n200_seed42
python -m tecqa.eval.run_eval --k 10 --sample multitq_n200_seed42
```

**Interrupting is safe, and repeating is free.** Work is queued in Redis
(`localhost:6379`), but *disk is authoritative*: each answer is fsync'd to
`results/<run_id>.jsonl` before it is recorded in Redis. Turning Redis off, or
killing the process, loses nothing that was paid for. Re-running the same
command resumes; a question already answered is never sent again, which is also
what keeps adding new data cheap.

`--max-usd` is a hard ceiling, and the run additionally stops before the
OpenRouter account's own remaining credit so it can never fail mid-write.

Offline tests — no API key, no network, no cost:

```bash
python -m unittest discover -s tests
```

## 6. Benchmark Results

> **These numbers predate the switch to the paper's backbones.** `config.py` now
> uses Gemini-2.5-flash for Stage 1 and Qwen3-8B for Stage 3, as the paper
> specifies, so the table below is not comparable to current runs. It is kept
> because it is what motivated using a strong instruction-follower for Stage 1.
> Current numbers live in `results/` and are generated, never typed by hand.

*Latest run (100 samples) using `google/gemini-3.7-flash` (Stage 1) and `qwen/qwen3-30b-a3b` (Stage 3):*

| Language   | Samples | Hits@1 | Avg Latency / Question | Total Cost |
|------------|---------|--------|------------------------|------------|
| English    | 100     | 73.0%  | ~8.1s                  | *N/A*      |
| Vietnamese | 100     | 58.0%  | ~8.9s                  | *N/A*      |

*(Note: Token usage and exact cost per request are not currently logged in the `eval_results` JSON.)*

*Latest run (100 samples) using `deepseek/deepseek-v4-flash-0731` for BOTH extraction (Stage 1) and reasoning (Stage 3):*

| Language   | Samples | Hits@1 | Avg Latency / Question |
|------------|---------|--------|------------------------|
| English    | 100     | ~39.4% | ~200s+                 |
| Vietnamese | 100     | 68.0%  | 14.7s                  |

*(Note: The significant drop in English accuracy (to ~39%) is because the reasoning model `DeepSeek-V4-Flash` performs poorly on the strict Few-Shot Extraction prompts (Stage 1), leading to empty facts retrieved and ultimately `[]` predictions. We recommend using `google/gemini-3.7-flash` for extraction.)*

*Latest run (100 samples) using Hybrid Config (`google/gemini-3.7-flash` for Stage 1 Extraction, `deepseek/deepseek-v4-flash-0731` for Stage 3 Reasoning):*

| Language   | Samples | Hits@1 | Avg Latency / Question |
|------------|---------|--------|------------------------|
| English    | 100     | 78.0%  | 17.4s                  |
| Vietnamese | 100     | 55.0%  | 43.6s                  |

*(Note: Using Gemini for Stage 1 successfully restored extraction quality, allowing DeepSeek to reason over accurate facts. The English score perfectly recovered to 78.0%.)*
