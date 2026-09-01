"""
Multi-sample smoke test + debug for the original English TECQA pipeline.
Usage: python smoke_test.py [N]   (default N=3)

Each question prints stage-by-stage debug info so failures are easy to trace:
  Stage 1 -> grounded entities, relation, subgraph size
  Stage 2 -> chain size, nearest anchor
  Stage 3 -> predicted answer vs gold
"""
import json
import os
import sys
from pathlib import Path

# Unbuffered line output so progress streams live to terminal and logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Prevent tokenizers deadlock
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TECQA_ROOT = Path(__file__).parent
if str(TECQA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(TECQA_ROOT.parent))

from tecqa.data.kg_multitq import MultiTQGraph
from tecqa.pipeline import TECQA
from tecqa.stages.stage2_chain import extract_explicit_anchors

import argparse
parser = argparse.ArgumentParser(description="Smoke test for TECQA pipeline.")
parser.add_argument("--n", type=int, default=3, help="Number of samples to run")
parser.add_argument("--kg-path", type=str, default="data/kg/full.txt", help="Path to KG full.txt")
parser.add_argument("--data-path", type=str, default="data/questions/test_en.json", help="Path to questions JSON")
args = parser.parse_args()

N_SAMPLES = args.n
DATA_PATH = Path(args.data_path)
KG_PATH = Path(args.kg_path)

print(f"Loading full MultiTQ graph from {KG_PATH} (461K facts)...")

graph = MultiTQGraph(kg_path=KG_PATH).load()
print(f"  {len(graph.facts)} facts, {len(graph.entities)} entities, {len(graph.relations)} relations\n")

tecqa = TECQA(graph)
data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

hits = 0
for i, q in enumerate(data[:N_SAMPLES]):
    print(f"\n{'='*65}")
    print(f"[{i+1}/{N_SAMPLES}] {q['question']}")
    print(f"  type={q.get('question_type', 'simple')}  label={q.get('qlabel', 'Single')}  answer_type={q.get('answer_type', 'entity')}")
    print(f"  Gold: {q['answers']}")

    try:
        # --- Stage 1 debug ---
        main_entity, f_q, grounded = tecqa.structure_guided_subgraph(q["question"])
        explicit = extract_explicit_anchors(q["question"])
        print(f"\n  [Stage 1]")
        print(f"    grounded entities : {sorted(grounded)}")
        print(f"    main_entity       : {main_entity}")
        print(f"    subgraph size     : {len(f_q)} facts")

        if not f_q:
            print("    ⚠ Empty subgraph — skipping Stage 2 & 3")
            print(f"\n  [MISS]  (empty subgraph)")
            continue

        # --- Stage 2 debug ---
        chain = tecqa.temporal_evidence_chain(q["question"], f_q, grounded_entities=grounded)
        print(f"\n  [Stage 2]")
        print(f"    explicit anchors  : {explicit}")
        print(f"    chain size        : {len(chain)} facts")
        if chain:
            print(f"    closest fact      : {graph.fact_text(chain[0])}")
            print(f"    farthest fact     : {graph.fact_text(chain[-1])}")

        # --- Stage 3 debug ---
        pred = tecqa.reason(q["question"], chain,
                            answer_type=q.get("answer_type", "entity"),
                            qlabel=q.get("qlabel", "Single"))
        print(f"\n  [Stage 3]")
        print(f"    predicted : {pred}")

        # --- Result ---
        gold_norm = {g.strip().lower().replace("_", " ") for g in q["answers"]}
        hit = any(p.strip().lower().replace("_", " ") in gold_norm for p in pred)
        hits += int(hit)
        print(f"\n  [{'HIT ✓' if hit else 'MISS ✗'}]")

    except Exception as exc:
        import traceback
        print(f"\n  [ERROR]: {exc}")
        traceback.print_exc()

print(f"\n{'='*65}")
print(f"SUMMARY: {hits}/{N_SAMPLES} correct  (Hits@1 = {hits/N_SAMPLES:.1%})")
print(f"{'='*65}")
