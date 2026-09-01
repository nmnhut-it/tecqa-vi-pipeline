"""
Utility modules: grounding (Eq. 2), the LLM disk cache, and timestamp parsing.

Deliberately empty of imports. `grounding` pulls in sentence-transformers and
torch, and re-exporting it here would make even `from tecqa.utils.timeutil
import parse_ts` load a deep-learning stack — which the offline scorer and the
notebook's replay mode must be able to skip (docs/EVAL_DESIGN.md Sec 8).
"""
