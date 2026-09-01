"""
MultiTQ Knowledge Graph in-memory representation.

Parses dataset/MultiTQ/kg/full.txt into indexed lookup structures:
  - self.facts: list of tuples (sub, rel, obj, timestamp)
  - self.entities: set of canonical entity strings
  - self.relations: set of canonical relation strings
  - self.facts_touching(entity): all facts where entity is subject or object
"""
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = (
    Path(__file__).parent.parent.parent
    / "dataset_vi" / "raw" / "extracted" / "MultiTQ"
)
KG_PATH = DATASET_ROOT / "kg" / "full.txt"



class MultiTQGraph:
    """In-memory indexing for the MultiTQ temporal knowledge graph."""

    def __init__(self, kg_path: Path = KG_PATH):
        self.kg_path = Path(kg_path)
        self.facts: list[tuple[str, str, str, str]] = []
        self.entities: set[str] = set()
        self.relations: set[str] = set()
        self._by_entity: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    def load(self) -> "MultiTQGraph":
        """Load and index full.txt (461,329 facts)."""
        with open(self.kg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 4:
                    continue
                sub, rel, obj, ts = parts
                t = (sub, rel, obj, ts)
                self.facts.append(t)
                self.entities.add(sub)
                self.entities.add(obj)
                self.relations.add(rel)
                self._by_entity[sub].append(t)
                self._by_entity[obj].append(t)
        return self

    def facts_touching(self, entity: str) -> list[tuple[str, str, str, str]]:
        """Return all facts where entity is the subject or object."""
        return self._by_entity.get(entity, [])

    @staticmethod
    def fact_tuple_str(f: tuple[str, str, str, str]) -> str:
        """Paper Appendix F.3 rendering: [sub, rel, obj, date]."""
        return f"[{f[0]}, {f[1]}, {f[2]}, {f[3]}]"

    @staticmethod
    def fact_text(f: tuple[str, str, str, str]) -> str:
        """Human-readable rendering for embedding / semantic similarity."""
        s, r, o, t = f
        return f"{s.replace('_', ' ')} {r.replace('_', ' ')} {o.replace('_', ' ')} ({t})"
