"""
Redis-backed work queue, result store and shared budget for an evaluation run.

OWNER: EVAL (docs/TEAM_PLAN.md H5).

DISK IS AUTHORITATIVE, REDIS ONLY COORDINATES. Every answer is fsync'd to
results/<run_id>.jsonl before it is recorded in Redis, and the answered set is
rebuilt from that file on every startup. Redis can therefore be flushed,
restarted, or switched off permanently without losing a single paid-for answer;
the worst case is rebuilding the queue from the sample, which costs nothing.
Nothing downstream reads Redis — make_tables.py reads results/ (contract H3).

What Redis buys, given that:

  * A queue that several workers can share without stepping on each other, so a
    second worker can join a run already in progress just by being started.
  * A spending cap that is atomic across those workers. INCRBYFLOAT is; a
    Python float in one process is not.
  * Cheap resumption: what is still outstanding is a fact about the queue, not
    something to recompute by diffing files.

Keys, all under tecqa:run:<run_id>:
    order      LIST   every qid in sample order, for a deterministic dump
    queue      LIST   qids still to do
    inflight   LIST   qids claimed by a worker but not yet finished
    done       HASH   qid -> the result row as JSON
    spend      STRING dollars spent, shared across workers
    cap        STRING the run's dollar ceiling

Input:  a run_id and a list of Question objects.
Output: claimed qids, stored rows, and a .jsonl dump.
"""
import json
import os
import threading
from collections import deque
from pathlib import Path

import redis

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

KEY_PREFIX = "tecqa:run"
DEFAULT_URL = os.environ.get("TECQA_REDIS_URL", "redis://localhost:6379/0")
CLAIM_TIMEOUT = 5  # seconds a worker waits for new work before concluding it is done


def connect(url: str = None):
    """Client with str (not bytes) values. Raises if Redis is unreachable.

    socket_timeout must exceed CLAIM_TIMEOUT: claim() blocks server-side for
    CLAIM_TIMEOUT waiting for work, and a shorter socket timeout makes the
    client give up on its own healthy request.
    """
    client = redis.Redis.from_url(url or DEFAULT_URL, decode_responses=True,
                                  socket_timeout=CLAIM_TIMEOUT * 2 + 5)
    client.ping()
    return client


def is_answered(row: dict) -> bool:
    """Whether a journal row is a real answer, and so must never be re-asked.

    score_question() writes a row carrying meta.error when a question raises, so
    that one bad question cannot abandon a whole run. That row is a crash, not
    an answer: pred is empty and Hits@1 has already scored it zero. Adopting it
    as done would freeze a transient failure — a dead API, a bug shipped
    mid-run — into a permanent zero for that question, which is exactly the
    mistake llm_cache refuses to make when it declines to cache an unparseable
    response. Error rows are therefore replayed on the next run; the fresh row
    is appended after the old one, and rows() keeps the later of the two.
    """
    return bool(row.get("qid")) and not (row.get("meta") or {}).get("error")


class RunStore:
    """The queue, the results and the budget for one run_id.

    Disk is authoritative, Redis is the coordinator. Every answer is appended to
    results/<run_id>.jsonl the instant it is produced, and the answered set is
    re-seeded from that file on startup. So Redis can be flushed, restarted or
    switched off entirely without losing a single paid-for answer — the worst
    case is that the queue has to be rebuilt from the sample, which is free.
    """

    def __init__(self, run_id: str, client=None, url: str = None, journal_dir=None):
        self.run_id = run_id
        self.redis = client or connect(url)
        self.journal = Path(journal_dir or RESULTS_DIR) / f"{run_id}.jsonl"
        self._journal_lock = threading.Lock()

    def _key(self, name: str) -> str:
        return f"{KEY_PREFIX}:{self.run_id}:{name}"

    # -- disk journal --------------------------------------------------------
    def journal_rows(self) -> list:
        """Rows already on disk. A half-written final line — killed mid-append —
        is skipped rather than aborting the whole read."""
        if not self.journal.exists():
            return []
        rows = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def _append_journal(self, row: dict) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_lock, self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())  # survive a hard kill, not just a clean exit

    def adopt_journal(self) -> int:
        """Load the disk journal into Redis' answered set. This is what makes a
        Redis wipe survivable: the questions already paid for come back, and
        only genuinely-missing ones get queued."""
        rows = [row for row in self.journal_rows() if is_answered(row)]
        if rows:
            self.redis.hset(self._key("done"),
                            mapping={row["qid"]: json.dumps(row, ensure_ascii=False)
                                     for row in rows})
        return len(rows)

    def _answered_in_redis(self) -> set:
        """qids Redis holds a real answer for. complete() files every row under
        `done`, error rows included, so that one bad question does not stall a
        worker -- but is_answered() says those rows must be asked again on the
        next run. The disk store honours that through adopt_journal(); this
        store must drop them here too, or a 429 answered during one run is
        frozen as a permanent zero for as long as Redis remembers it (ten of
        them, in the 1,800-question VI run)."""
        answered, stale = set(), []
        for qid, blob in self.redis.hgetall(self._key("done")).items():
            if is_answered(json.loads(blob)):
                answered.add(qid)
            else:
                stale.append(qid)
        if stale:
            self.redis.hdel(self._key("done"), *stale)
        return answered

    # -- queue ---------------------------------------------------------------
    def enqueue(self, questions) -> int:
        """Queue every question not already answered. Idempotent: running it
        again after a crash re-queues only what is still outstanding, so a
        second worker can join a run in progress without coordination."""
        self.adopt_journal()
        done = self._answered_in_redis()
        pipe = self.redis.pipeline()
        pipe.delete(self._key("order"))
        pipe.rpush(self._key("order"), *[q.qid for q in questions])
        pending = [q.qid for q in questions if q.qid not in done]
        queued = set(self.redis.lrange(self._key("queue"), 0, -1))
        queued |= set(self.redis.lrange(self._key("inflight"), 0, -1))
        fresh = [qid for qid in pending if qid not in queued]
        if fresh:
            pipe.rpush(self._key("queue"), *fresh)
        pipe.execute()
        return len(pending)

    def claim(self) -> str:
        """Atomically move one qid from the queue to the in-flight list. Returns
        None when the queue is empty. Blocking, so a worker that finishes early
        waits briefly for a slower sibling to release work rather than exiting.

        Already-answered qids are skipped rather than handed out. enqueue() can
        put a duplicate on the queue if it runs while another worker is mid-
        question — it reads `done` and `inflight` at slightly different moments,
        and a question that completes in between looks like neither. Checking
        here is the cheap, always-correct backstop: paying twice for the same
        answer is exactly what this whole design is meant to prevent.
        """
        while True:
            try:
                qid = self.redis.blmove(self._key("queue"), self._key("inflight"),
                                        CLAIM_TIMEOUT, "LEFT", "RIGHT")
            except redis.exceptions.TimeoutError:
                return None  # nothing arrived in the window: treat as drained
            if qid is None:
                return None
            if not self.redis.hexists(self._key("done"), qid):
                return qid
            self.redis.lrem(self._key("inflight"), 1, qid)  # drop the duplicate

    def complete(self, qid: str, row: dict) -> None:
        """Disk first, then Redis. If the process dies between the two, the
        answer is still on disk and adopt_journal() puts it back — whereas the
        reverse order could lose a paid-for answer to a Redis restart."""
        self._append_journal(row)
        pipe = self.redis.pipeline()
        pipe.hset(self._key("done"), qid, json.dumps(row, ensure_ascii=False))
        pipe.lrem(self._key("inflight"), 1, qid)
        pipe.execute()

    def release(self, qid: str) -> None:
        """Hand a claimed-but-unfinished question back to the queue, so a worker
        stopping on budget does not strand it as permanently in-flight."""
        pipe = self.redis.pipeline()
        pipe.lrem(self._key("inflight"), 1, qid)
        pipe.lpush(self._key("queue"), qid)
        pipe.execute()

    def requeue_stale(self) -> int:
        """Return every in-flight question to the queue. Call at startup when no
        other worker is running: anything still in-flight belongs to a worker
        that died holding it."""
        stale = self.redis.lrange(self._key("inflight"), 0, -1)
        if stale:
            pipe = self.redis.pipeline()
            pipe.delete(self._key("inflight"))
            pipe.rpush(self._key("queue"), *stale)
            pipe.execute()
        return len(stale)

    # -- results -------------------------------------------------------------
    def rows(self) -> list:
        """Every answered row, in the sample's original order.

        Merges the disk journal with Redis rather than trusting either alone: a
        wiped Redis still yields the full history, and a journal that lost its
        tail to a hard kill is topped up from Redis. Rows whose qid is no longer
        in `order` (the sample shrank) go at the end, so nothing paid for is
        quietly dropped.
        """
        merged = {row["qid"]: row for row in self.journal_rows() if row.get("qid")}
        for qid, blob in self.redis.hgetall(self._key("done")).items():
            merged.setdefault(qid, json.loads(blob))
        order = self.redis.lrange(self._key("order"), 0, -1)
        ordered = [merged[qid] for qid in order if qid in merged]
        seen = set(order)
        return ordered + [row for qid, row in merged.items() if qid not in seen]

    def counts(self) -> dict:
        return {"done": self.redis.hlen(self._key("done")),
                "queued": self.redis.llen(self._key("queue")),
                "inflight": self.redis.llen(self._key("inflight"))}

    def dump(self, path) -> int:
        """Write results/<run_id>.jsonl from Redis. Rewritten in full each time,
        so it is always a consistent snapshot rather than an append log."""
        rows = self.rows()
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    def clear(self) -> None:
        """Discard this run entirely, disk journal included — otherwise --fresh
        would immediately re-adopt the answers it was asked to throw away."""
        for name in ("order", "queue", "inflight", "done", "spend", "cap"):
            self.redis.delete(self._key(name))
        if self.journal.exists():
            self.journal.unlink()


class LocalStore:
    """Same interface as RunStore, backed only by the disk journal.

    Used when Redis is not running. Everything that actually protects paid-for
    work — the fsync'd journal, resuming from it, never re-asking a question —
    lives on disk anyway, so losing Redis costs only the ability for a second
    process to join the same run.
    """

    def __init__(self, run_id: str, journal_dir=None):
        self.run_id = run_id
        self.journal = Path(journal_dir or RESULTS_DIR) / f"{run_id}.jsonl"
        self._journal_lock = threading.Lock()
        self._queue = deque()
        self._inflight = set()
        self._order = []
        self._done = {}

    journal_rows = RunStore.journal_rows
    _append_journal = RunStore._append_journal

    def adopt_journal(self) -> int:
        self._done = {row["qid"]: row for row in self.journal_rows() if is_answered(row)}
        return len(self._done)

    def enqueue(self, questions) -> int:
        self.adopt_journal()
        self._order = [q.qid for q in questions]
        pending = [q.qid for q in questions if q.qid not in self._done]
        self._queue = deque(pending)
        return len(pending)

    def claim(self):
        with self._journal_lock:
            if not self._queue:
                return None
            qid = self._queue.popleft()
            self._inflight.add(qid)
            return qid

    def complete(self, qid: str, row: dict) -> None:
        self._append_journal(row)
        self._done[qid] = row
        self._inflight.discard(qid)

    def release(self, qid: str) -> None:
        self._inflight.discard(qid)
        self._queue.appendleft(qid)

    def requeue_stale(self) -> int:
        return 0  # nothing outlives this process

    def rows(self) -> list:
        merged = {row["qid"]: row for row in self.journal_rows() if row.get("qid")}
        ordered = [merged[qid] for qid in self._order if qid in merged]
        seen = set(self._order)
        return ordered + [row for qid, row in merged.items() if qid not in seen]

    def counts(self) -> dict:
        return {"done": len(self._done), "queued": len(self._queue),
                "inflight": len(self._inflight)}

    def clear(self) -> None:
        self._queue.clear()
        self._inflight.clear()
        self._done.clear()
        if self.journal.exists():
            self.journal.unlink()


class LocalBudget:
    """Process-local spending cap, for the no-Redis path."""

    def __init__(self, store, hard_cap_usd: float):
        self.hard_cap_usd = hard_cap_usd
        self._lock = threading.Lock()
        self._spent = 0.0

    def add(self, model: str, prompt_tokens: int, completion_tokens: int,
            real_cost: float = None) -> float:
        with self._lock:
            self._spent += real_cost or 0.0
            return self._spent

    @property
    def total_usd(self) -> float:
        return self._spent

    def over_budget(self) -> bool:
        return self._spent >= self.hard_cap_usd


def spent_so_far(store) -> float:
    """Dollars this run has already spent, across every earlier session."""
    if isinstance(store, LocalStore):
        return 0.0  # process-local budget: nothing carries over
    return float(store.redis.get(store._key("spend")) or 0.0)


class ReadOnlyBudget:
    """Reports a run's spend without ever writing to it. Used by --dump-only,
    which must not disturb a run another process is executing."""

    def __init__(self, store):
        self.store = store
        self.hard_cap_usd = float("inf")

    @property
    def total_usd(self) -> float:
        return spent_so_far(self.store)

    def over_budget(self) -> bool:
        return False


def open_store(run_id: str, prefer_redis: bool = True) -> tuple:
    """(store, budget_class, note). Falls back to disk-only when Redis is down,
    so turning Redis off degrades the run rather than breaking it."""
    if prefer_redis:
        try:
            return RunStore(run_id), RedisBudget, "queue: redis"
        except Exception as exc:
            return (LocalStore(run_id), LocalBudget,
                    f"queue: local disk only (Redis unavailable: {type(exc).__name__})")
    return LocalStore(run_id), LocalBudget, "queue: local disk only (--no-redis)"


class RedisBudget:
    """Spending cap shared by every worker on a run.

    Implements the interface dataset_vi.build_vi_dataset_or.call_or expects
    (.add / .over_budget), so it drops straight into the existing LLM caller.
    Cost comes from OpenRouter's own usage.cost when the response carries it —
    the local price table was measured undercounting reasoning models badly
    enough to bust a cap.
    """

    def __init__(self, store: RunStore, hard_cap_usd: float):
        self.store = store
        self.hard_cap_usd = hard_cap_usd
        self.store.redis.set(store._key("cap"), hard_cap_usd)

    def add(self, model: str, prompt_tokens: int, completion_tokens: int,
            real_cost: float = None) -> float:
        cost = real_cost if real_cost is not None else 0.0
        return float(self.store.redis.incrbyfloat(self.store._key("spend"), cost))

    @property
    def total_usd(self) -> float:
        return float(self.store.redis.get(self.store._key("spend")) or 0.0)

    def over_budget(self) -> bool:
        return self.total_usd >= self.hard_cap_usd
