"""HybridMemoryAgent — episodic memory (vector) + stable profile (feature store).

Bonus challenge for Day 19. Design notes live in `bonus/ARCHITECTURE.md`; this
file is the runnable POC behind them.

Two memories, two lifecycles:

    remember()  ->  Qdrant collection `bonus_memory`, one point per chunk,
                    payload carries user_id + written-at day (for decay).
    recall()    ->  Feast online lookup ("who is this user")
                 +  hybrid retrieval over that user's own chunks ("what is relevant")
                 -> one assembled context string, ready to hand an LLM.

Every retrieval decision here is a lab lesson applied, not a new invention:

  * NB2 — fusion is RRF with k=60, rank 1-based. Three ranked lists, not two:
          BM25, vector, and a profile-affinity list from the feature store.
  * NB5 — the per-user isolation filter goes *inside* the ANN call
          (`filtered_ann`, never post-filter): a user with few memories is
          exactly the high-selectivity case where post-filter returns nothing.
  * NB7 — memories carry a written-at day so decay is a range filter, and the
          user_id filter is the same namespace discipline that stops the
          cross-tenant leak.
  * NB4 — profile features come from the online store, so training-time and
          serving-time personalisation read the same values.

No LLM call: `recall()` returns the context string an LLM would receive. That
keeps the POC zero-key and makes the retrieval behaviour the thing under test.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

MEMORY_COLLECTION = "bonus_memory"
RRF_K = 60                    # same constant as NB2 — do not "tune" it blind
CHUNK_TARGET_WORDS = 45       # see ARCHITECTURE.md §Decision 1
EPOCH = date(2026, 1, 1)

# Vietnamese sentence enders + the usual ASCII ones. Vietnamese text in this
# corpus is unaccented-safe: we split on punctuation, never on diacritics.
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _day_index(when: datetime | None = None) -> int:
    """Days since EPOCH — an int so Qdrant can range-filter it (NB5 pattern)."""
    when = when or datetime.now(timezone.utc)
    return (when.date() - EPOCH).days


def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """Sentence-window chunking: pack whole sentences up to ~target_words.

    Not per-message (too small to embed meaningfully, and a single note gets
    scattered across many near-duplicate points) and not per-conversation
    (one vector for ten topics lands between clusters and matches none).
    Sentence boundaries are cheap, language-agnostic, and never cut a Vietnamese
    word in half the way a fixed token window does.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    n = 0
    for s in sentences:
        w = len(s.split())
        if buf and n + w > target_words:
            chunks.append(" ".join(buf))
            buf, n = [], 0
        buf.append(s)
        n += w
    if buf:
        chunks.append(" ".join(buf))
    return chunks or ([text.strip()] if text.strip() else [])


@dataclass
class Memory:
    """One retrieved chunk plus why it was retrieved."""
    memory_id: int
    user_id: str
    text: str
    topic: str
    day: int
    score: float

    @property
    def age_days(self) -> int:
        return _day_index() - self.day


@dataclass
class HybridMemoryAgent:
    """Minimal personal-assistant memory over Qdrant + Feast."""

    client: QdrantClient = field(default_factory=lambda: QdrantClient(":memory:"))
    embedder: Embedder = field(default_factory=Embedder)
    feast_repo: Path | None = None
    ttl_days: int = 180           # episodic decay; profile has no TTL here
    _next_id: int = 0
    _texts: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    _store: object | None = None

    def __post_init__(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if MEMORY_COLLECTION not in names:
            self.client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.embedder.dim, distance=models.Distance.COSINE
                ),
            )
        # Payload indexes: on a real Qdrant these are what keep the per-user
        # filter inside the HNSW walk instead of after it (NB5).
        for fname, ftype in (("user_id", models.PayloadSchemaType.KEYWORD),
                             ("topic", models.PayloadSchemaType.KEYWORD),
                             ("day", models.PayloadSchemaType.INTEGER)):
            try:
                self.client.create_payload_index(MEMORY_COLLECTION, fname, field_schema=ftype)
            except Exception:      # local mode ignores payload indexes
                pass
        self._store = self._open_feast()

    # ── feature store ───────────────────────────────────────────────────
    def _open_feast(self):
        """Open the Feast repo from NB4 if it has been applied; else run without.

        The assistant must degrade to grounding-only rather than crash: a brand
        new user has no profile row either, and that path has to work.
        """
        repo = self.feast_repo or (ROOT / "app" / "feast_repo")
        if not (repo / "registry.db").exists():
            return None
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(repo))
        except Exception:          # noqa: BLE001 — profile is optional, never fatal
            return None

    def profile(self, user_id: str) -> dict:
        """Stable profile + recent activity, one online lookup (NB4, < 10 ms)."""
        if self._store is None:
            return {}
        try:
            out = self._store.get_online_features(
                features=[
                    "user_profile_features:topic_affinity",
                    "user_profile_features:preferred_language",
                    "user_profile_features:reading_speed_wpm",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
        except Exception:          # noqa: BLE001
            return {}
        return {k: v[0] for k, v in out.items() if k != "user_id"}

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001",
                 topic: str = "general", when: datetime | None = None) -> int:
        """Add one piece of episodic memory. Returns the number of chunks stored."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        day = _day_index(when)
        vectors = list(self.embedder.embed(chunks))
        points = []
        for body, vec in zip(chunks, vectors):
            points.append(models.PointStruct(
                id=self._next_id,
                vector=np.asarray(vec, dtype=np.float32).tolist(),
                payload={"user_id": user_id, "text": body, "topic": topic, "day": day},
            ))
            self._texts.setdefault(user_id, []).append((self._next_id, body))
            self._next_id += 1
        self.client.upsert(collection_name=MEMORY_COLLECTION, points=points)
        return len(points)

    # ── read path ───────────────────────────────────────────────────────
    def _user_filter(self, user_id: str) -> models.Filter:
        """Isolation + decay in one filter, handed to the engine (never post-hoc)."""
        return models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="day",
                                  range=models.Range(gte=_day_index() - self.ttl_days)),
        ])

    def _vector_ids(self, query: str, user_id: str, depth: int) -> list[int]:
        qv = np.asarray(next(self.embedder.embed([query])), dtype=np.float32)
        hits = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=qv.tolist(),
            query_filter=self._user_filter(user_id),
            limit=depth,
        ).points
        return [int(h.id) for h in hits]

    def _bm25_ids(self, query: str, user_id: str, depth: int) -> list[int]:
        owned = self._texts.get(user_id, [])
        if not owned:
            return []
        bm25 = BM25Okapi([t.lower().split() for _, t in owned])
        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(owned)), key=lambda i: -scores[i])[:depth]
        return [owned[i][0] for i in order if scores[i] > 0]

    def _affinity_ids(self, user_id: str, affinity: str | None, depth: int) -> list[int]:
        """Third retriever: this user's own memories in their favourite topic.

        Personalisation as a *ranked list fused by RRF*, not a score multiplier —
        a multiplier needs the two score scales to be comparable, and cosine
        similarity and BM25 are not.
        """
        if not affinity:
            return []
        hits, _ = self.client.scroll(
            collection_name=MEMORY_COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="topic", match=models.MatchValue(value=affinity)),
            ]),
            limit=depth,
            with_payload=True,
        )
        # Freshest first — within one topic, recency is the best cheap prior.
        hits = sorted(hits, key=lambda p: -p.payload.get("day", 0))
        return [int(p.id) for p in hits]

    def search_memories(self, query: str, user_id: str = "u_001",
                        top_k: int = 3, affinity: str | None = None) -> list[Memory]:
        """Hybrid retrieval over one user's memories: RRF(BM25, vector, affinity)."""
        depth = max(top_k * 5, 20)
        lists = [
            self._bm25_ids(query, user_id, depth),
            self._vector_ids(query, user_id, depth),
            self._affinity_ids(user_id, affinity, depth),
        ]
        rrf: dict[int, float] = {}
        for ids in lists:
            for rank, mid in enumerate(ids, start=1):        # 1-based, as in NB2
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
        if not rrf:
            return []
        best = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        got = self.client.retrieve(collection_name=MEMORY_COLLECTION,
                                   ids=[mid for mid, _ in best], with_payload=True)
        payloads = {int(p.id): p.payload for p in got}
        out = []
        for mid, score in best:
            p = payloads.get(mid)
            if p is None:
                continue
            out.append(Memory(memory_id=mid, user_id=p["user_id"], text=p["text"],
                              topic=p.get("topic", "general"), day=int(p.get("day", 0)),
                              score=score))
        return out

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Profile + episodic memories, assembled into one context string."""
        t0 = time.perf_counter()
        prof = self.profile(user_id)
        affinity = prof.get("topic_affinity")
        mems = self.search_memories(query, user_id, top_k=top_k, affinity=affinity)
        ms = (time.perf_counter() - t0) * 1000

        lines = [f"### Ngữ cảnh cho: {query!r}  (user={user_id})"]
        if prof:
            lines.append(
                f"[Hồ sơ]  ngôn ngữ={prof.get('preferred_language')}  "
                f"tốc độ đọc={prof.get('reading_speed_wpm')} wpm  "
                f"quan tâm={affinity}"
            )
            lines.append(
                f"[Gần đây] {prof.get('queries_last_hour')} truy vấn/giờ  ·  "
                f"{prof.get('distinct_topics_24h')} chủ đề/24h"
            )
        else:
            lines.append("[Hồ sơ]  (chưa có — chạy NB4 `feast apply` để bật cá nhân hoá)")

        if mems:
            lines.append(f"[Ký ức] top-{len(mems)}:")
            for i, m in enumerate(mems, 1):
                lines.append(f"  {i}. ({m.topic}, {m.age_days}d trước, rrf={m.score:.4f}) {m.text}")
        else:
            lines.append("[Ký ức] (không có ký ức nào khớp — trả lời không có grounding)")
        lines.append(f"[Chi phí] lắp ngữ cảnh trong {ms:.1f} ms")
        return "\n".join(lines)
