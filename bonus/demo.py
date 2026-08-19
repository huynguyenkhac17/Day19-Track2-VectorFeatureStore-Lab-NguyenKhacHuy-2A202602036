"""bonus/demo.py — five queries through HybridMemoryAgent.

    python bonus/demo.py          # exits 0

Query 1 needs only the vector store. Query 2 and 3 are unanswerable without the
feature store. Query 4 is a pure paraphrase (no shared keyword with the stored
note). Query 5 needs both halves. The point of printing all five side by side is
that you can see *which* half of the memory system each one leans on.

The user is `u_001` because that entity already exists in the NB4 Feast repo, so
the profile block is real data from the online store rather than a mock.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent

USER = "u_001"
NOW = datetime.now(timezone.utc)

# What the assistant has "seen" this user read or say. Deliberately mixed
# Vietnamese/English the way a real VN engineer writes — that code-switching is
# the reason chunking splits on sentences and BM25 stays in the fusion.
NOTES: list[tuple[str, str, int]] = [
    ("Hôm nay đọc tài liệu về Kubernetes Pod lifecycle. Pod đi qua các pha Pending, "
     "Running, Succeeded, Failed. Ghi nhớ: readiness probe khác liveness probe, "
     "readiness quyết định có nhận traffic hay không.", "cloud", 3),
    ("Đọc bài về auto-scaling: HPA scale theo CPU và custom metrics, cluster "
     "autoscaler thêm node khi Pod pending. Chi phí giảm rõ khi dùng spot instance "
     "cho worker không critical.", "cloud", 5),
    ("Note nhanh về OAuth2: authorization code flow + PKCE cho mobile app. "
     "Không bao giờ để refresh token trong localStorage của trình duyệt.", "security", 8),
    ("Đọc về mã hoá dữ liệu nhạy cảm: TLS cho dữ liệu trên đường truyền, AES-256 "
     "cho dữ liệu lưu trữ. Nghị định 13 yêu cầu dữ liệu cá nhân của người dùng "
     "Việt Nam phải có biện pháp bảo vệ tương xứng.", "security", 9),
    ("Thử benchmark hybrid search: BM25 cộng vector rồi fuse bằng RRF k=60 cho "
     "Precision@10 cao hơn cả hai mode đơn lẻ trên golden set 50 câu.", "ai_ml", 1),
    ("Ghi chú về Feast: materialize-incremental đẩy giá trị mới nhất sang online "
     "store, còn get_historical_features làm point-in-time join cho training.",
     "ai_ml", 2),
    ("Tuần trước đọc về tối ưu truy vấn Postgres: chỉ mục B-tree, EXPLAIN ANALYZE, "
     "và khi nào partial index đáng dùng hơn index đầy đủ.", "database", 14),
]

QUERIES: list[tuple[str, str]] = [
    ("Tôi đã đọc gì về Kubernetes?",
     "chỉ cần vector/BM25 — keyword có mặt nguyên văn trong ký ức"),
    ("Recommend đọc gì tiếp",
     "không có keyword nào để bám — phải dựa vào topic_affinity từ feature store"),
    ("Tôi đang quan tâm gì gần đây?",
     "cần queries_last_hour + distinct_topics_24h, tức là streaming feature"),
    ("Tài liệu về tự động mở rộng hạ tầng?",
     "paraphrase thuần: ký ức viết 'auto-scaling', câu hỏi không có từ đó"),
    ("Cho tôi summary cloud security",
     "ghép hai chủ đề — cần episodic của cả hai cụm cộng hồ sơ để xếp hạng"),
]


def main() -> int:
    agent = HybridMemoryAgent()
    n = sum(agent.remember(text, user_id=USER, topic=topic,
                           when=NOW - timedelta(days=days))
            for text, topic, days in NOTES)
    print(f"Đã ghi {n} chunk ký ức cho {USER} từ {len(NOTES)} ghi chú.")
    print(f"Feature store: {'đã kết nối' if agent._store else 'chưa apply (chạy NB4 để bật)'}")

    for i, (q, why) in enumerate(QUERIES, 1):
        print("\n" + "=" * 78)
        print(f"QUERY {i}: {q}\n   (vì sao đáng test: {why})")
        print("-" * 78)
        print(agent.recall(q, user_id=USER))

    print("\n" + "=" * 78)
    # Isolation check: the same question from another user must not reach u_001's
    # notes. Same failure mode as the cross-tenant cache leak in NB7.
    other = agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_042")
    leaked = "Kubernetes Pod lifecycle" in other
    print(f"Cách ly per-user: u_042 hỏi cùng câu → {'RÒ RỈ' if leaked else 'không thấy ký ức của u_001 (đúng)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
