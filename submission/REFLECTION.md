# Reflection — Lab 19

**Tên:** Nguyễn Khắc Huy
**Cohort:** A20 — 2A202602036
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 đo được: `exact` (n=15) BM25 96,7% = hybrid 96,7%, vector 88,7%.
`paraphrase` (n=15) BM25 33,3% > hybrid 32,0% > vector 24,0%. `mixed` (n=20)
hybrid 100% > vector 98,5% > BM25 97,0%. Trung bình hybrid 78,6% > BM25 77,8% >
vector 73,2%.

Hybrid chỉ **thực sự** thắng ở `mixed` — loại query vừa có thuật ngữ nguyên văn
vừa có ý diễn đạt lại, nên hai retriever bù lỗi cho nhau. Ở `exact`, BM25 đã
bão hoà; hybrid không thêm gì.

Điều bất ngờ là `paraphrase`: hybrid **thấp hơn** BM25. Lý do là
`bge-small-en-v1.5` được huấn luyện cho tiếng Anh, nên trên câu tiếng Việt diễn
đạt lại nó chỉ đạt 24% — RRF trộn một retriever yếu vào thì kéo kết quả xuống,
chứ fusion không tự biết retriever nào đáng tin.

Nên **không** dùng hybrid khi: (1) query là mã/định danh chính xác — BM25 đủ và
rẻ hơn 5× (P99 3,3 ms so với 16,1 ms); (2) embedding model không phủ ngôn ngữ
của corpus — phải đổi model (bge-m3) trước, chứ không phải thêm fusion.

_(179 chữ)_

---

## Điều ngạc nhiên nhất khi làm lab này

Nút thắt latency không nằm trong code lab mà trong **file model**: fastembed
≥ 0.8 phát hành `bge-small-en-v1.5` dạng float16, mà CPU provider của ONNX
Runtime không có kernel fp16 nên cast lại 33M trọng số ở *mỗi* lần embed —
~30 ms cố định/lời gọi. Convert sang fp32 một lần (`make fix-model`):
62 ms → 3,1 ms mỗi query, index corpus 179 s → 15,5 s, hybrid P99 116 ms →
14,6 ms, còn Precision@10 **không đổi một chữ số nào**. Bài học: trước khi
tinh chỉnh thuật toán, hãy đo xem thời gian thực sự đi đâu.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(làm một mình)_
