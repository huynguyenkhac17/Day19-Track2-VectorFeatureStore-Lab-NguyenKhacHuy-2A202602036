# Reflection — Lab 19

**Tên:** Nguyễn Khắc Huy
**MSSV:** 2A202602036
**Cohort:** A20
**Path đã chạy:** lite (Python 3.11.9, Windows 11, i7-13700HX)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Đo được (Precision@10): trung bình kw 77,8% · sem 73,2% · **hyb 78,6%**.
Theo lát cắt — `exact`: kw 96,7% = hyb 96,7% > sem 88,7%; `paraphrase`: kw
33,3% > hyb 32,0% > sem 24,0%; `mixed`: **hyb 100%** > sem 98,5% > kw 97,0%.

`exact` chứa thuật ngữ nguyên văn nên BM25 đã đủ, hybrid chỉ hoà. `mixed` có
cả từ khoá lẫn ý diễn đạt lại — mỗi retriever bù đúng chỗ kia hụt, nên hybrid
thắng và kéo luôn trung bình.

Bất ngờ nhất: `paraphrase` **vector thua cả BM25**. Đây không phải lỗi RRF mà
là lỗi chọn model: `bge-small-en-v1.5` được huấn luyện tiếng Anh, câu hỏi
tiếng Việt diễn đạt lại nằm ngoài phân phối của nó. Hybrid kế thừa luôn phần
yếu đó.

Không dùng hybrid khi: (1) ngân sách độ trễ chặt — đo được keyword P99 2,9 ms
so với hybrid 83,5 ms, toàn bộ chênh lệch là một lần forward pass embedding;
(2) query là mã định danh (SKU, mã lỗi, tên hàm) — BM25 thuần đúng hơn;
(3) embedding model lệch ngôn ngữ/miền như trên — thêm vector chỉ thêm nhiễu.

---

## Điều ngạc nhiên nhất khi làm lab này

Rằng "cải thiện chất lượng" và "vượt ngưỡng rubric" là hai chuyện tách rời:
hybrid thắng về Precision@10 nhưng **không thể** đạt P99 < 50 ms trên máy này,
vì riêng một lần forward pass của `bge-small` đã tốn ~57 ms — nút thắt nằm ở
model, không nằm ở đường truy xuất (keyword P99 chỉ 2,9 ms trên cùng phép đo).
Muốn cả hai thì phải cache embedding của query hoặc đổi runtime, chứ tối ưu RRF
không cứu được.

Hai con số latency trong bài **không** khớp nhau và đó là chuyện có thật đáng
ghi lại: `make benchmark` đo in-process cho hybrid P99 = 83,5 ms, còn NB3 đo
qua HTTP cho 535 ms. Chênh lệch đến từ (a) mỗi request là một kết nối TCP mới
— P99 wall-clock lên tới ~3 s ngay cả với keyword vốn chỉ tốn 3,5 ms
server-side, và (b) endpoint đồng bộ của FastAPI chạy trên threadpool: đo được
rằng gọi ONNX lặp lại từ một worker thread cố định tốn 60–320 ms tuỳ lần, so
với 57–76 ms khi gọi từ main thread. Bài học: **P99 phụ thuộc chỗ bạn đặt
đồng hồ**, và một con số latency không kèm mô tả đường đo thì vô nghĩa.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/` — `ARCHITECTURE.md`, `agent.py`, `demo.py`)
- [ ] Pair work với: _(làm một mình)_
