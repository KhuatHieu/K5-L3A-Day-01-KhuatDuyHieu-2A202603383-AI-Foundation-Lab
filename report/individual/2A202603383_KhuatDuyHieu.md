# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV: 2A202603383**
- **Họ và tên: Khuất Duy Hiếu**
- **SEED: 7655**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — có seed, checks, predictions, results | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Hàng LLM chỉ đo trên SUB 40 tin. Cột F1 test của hàng 4 là F1 trên SUB, không phải toàn bộ test.

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.882 | 0.947 | 0.15 | 0 |
| 2. BoW + MLP | 0.928 | 1.000 | 0.27 | 0 |
| 3a. Encoder + kNN | 0.923 | 1.000 | 32.6 | 0 |
| 3b. Encoder + LogReg | 0.951 | 1.000 | 9.9 | 0 |
| 3c. Encoder + MLP head | 0.957 | 1.000 | 12.1 | 0 |
| 4a. LLM nhỏ zero-shot | — | 1.000 | 1561 | 0.006 |
| 4b. LLM lớn zero-shot | — | 1.000 | 1776 | 0.057 |
| 4c. LLM nhỏ few-shot | — | 0.974 | 1403 | 0.019 |

Lần chạy Colab đầu cho F1 LLM = 0 và `invalid=40` vì `max_completion_tokens=3` trả content rỗng. Bảng trên là lần chạy lại với 16 token, `invalid=0`.

## 3. Dự đoán và thực tế

1. **tradeoff / chất lượng LLM.** Đoán LLM nhỏ sẽ nằm trên Pareto nếu F1 tốt nhất, và kết luận nháp là LLM lỗi rất cao. Lần F1=0 là câu trả lời rỗng, không phải model kém. Sau khi sửa, 4a/4b có F1 SUB = 1.000, nhưng p95 khoảng 1.5 giây. BoW + MLP cũng F1 SUB = 1.000 mà p95 chỉ 0.27 ms, nên vẫn lấn át LLM. Pareto thực tế là BoW + LogReg và BoW + MLP.
2. **hard_eval.** Đoán accuracy LLM trên 50 case khoảng 0.70–0.85, và BoW khó nhất ở ham trông giống spam. Thực tế 4a đúng 50/50. BoW đúng 100% nhóm đó, và thấp nhất ở thao túng (12%) cùng né từ khoá (17%).
3. **llm_fewshot.** Đoán F1 giữ nguyên và chi phí tăng 4–6 lần. F1 giảm 1.000 → 0.974 vì một tin ringtone bị gọi `ham`. Chi phí tăng khoảng 3.3 lần ($0.00023 → $0.00077), không phải 4–6 lần. Model lớn cũng không đắt gấp 5 lần như đoán: $0.057 so với $0.006 mỗi 1k tin, gần 10 lần, và hòa 3b trên SUB chứ không thua.

## 4. Phân tích 50 case khó

BoW tụt nhiều nhất trong các model tự train: F1 SUB 0.947 xuống 0.378. MLP 1.000 → 0.462, 3c 1.000 → 0.612, 3a chỉ xuống 0.759. LLM giữ 1.000 trên bộ viết tay này. Điểm test thường lạc quan vì cùng phân phối với train, và spam gần trùng vẫn có thể nằm cả hai phía.

Hai case mọi model local đều sai, nhóm spam giọng tự nhiên:

- **#12:** "hi there, sorry to bother you, my friend gave me your number..." Bẫy là lời mở lịch sự, không từ khoá, dụ vào trang hẹn hò. BoW chỉ thấy token thường. Embedding kéo về các tin ham lịch sự gần nghĩa.
- **#16:** "Long time no see! I've started a little side business with skincare..." Giả bạn cũ để bán hàng. Không có `free`/`win`/`call` nên trọng số spam không được kích. LLM đọc cả câu và gắn spam sau khi parse được nhãn.

## 5. Thử thách cá nhân

Bỏ qua đề #6 (SEED 7655): sửa prompt để LLM trả JSON `label`/`reason`, parse JSON, chạy 10 tin SUB và so chi phí. Không có code, không có kết quả.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày, p95 < 50 ms/tin.

Đạt độ trễ: hàng 1, 2, 3a, 3b, 3c. LLM không đạt (p95 1.4–1.8 giây).

Đạt cả độ trễ lẫn F1 test cao nhất: **3c Encoder + MLP head**, F1 test 0.957, p95 12.1 ms, chi phí API bằng 0. Đánh đổi: F1 trên 50 case khó chỉ 0.612, kém 3a (0.759, p95 32.6 ms vẫn dưới SLA). Nếu ưu tiên spam ngụy trang hơn điểm test UCI thì chọn 3a. Không chọn BoW + MLP chỉ vì SUB F1 = 1.000: trên 40 tin, nhiều cách cùng hòa, còn F1 test đầy đủ của MLP chỉ 0.928.

LLM nhỏ khoảng $0.006/1k tin, tức $60/ngày và khoảng $1.800/tháng cho 10 triệu tin/ngày, mà vẫn trượt SLA. Model lớn khoảng $17.100/tháng.

Router: 3c xử lý tin chắc (p < 0.3 hoặc p > 0.7). Dải lưỡng lự chỉ được đưa sang LLM nếu dưới 5% lưu lượng, nếu không p95 cả hệ thống bị kéo lên khoảng 1.5 giây. Cách an toàn hơn cho SLA là đưa dải đó sang 3a.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 | Vì sao F1 LLM = 0 dù API có trả lời? Chạy lại bằng key local. | `max_completion_tokens=3` cắt content rỗng. Tăng lên 16 thì 4a/4b F1 SUB = 1.000, 4c = 0.974, 50 case khó = 1.000, invalid=0. | Đối chiếu raw chỉ còn `spam`/`ham`. Giữ chữ ký `call_openai`. |
| Báo cáo | Điền báo cáo và sửa câu "khớp dự đoán", "LLM lỗi rất cao". Bỏ các mục đã nêu, kể cả thử thách #6. | Viết lại kết luận theo số trong `lab1_report.json`. | Không viết đáp án các mục bỏ qua. Không đưa API key vào bài. |

Điều rút ra: F1 bằng 0 có thể là lỗi đo (token bị cắt), chưa phải model kém. Sau khi đo đúng, LLM đọc được spam ngụy trang nhưng p95 khoảng 1.5 giây nên không dùng làm bộ lọc duy nhất cho 10 triệu tin/ngày. Trên SUB 40 tin, nhiều cách cùng F1 1.000; một tin sai đã kéo few-shot xuống 0.974, nên không xếp hạng sát nút bằng cột đó.
