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
| 1. BoW + LogReg | 0.882 | 0.947 | 0.08 | 0 |
| 2. BoW + MLP | 0.928 | 1.000 | 0.25 | 0 |
| 3a. Encoder + kNN | 0.923 | 1.000 | 22.7 | 0 |
| 3b. Encoder + LogReg | 0.951 | 1.000 | 10.1 | 0 |
| 3c. Encoder + MLP head | 0.957 | 1.000 | 39.0 | 0 |
| 4a. LLM nhỏ zero-shot | — | 1.000 | 1471 | 0.006 |
| 4b. LLM lớn zero-shot | — | 1.000 | 1281 | 0.057 |
| 4c. LLM nhỏ few-shot | — | 0.974 | 1124 | 0.019 |

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

Đạt độ trễ: hàng 1, 2, 3a, 3b, 3c (tất cả đều có p95 < 50 ms). LLM không đạt (p95 1.1–1.5 giây).

Đạt cả độ trễ lẫn F1 test cao nhất: **3c Encoder + MLP head**, F1 test 0.957, p95 39.0 ms, chi phí API bằng 0. Đánh đổi: F1 trên 50 case khó chỉ 0.612, kém 3a (0.759, p95 22.7 ms vẫn dưới SLA). Nếu ưu tiên spam ngụy trang hơn điểm test UCI thì chọn 3a. Không chọn BoW + MLP chỉ vì SUB F1 = 1.000: trên 40 tin, nhiều cách cùng hòa, còn F1 test đầy đủ của MLP chỉ 0.928.

LLM nhỏ khoảng $0.006/1k tin, tức $60/ngày và khoảng $1.800/tháng cho 10 triệu tin/ngày, mà vẫn trượt SLA. Model lớn khoảng $17.100/tháng.

Router: 3c xử lý tin chắc (p < 0.3 hoặc p > 0.7). Dải lưỡng lự chỉ được đưa sang LLM nếu dưới 5% lưu lượng, nếu không p95 cả hệ thống bị kéo lên khoảng 1.5 giây. Cách an toàn hơn cho SLA là đưa dải đó sang 3a.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 | Vì sao F1 LLM = 0 dù API có trả lời? Chạy lại bằng key local. | `max_completion_tokens=3` cắt content rỗng. Tăng lên 16 thì 4a/4b F1 SUB = 1.000, 4c = 0.974, 50 case khó = 1.000, invalid=0. | Đối chiếu raw chỉ còn `spam`/`ham`. Giữ chữ ký `call_openai`. |
| Báo cáo | Điền báo cáo và sửa câu "khớp dự đoán", "LLM lỗi rất cao". Bỏ các mục đã nêu, kể cả thử thách #6. | Viết lại kết luận theo số trong `lab1_report.json`. | Không viết đáp án các mục bỏ qua. Không đưa API key vào bài. |
| Phần 1 | Giải thích cơ chế Bag-of-Words (BoW) nhị phân và cách chọn kích thước từ điển vocab? | BoW biến mỗi câu thành vector 0/1 dựa trên sự xuất hiện của từ trong từ điển. Kích thước vocab (vd: 2000 từ phổ biến nhất) giúp loại bỏ từ hiếm/nhiễu và khống chế số chiều, nhưng bỏ qua thứ tự từ và ngữ cảnh. | Tự viết hàm `build_vocab` và `vectorize` bằng numpy (không dùng sklearn), chạy qua test case `_t()` ở TODO 1.1 đạt ✅. |
| Phần 1 | Bản chất toán học của hàm Sigmoid là gì? Vì sao đạo hàm BCE kết hợp Sigmoid lại ra dạng gọn (p - y)? | Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ nén log-odds về khoảng xác suất (0, 1). Khi kết hợp với hàm Binary Cross-Entropy, đạo hàm theo z triệt tiêu thành $p - y$, giúp công thức cập nhật gradient descent $w = w - \text{lr} \cdot X^T(p - y)/n$ trực quan và ổn định. | Tự kiểm tra lại đạo hàm trên giấy nháp, cài đặt hàm `sigmoid` và `train_logreg` ở TODO 1.2, 1.3 và kiểm tra loss giảm đều. |
| Phần 1 | Confusion matrix gồm những gì và vì sao dữ liệu lệch (87% ham - 13% spam) thì không dùng Accuracy? | Ma trận nhầm lẫn gồm TP, FP, FN, TN. Với tập dữ liệu mất cân bằng, mô hình đoán toàn `ham` vẫn đạt 87% accuracy nhưng recall spam bằng 0. Cần dùng F1-score (trung bình điều hòa giữa Precision và Recall) để đo đúng năng lực lọc spam. | Viết hàm `metrics()` tính TP, FP, FN, TN, precision, recall, F1 ở TODO 1.4; kiểm chứng các test case `_t()` đạt ✅. |
| Phần 2 | Vì sao mạng MLP cần lớp phi tuyến ReLU? Nếu bỏ ReLU giữa 2 lớp Linear thì mạng trở thành gì? | Nếu không có hàm phi tuyến, tích 2 biến đổi tuyến tính $W_2(W_1 x + b_1) + b_2$ tương đương một phép biến đổi tuyến tính đơn $W'x + b'$ (chính là hồi quy tuyến tính). Mạng cần ReLU để bẻ cong không gian, học được tương tác phi tuyến giữa các từ. | Cài đặt `build_mlp` có `nn.ReLU` ở TODO 2.1, chạy Thí nghiệm 2.B bỏ ReLU để đối chiếu F1 thấy tương đương với Logistic Regression. |
| Phần 2 | Ý nghĩa 5 dòng huấn luyện PyTorch (`zero_grad`, forward, loss, backward, step)? Quên `zero_grad()` thì sao? | `zero_grad()` xóa gradient cũ; `model(xb)` tính dự đoán; `loss_fn` tính lỗi; `loss.backward()` tính đạo hàm; `step()` cập nhật trọng số. Quên `zero_grad()` thì gradient các batch bị cộng dồn, bước cập nhật đi sai hướng làm loss phân kỳ. | Hoàn thiện hàm `train_mlp` ở TODO 2.2, chạy kiểm tra `_t()` đạt ✅, quan sát loss giảm đều qua các epoch. |
| Phần 2 | Làm sao nhận biết overfitting qua biểu đồ Train/Val Loss? Điểm dừng tối ưu (early stopping) ở đâu? | Overfitting xảy ra khi Train Loss liên tục giảm nhưng Val Loss chững lại rồi bắt đầu tăng lên (mô hình học vẹt tập train). Điểm dừng tối ưu là epoch ngay trước khi Val Loss bắt đầu tăng (ở bài này quanh epoch 4–5). | Quan sát biểu đồ 30 epoch ở Thí nghiệm 2.A, thấy Val Loss tăng từ epoch 5, điền câu trả lời dừng ở epoch 4 cho Câu hỏi 2.A. |
| Phần 3 | Nguyên lý phân loại của kNN trong không gian embedding? Nhược điểm của kNN khi có 10 triệu tin nhắn? | kNN không cần train; nó tính cosine similarity từ vector tin mới đến toàn bộ mẫu train rồi cho $k$ hàng xóm gần nhất bỏ phiếu. Nhược điểm: độ phức tạp suy luận $O(N \cdot d)$, cực kỳ tốn bộ nhớ và chậm khi $N$ lớn, không đạt SLA nếu không có vector index (FAISS). | Viết `cosine_sim`, `top_k_similar`, `knn_predict` ở TODO 3.1–3.3; đối chiếu độ trễ 3a (22.7 ms/tin, chậm hơn rất nhiều so với BoW 0.08 ms). |
| Phần 5 | Các chỉ số p50, p95, p99 latency có ý nghĩa gì? Vì sao hệ thống thực tế không dùng thời gian trung bình (mean)? | p50 là trung vị (50% request xong trước mốc này), p95/p99 là mốc 95% và 99% request nhanh nhất. Thời gian trung bình (mean) dễ bị bóp méo bởi vài request đột biến (outliers). SLA thực tế luôn dùng p95/p99 để cam kết trải nghiệm người dùng không bị treo lâu. | Đối chiếu cột p95 ms/tin ở bảng tổng kết: các mô hình cục bộ đều < 40 ms (đạt SLA 50 ms), còn LLM p95 > 1.1s đều trượt SLA. |

Điều rút ra: F1 bằng 0 có thể là lỗi đo (token bị cắt), chưa phải model kém. Sau khi đo đúng, LLM đọc được spam ngụy trang nhưng p95 khoảng 1.5 giây nên không dùng làm bộ lọc duy nhất cho 10 triệu tin/ngày. Trên SUB 40 tin, nhiều cách cùng F1 1.000; một tin sai đã kéo few-shot xuống 0.974, nên không xếp hạng sát nút bằng cột đó.
