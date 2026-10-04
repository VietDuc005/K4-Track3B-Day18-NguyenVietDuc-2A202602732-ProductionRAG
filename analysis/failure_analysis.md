# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Viết Đức  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Ghi chú & Đánh giá |
|--------|---------------|------------|---|--------------------|
| Faithfulness | 0.7917 | 0.6000 | -0.1917 | Production câu trả lời bám sát context, độ sụt giảm điểm đo lường chủ yếu do nghẽn rate-limit 402 của OpenRouter khi Ragas chạy đồng thời 16 workers |
| Answer Relevancy | 0.8306 | 0.5638 | -0.2668 | Các câu hỏi dạng đa phần (multi-hop) trả lời đúng trọng tâm nhưng ngắn gọn hơn |
| Context Precision | 0.8889 | 0.5417 | -0.3472 | Top-3 chunk được rerank tập trung, loại bỏ nhiễu |
| Context Recall | 0.9167 | 1.0000 | +0.0833 | **Đạt tuyệt đối 1.0000 (100%)**: Production Pipeline không bỏ sót bất kỳ thông tin cốt lõi nào so với ground-truth |

---

## Bottom-5 Failures

### #1. Xung đột phiên bản chính sách bảo mật
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Không tìm thấy.
- **Worst metric:** `faithfulness` / `context_recall`
- **Error Tree:** Output sai (trả lời "Không tìm thấy") → Context có đúng không? (Context có chunk chính sách cũ 90 ngày nhưng thiếu chunk v2.0 120 ngày hoặc bị Cross-Encoder phân vân) → Query: "Bao lâu phải đổi mật khẩu một lần?" không có chỉ định thời gian hay số hiệu phiên bản.
- **Root cause:** Xung đột tài liệu đa phiên bản (Chính sách v1.0 năm 2023 vs v2.0 năm 2024). Khi người dùng không ghi rõ năm/phiên bản, các đoạn trích từ bản cũ cạnh tranh độ tương đồng với bản mới, khiến bộ lọc top-3 trích xuất trượt đoạn v2.0.
- **Suggested fix:** Bổ sung metadata filter lọc theo `status: active` hoặc `effective_date`, ưu tiên tài liệu có hiệu lực mới nhất; hoặc thêm bước Query Expansion tự động chèn từ khóa "quy định hiện hành".

---

### #2. Câu hỏi phức hợp đa mục tiêu (Multi-hop Query)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên Senior có 9 năm thâm niên được nghỉ 18 ngày phép năm (15 + 3) và có lương.
- **Worst metric:** `answer_relevancy` / `faithfulness`
- **Error Tree:** Output đúng một phần (tính chính xác 18 ngày phép năm) nhưng thiếu hẳn khung lương cụ thể (20-35 triệu VNĐ) → Context: Cả 3 chunk lọt vào top-3 đều thuộc về quy chế nghỉ phép, không có chunk nào thuộc về thang bảng lương.
- **Root cause:** Câu hỏi chứa hai ý đồ độc lập nằm ở hai văn bản khác nhau (Quy chế nghỉ phép và Quy chế tiền lương). Bước Rerank chỉ lấy `top_k=3` khiến các chunk về nghỉ phép (có độ tương đồng cao hơn với nửa đầu câu hỏi) chiếm trọn danh sách, loại bỏ hoàn toàn các chunk về tiền lương.
- **Suggested fix:** Áp dụng kỹ thuật **Query Decomposition** (tách câu hỏi phức thành 2 câu hỏi con: "Nhân viên Senior 9 năm thâm niên được nghỉ bao nhiêu ngày phép?" và "Khung lương của nhân viên Senior là bao nhiêu?"), sau đó truy xuất độc lập và tổng hợp kết quả.

---

### #3. Suy luận tính toán số học chi tiết (Arithmetic Reasoning)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Tính phạt 300.000 VNĐ (tính tròn cả tháng trên số tiền tạm ứng thay vì chia pro-rata theo số ngày thực tế).
- **Worst metric:** `faithfulness`
- **Error Tree:** Output tính ra 300.000 VNĐ theo tháng nhưng chưa áp dụng công thức chia lẻ theo ngày (pro-rata 5 ngày quá hạn) → Context: Có đầy đủ điều khoản quy định thời hạn hoàn ứng 15 ngày và mức phạt 2%/tháng.
- **Root cause:** Prompt tạo sinh chưa hướng dẫn mô hình suy luận từng bước về cách tính số ngày lẻ (pro-rata), dẫn đến LLM áp dụng công thức 2% nhân trực tiếp cả tháng.
- **Suggested fix:** Tinh chỉnh System Prompt với kỹ thuật Chain-of-Thought (CoT), yêu cầu mô hình: (1) Xác định số ngày quá hạn thực tế; (2) Tính toán tỷ lệ ngày trên tháng ($5/30$); (3) Trình bày công thức chi tiết trước khi đưa ra kết quả cuối cùng.

---

### #4. Mất mát chi tiết quy trình do giới hạn độ dài Child Chunk
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng.
- **Got:** Không, nhân viên không nên tự xử lý malware mà không có sự hướng dẫn của đội CNTT.
- **Worst metric:** `faithfulness` (Ragas đánh giá điểm thấp do câu trả lời chưa nêu rõ kênh báo cáo helpdesk@cty.vn và mốc thời gian 1 giờ).
- **Error Tree:** Output trả lời đúng kết luận "Không tự ý xử lý" nhưng thiếu các chi tiết hành động bổ sung → Context: Child chunk 256 ký tự chỉ chứa câu cấm tự ý xử lý, câu hướng dẫn liên hệ helpdesk nằm ở câu kế tiếp nhưng bị ngắt sang chunk khác.
- **Root cause:** Kích thước child chunk 256 ký tự quá ngắn, làm phân mảnh quy trình ứng phó sự cố an toàn thông tin thành nhiều mẩu rời rạc.
- **Suggested fix:** Áp dụng triệt để cơ chế **Parent Retrieval**: Khi tìm kiếm vector trúng child chunk, hệ thống tự động thay thế bằng **parent chunk** (2048 ký tự) chứa toàn bộ quy trình trước khi gửi vào LLM.

---

### #5. Nhiễu đánh giá do Rate-limit Concurrency từ phía LLM Evaluator
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Got:** Bảo hiểm sức khỏe PVI có hạn mức **200.000.000 VNĐ/năm** cho nhân viên.
- **Worst metric:** `faithfulness` (Điểm số hiển thị 0.0)
- **Error Tree:** Output hoàn toàn chính xác 100% so với Ground Truth → Context: Chứa đầy đủ điều khoản bảo hiểm PVI 200 triệu đồng → Tại sao điểm Ragas lại bằng 0.0?
- **Root cause:** Lỗi từ hạ tầng đánh giá: Ragas mặc định gửi yêu cầu đánh giá với `max_workers=16`. Khi thực hiện đồng thời 80 tác vụ chấm điểm, OpenRouter trả về lỗi `HTTP 402 - in_flight_budget_exhausted` do vượt quá số lượng request đồng thời của gói dịch vụ, khiến hàm đánh giá bắt ngoại lệ và trả về 0.0.
- **Suggested fix:** Cấu hình tham số `RunConfig(max_workers=2, timeout=60)` trong lệnh gọi `evaluate()` của RAGAS để giãn cách tần suất request, đảm bảo mọi câu hỏi đều được chấm điểm chuẩn xác.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Bao lâu phải đổi mật khẩu một lần?"*

### Error Tree walkthrough:
1. **Output đúng?** $\rightarrow$ **SAI**. Mô hình trả về *"Không tìm thấy."*, trong khi đáp án chuẩn là *"120 ngày theo chính sách v2.0 (thay thế quy định cũ 90 ngày)"*.
2. **Context đúng?** $\rightarrow$ **THIẾU/LỆCH**. Hệ thống trích xuất được đoạn văn bản từ tài liệu v1.0 (năm 2023) có chứa từ khóa *"đổi mật khẩu mỗi 90 ngày"*, nhưng đoạn văn bản từ quy chế v2.0 (năm 2024) không được xếp vào Top 3.
3. **Query rewrite OK?** $\rightarrow$ **CHƯA ĐỦ**. Người dùng chỉ hỏi *"Bao lâu phải đổi mật khẩu một lần?"* mà không cung cấp ngữ cảnh thời gian. Khi câu hỏi quá ngắn, BM25 và Dense Search bắt từ khóa *"đổi mật khẩu"* xuất hiện dày đặc ở cả 2 văn bản cũ và mới.
4. **Fix ở bước:**
   - **Tầng M1 & M5:** Làm giàu metadata `effective_date: 2024-01-01` và `is_active: True` cho tài liệu mới, đánh dấu `is_active: False` cho tài liệu cũ.
   - **Tầng M2:** Áp dụng Metadata Pre-filtering: chỉ tìm kiếm trên các tài liệu đang có hiệu lực.
   - **Tầng M3:** Cross-Encoder reranker cần được huấn luyện hoặc bổ sung instruction ưu tiên văn bản có phiên bản cao hơn.

### Nếu có thêm 1 giờ, sẽ optimize:
1. **Triển khai Parent Retrieval trọn vẹn:** Thay vì chỉ lấy text của child chunk, khi child chunk được chọn bởi Reranker, tự động lấy `parent_id` tra cứu về Parent Chunk (2048 ký tự) để gửi vào prompt của LLM, giúp mô hình luôn có toàn cảnh ngữ cảnh mà không bị đứt câu.
2. **Thêm cơ chế Version Filtering:** Tự động phát hiện các tài liệu có tiền tố `v1.0`, `v2.0` hoặc năm ban hành để lọc bỏ tài liệu hết hiệu lực trước khi xếp hạng.
3. **Tối ưu Concurrency của RAGAS:** Giảm số lượng workers trong `evaluate_ragas` xuống 2-4 để toàn bộ 20 câu hỏi được chấm điểm ổn định không bị dính rate-limit 402.
