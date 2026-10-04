# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Viết Đức  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code đã viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng regex bóc tách câu tiếng Việt, so sánh cosine similarity giữa embedding câu liên tiếp (`all-MiniLM-L6-v2`) với ngưỡng threshold 0.85 để ngắt đoạn khi người viết chuyển ý, bảo toàn tính liền mạch ngữ nghĩa. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Cắt phân cấp Parent-Child (Parent 2048 ký tự, Child 256 ký tự). Child chunks dùng để tìm kiếm vector chính xác cao, khi trích xuất sẽ mang `parent_id` liên kết về parent chunk để cung cấp ngữ cảnh đầy đủ cho LLM đọc. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Phân tích cấu trúc tiêu đề Markdown (#, ##, ###), trích xuất cây phân cấp section đưa vào metadata, đồng thời prepend tiêu đề vào đầu chunk giúp chunk không bị mất phương hướng. |
| Hybrid search (BM25 + Dense) | M2 | `reciprocal_rank_fusion()`, `BM25Search`, `DenseSearch` | Kết hợp BM25 (xử lý từ ghép tiếng Việt qua `underthesea` và thay thế `_` bằng khoảng trắng) cùng Dense Vector Qdrant (`BAAI/bge-m3`). Hợp nhất bằng RRF với $k=60$ để khắc phục nhược điểm của từng phương pháp. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Tầng lọc thứ hai nhận trực tiếp cặp (query, doc) vào mô hình Transformer `BAAI/bge-reranker-v2-m3` để đối chiếu chéo chi tiết từng từ, sắp xếp lại top 20 candidates xuống top 3 tinh hoa nhất gửi vào LLM. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đánh giá tự động chất lượng RAG qua 4 chỉ số cốt lõi: Faithfulness (độ trung thực), Answer Relevancy (độ bám sát câu hỏi), Context Precision (độ chính xác thứ hạng), Context Recall (độ bao phủ thông tin). |
| Diagnostic Tree failure analysis | M4 | `failure_analysis()` | Phân tích nguyên nhân lỗi dựa trên chỉ số thấp nhất (worst metric) trong 4 metric của RAGAS, tự động gán chẩn đoán (diagnosis) và hướng khắc phục (suggested fix). |
| AI Enrichment (HyQA & Contextual) | M5 | `_enrich_single_call()` / `enrich_chunks()` | Tối ưu chi phí và độ trễ bằng 1 API call duy nhất mỗi chunk đến `gpt-4o-mini`, cùng lúc tạo tóm tắt (Summary), câu hỏi giả định (HyQA), tiền tố ngữ cảnh (Contextual Prepend) và siêu dữ liệu tự động (Auto Metadata). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi 1 (Môi trường Windows & Python 3.13 Build Tools):**
  - *Lỗi gặp phải:* Khi cài đặt gói `numpy 1.26.4` trên môi trường Python 3.13 của Windows, tiến trình cài đặt thất bại với mã lỗi `error: subprocess-exited-with-error` do không tìm thấy Microsoft Visual C++ 14.0 để biên dịch từ C source.
  - *Nguyên nhân & Cách xử lý:* Python 3.13 còn quá mới và nhiều thư viện AI chưa phát hành pre-built wheel tương thích. Đã chuyển sang sử dụng Python 3.11.9 (`C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`) và tạo môi trường ảo `.venv` hoàn toàn mới. Tất cả các thư viện (`torch`, `sentence-transformers`, `ragas`, `underthesea`, `qdrant-client`) đều có wheel chuẩn và cài đặt thành công 100%.

- **Lỗi 2 (Tách từ ghép tiếng Việt cho BM25 với Underthesea):**
  - *Lỗi gặp phải:* Thư viện `underthesea.word_tokenize(text, format="text")` khi tách từ ghép thường nối các âm tiết bằng dấu gạch dưới (ví dụ: `nghỉ_phép`, `thử_việc`). Tuy nhiên, khi người dùng hoặc test suite nhập câu hỏi dạng văn bản thông thường "nghỉ phép", BM25 tokenizer mặc định xem đây là 2 từ tách biệt và không khớp trúng token `nghỉ_phép`.
  - *Nguyên nhân & Cách xử lý:* Trong hàm `segment_vietnamese()`, sau khi gọi word tokenize, thêm bước `.replace("_", " ")`. Nhờ vậy, câu chữ vẫn được chuẩn hóa hình thái học mà BM25 vẫn so khớp từ khóa một cách tự nhiên.

- **Lỗi 3 (API Deprecation trong Qdrant Client >= 1.9):**
  - *Lỗi gặp phải:* Phương thức `client.search()` bị cảnh báo deprecated hoặc thay đổi chữ ký hàm trong các bản cập nhật mới nhất của `qdrant-client`.
  - *Nguyên nhân & Cách xử lý:* Thay thế bằng API chuẩn mới `client.query_points(collection_name=..., query=query_vector, limit=top_k)` để đảm bảo tính ổn định lâu dài trên chuẩn Production.

- **Lỗi 4 (Xung đột thư viện Tokenizer với FlagEmbedding):**
  - *Lỗi gặp phải:* Gói `FlagEmbedding` (FlagReranker) thường xuyên xung đột với phiên bản mới của HuggingFace `transformers`.
  - *Nguyên nhân & Cách xử lý:* Dùng trực tiếp lớp `CrossEncoder` từ `sentence_transformers` với trọng số `BAAI/bge-reranker-v2-m3`. Đồng thời áp dụng mô hình singleton caching trong `_get_cross_encoder()` để tái sử dụng weights trong RAM, tránh overhead nạp lại 2.2GB giữa các truy vấn.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của tôi:

### Project: Trợ lý Tra cứu Quy chế & Quy trình Vận hành Nội bộ Doanh nghiệp (Enterprise Policy AI Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản: cắt văn bản cố định 500 ký tự (Fixed-size Chunking), nhúng vector bằng mô hình đa ngôn ngữ chung chung và tìm kiếm Dense vector đơn thuần trên Pinecone/ChromaDB.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Đứt đoạn thông tin:* Các bảng biểu mức phụ cấp, điều khoản loại trừ bị cắt ngang chừng dẫn đến LLM trả lời sai lệch hoặc hallucination.
  - *Tìm kiếm trượt từ khóa quy chuẩn:* Các câu hỏi chứa số hiệu văn bản (ví dụ "Nghị định 13", "Thông tư 05") hay thuật ngữ viết tắt (PVI, MFA, VPN) bị tìm trượt do mô hình vector chỉ hiểu ngữ nghĩa chung chung.
  - *Xung đột phiên bản văn bản:* Quy chế năm 2023 và quy chế sửa đổi năm 2024 có nhiều điều khoản tương tự, hệ thống thường trích xuất lẫn lộn văn bản đã hết hiệu lực.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang mô hình kết hợp **Hierarchical Parent-Child Chunking** (Parent: 2048 chars, Child: 256 chars) kết hợp **Structure-aware Chunking** đối với tài liệu có định dạng Markdown/Word. Parent-Child giúp tìm kiếm chính xác từng điều khoản nhỏ nhưng vẫn gửi toàn bộ điều luật đầy đủ vào prompt cho LLM.
2. **Search retrieval:** Áp dụng **Hybrid Search (BM25 + Dense Search BAAI/bge-m3)** hợp nhất bằng **RRF ($k=60$)**. BM25 đảm bảo bắt trọn 100% các mã hiệu quy định, số ngày, số tiền; trong khi Dense Search bao quát các câu hỏi diễn đạt tự nhiên theo phong cách người dùng.
3. **Reranking:** Tích hợp tầng **Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)** để lọc từ Top 20 candidates xuống Top 3 đoạn trích đắt giá nhất. Đoạn trích từ quy chế mới nhất (2024) và điều khoản loại trừ sẽ được xếp hạng chính xác lên Rank 0.
4. **Enrichment:** Sử dụng **Contextual Prepend** để gắn tên tài liệu và vị trí chương mục vào đầu mỗi chunk trước khi nhúng vector, giúp loại bỏ hoàn toàn hiện tượng chunk "vô danh".
5. **Evaluation:** Thiết lập bộ khung đánh giá tự động định kỳ bằng **RAGAS 4 metrics** trên tập câu hỏi vàng (Golden Test Set), sử dụng **Diagnostic Tree** để phát hiện nhanh khâu lỗi (Retrieval hay Generation) trong CI/CD pipeline.

#### 3. Timeline triển khai
- **Tuần 1:** Tái cấu trúc cơ sở dữ liệu quy chế nội bộ với bộ cắt đoạn phân cấp (Parent-Child) và gắn metadata phiên bản văn bản; thiết lập Qdrant vector database cục bộ.
- **Tuần 2:** Xây dựng module Hybrid Search (BM25 Underthesea + BGE-M3) kết hợp Cross-Encoder Reranker; đo kiểm độ trễ đảm bảo dưới 150ms.
- **Tuần 3:** Tích hợp bộ đánh giá RAGAS vào quy trình kiểm thử tự động, chạy benchmark so sánh phiên bản cũ và mới; tinh chỉnh prompt hệ thống dựa trên cây chẩn đoán lỗi.
- **Tuần 4:** Đóng gói REST API bằng FastAPI, triển khai Docker container và tích hợp giao diện tra cứu thử nghiệm cho nhân sự nội bộ.
