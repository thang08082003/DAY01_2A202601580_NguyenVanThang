# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Khi temperature tăng từ 0.0 lên 1.8, phản hồi đi từ chính xác, mang tính sự thật cố định (0.0–0.7) sang sáng tạo, linh hoạt hơn (1.2) và cuối cùng trở nên mất kiểm soát (1.8). Bắt đầu từ mức 1.8, phản hồi trở nên kém mạch lạc rõ rệt, xuất hiện tình trạng lặp từ, xáo trộn ngữ pháp hoặc sinh ra các từ/thông tin vô nghĩa (hallucination)*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Hợp đồng pháp lý: Đặt 0.0 (hoặc tối đa 0.2) để đảm bảo tính chính xác, nhất quán và tuyệt đối không sáng tạo ra các điều khoản sai lệch. Slogan quảng cáo: Đặt 0.7 đến 1.0 để kích thích ý tưởng mới lạ, độc đáo và đa dạng góc nhìn. Khác biệt: Hợp đồng đòi hỏi sự chuẩn xác định hình (deterministic), trong khi slogan cần sự biến hóa và bất ngờ (probabilistic).*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Tổng lượng token đầu ra mỗi ngày 20.000.000$ token (20Mtokens output).(Giả sử mức giá tham chiếu tiêu chuẩn: Model lớn như GPT-4o là $10/1M output tokens, Model nhỏ như GPT-4o-mini là $0.60/1M output tokens)Chi phí model lớn (ví dụ GPT-4o): ~$200 / ngày.Chi phí model nhỏ (ví dụ GPT-4o-mini): ~$12 / ngày (rẻ hơn 16+ lần).Trường hợp model lớn xứng đáng: Phân tích hợp đồng pháp lý phức tạp, chẩn đoán y tế, hoặc viết code hệ thống lõi — những nơi sai sót nhỏ gây hậu quả nghiêm trọng.Trường hợp model nhỏ là lựa chọn đúng: Phân loại phản hồi khách hàng (intent classification), tóm tắt email ngắn, hoặc làm chatbot hỗ trợ các câu hỏi FAQ đơn giản.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Hai phản hồi khác biệt hoàn toàn: System prompt nhà thơ cho ra đoạn văn ngắn, mượt mà, dùng hình ảnh ẩn dụ (như "con trẻ học đi") mà không có thuật ngữ; trong khi system prompt kỹ sư cho phản hồi dài, cấu trúc rõ ràng với khái niệm chính xác (X -> Y) kèm ví dụ code Python. Qua đó, system prompt giúp điều khiển trực tiếp giọng văn (tone), văn phong, cấu trúc đầu ra, độ sâu kỹ thuật và định dạng trình bày của mô hình*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Với tiếng Việt, tiktoken thường trả về số token lớn hơn ước lượng số từ / 0.75 khoảng 50% – 100% (hoặc hơn) do bảng mã BPE phân rã các từ tiếng Việt có dấu thành nhiều sub-word token. Nếu dùng công thức ước lượng thô (số từ / 0.75 - vốn tối ưu cho tiếng Anh), bạn sẽ bị dự toán thiếu ngân sách trầm trọng. Lý do là mô hình tốn nhiều token hơn đáng kể để biểu diễn cùng một lượng từ tiếng Việt.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Chatbot văn bản hưởng lợi nhiều nhất từ streaming vì nó giảm thời gian chờ phản hồi cảm nhận (perceived latency), giúp người dùng đọc ngay từng từ khi LLM đang sinh văn bản. Pipeline dịch tài liệu chạy ngầm ban đêm hoàn toàn không cần streaming vì tiến trình xử lý tự động theo lô (batch job), người dùng không ngồi chờ thời gian thực mà chỉ quan tâm đến kết quả hoàn chỉnh cuối cùng. Trong khi đó, trợ lý giọng nói chỉ cần streaming ở dạng chunk câu hoàn chỉnh để đưa vào bộ đọc Text-to-Speech (TTS) chứ không cần stream từng token lẻ.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Exponential Backoff: Giúp giải tỏa áp lực cho server khi quá tải bằng cách tăng dần thời gian chờ sau mỗi lần thất bại (ví dụ: 1s, 2s, 4s, 8s...), tránh việc hàng nghìn client liên tục dội request làm sập hệ thống. Jitter (Độ trễ ngẫu nhiên): Giải quyết vấn đề Thundering Herd Problem (các client retry đồng loạt tại cùng một mốc thời gian). Bằng cách cộng thêm một khoảng nhiễu ngẫu nhiên vào thời gian chờ, các request retry sẽ được rải đều ra thay vì đánh vào server cùng một lúc.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *Bạn là trợ lý hỗ trợ lập trình Python. (1) Luôn trả lời ngắn gọn, đi thẳng vào giải pháp và chỉ viết code bằng Python 3.10+. (2) Tuyệt đối không đưa ra lời chào hỏi thừa thãi hay giải thích lý thuyết trừ khi được yêu cầu." Hai vị trí quan trọng: Nếu xóa "(1) Luôn trả lời ngắn gọn...": Trợ lý sẽ bắt đầu viết các đoạn văn dài dòng, giải thích lan man trước khi đưa ra đoạn code. Nếu xóa "(2) Tuyệt đối không đưa ra lời chào hỏi...": Trợ lý sẽ quay lại phong cách xã giao thông thường ("Chào bạn! Rất vui được hỗ trợ...", "Hy vọng đoạn code trên giúp ích cho bạn!"), làm tốn token và giảm tốc độ trải nghiệm.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Tình huống mất ngữ cảnh: Người dùng thảo luận một bài toán từ lượt 1 (ví dụ: "Tôi đang dùng dataframe df_sales với các cột A, B, C"). Sau 4 lượt trao đổi tiếp theo về xử lý dữ liệu, ở lượt thứ 6 người dùng hỏi: "Hãy viết hàm group by cho dataframe ở đầu buổi nói chuyện". Giới hạn 4 lượt gần nhất khiến mô hình quên mất df_sales chứa những cột nào và báo lỗi hoặc hỏi lại. Cách khắc phục: Áp dụng kỹ thuật Conversation Summarization (Tóm tắt hội thoại). Khi bộ nhớ vượt quá 4 lượt, dùng một model nhỏ hơn để tóm tắt các thông tin quan trọng (biến, yêu cầu cốt lõi) của các lượt cũ thành một đoạn context_summary ngắn và đính kèm đoạn này vào đầu lịch sử hội thoại gửi lên API.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
