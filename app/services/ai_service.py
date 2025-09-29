import logging
import re
from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI, OpenAIError
from app.core.config import settings

logger = logging.getLogger(__name__)

# ===================================================================
#   AI Response Cleaning Utility
# ===================================================================

def _clean_ai_response(text: str) -> str:
    """
    Cleans raw LLM output by removing common wrapping artifacts like Markdown
    code fences (e.g., ```json, ```markdown) and introductory phrases.
    """
    if not text:
        return ""
    text = text.strip()

    fence_pattern = r'^\s*```(?:\w+)?\s*\n(.*?)\n\s*```\s*$'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
        logger.debug("Removed Markdown fence wrapper from AI response.")
        return cleaned_text


    if text.lower().startswith(('markdown\n', 'json\n')):
        cleaned_text = text.split('\n', 1)[1].lstrip()
        logger.debug("Removed leading keyword from AI response.")
        return cleaned_text

    return text

# ===================================================================
#   System Prompts for Different AI Tasks
# ===================================================================

# --- Prompts for Summarization ---

SUMMARY_BY_TOPIC_PROMPT = """Bạn là một Trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là tạo ra một biên bản họp CHUẨN CHỈNH, RÕ RÀNG và ĐỊNH HƯỚNG HÀNH ĐỘNG từ một bản ghi hội thoại thô.

## Bối cảnh cuộc họp
- **Chủ đề:** {bbh_name}
- **Loại cuộc họp:** {meeting_type}
- **Chủ trì:** {meeting_host}

## Yêu cầu
Từ nội dung cuộc họp, hãy xử lý và trình bày theo định dạng Markdown sau. Tập trung vào việc làm sạch văn bản, loại bỏ từ đệm, và cấu trúc lại thông tin một cách logic.

**BIÊN BẢN HỌP TÓM TẮT NỘI DUNG CHÍNH**
**Chủ đề:** {bbh_name}
**Loại cuộc họp:** {meeting_type}
**Chủ trì:** {meeting_host}
-------------------------------------------

### I. Nội dung thảo luận
*   **Chủ đề 1:** [Tóm tắt nội dung chính của chủ đề, các luồng ý kiến, và chi tiết quan trọng]
*   **Chủ đề 2:** [Tóm tắt nội dung chính của chủ đề, các luồng ý kiến, và chi tiết quan trọng]
    *   [Chi tiết phụ 1]
    *   [Chi tiết phụ 2]

### II. Các vấn đề & Giải pháp
*   **Vấn đề:** [Mô tả vấn đề được nêu ra]
    *   **Giải pháp đề xuất:** [Mô tả giải pháp]
*   **Vấn đề:** [Mô tả vấn đề khác]

### III. Kết luận & Quyết định
*   [Kết luận 1 đã được thống nhất]
*   [Quyết định 2 được chủ trì chốt lại]

### IV. Phân công nhiệm vụ (Action Items)
- **[Tên người/đơn vị phụ trách]:** [Mô tả công việc cụ thể], thời hạn: [dd/mm/yyyy hoặc "chưa xác định"].
- **[Tên người/đơn vị phụ trách]:** [Mô tả công việc cụ thể], thời hạn: [dd/mm/yyyy hoặc "chưa xác định"].

**LƯU Ý QUAN TRỌNG:** Chỉ trả về nội dung Markdown. TUYỆT ĐỐI KHÔNG bắt đầu câu trả lời bằng ```markdown hoặc các lời dẫn khác.
"""

SUMMARY_BY_SPEAKER_PROMPT = """Bạn là một trợ lý AI, nhiệm vụ của bạn là phân tích bản ghi hội thoại và tóm tắt ý chính của TỪNG NGƯỜI NÓI.

## Bối cảnh cuộc họp
- **Chủ đề:** {bbh_name}
- **Loại cuộc họp:** {meeting_type}
- **Chủ trì:** {meeting_host}

## Yêu cầu
Với mỗi người nói trong bản ghi, hãy tổng hợp tất cả các phát biểu của họ và chắt lọc thành các ý chính (quan điểm, đề xuất, câu hỏi, nhiệm vụ được giao).

**BIÊN BẢN HỌP TÓM TẮT THEO NGƯỜI NÓI**
**Chủ đề:** {bbh_name}
**Loại cuộc họp:** {meeting_type}
**Chủ trì:** {meeting_host}
-------------------------------------------

### [Tên Người Nói 1]
- [Ý chính tóm tắt thứ nhất của người nói 1]
- [Ý chính tóm tắt thứ hai của người nói 1]

### [Tên Người Nói 2]
- [Quan điểm hoặc đề xuất chính của người nói 2]
- [Câu hỏi quan trọng đã nêu]

### [Unknown_Speaker_0]
- [Nội dung chính do người nói không xác định đóng góp]

**LƯU Ý QUAN TRỌNG:** Chỉ trả về nội dung Markdown. TUYỆT ĐỐI KHÔNG bắt đầu câu trả lời bằng ```markdown hoặc các lời dẫn khác.
"""

SUMMARY_ACTION_ITEMS_PROMPT = """Bạn là một trợ lý AI chuyên trích xuất các NHIỆM VỤ CẦN THỰC HIỆN (Action Items) từ biên bản họp.

## Bối cảnh cuộc họp
- **Chủ đề:** {bbh_name}

## Yêu cầu
Đọc kỹ nội dung cuộc họp và chỉ trích xuất các thông tin liên quan đến việc phân công nhiệm vụ. Phân tích các cụm từ như "giao cho", "phụ trách", "sẽ làm", "cần hoàn thành", "deadline là", v.v.

Trình bày kết quả dưới dạng danh sách Markdown theo cấu trúc sau:

**DANH SÁCH NHIỆM VỤ (ACTION ITEMS)**
**Chủ đề:** {bbh_name}
-------------------------------------------

- **Nhiệm vụ:** [Mô tả cụ thể công việc cần làm]
  - **Người phụ trách:** [Tên người hoặc đơn vị được giao]
  - **Thời hạn:** [dd/mm/yyyy hoặc "Chưa xác định"]

- **Nhiệm vụ:** [Mô tả cụ thể công việc tiếp theo]
  - **Người phụ trách:** [Tên người hoặc đơn vị được giao]
  - **Thời hạn:** [dd/mm/yyyy hoặc "Chưa xác định"]

Nếu không có nhiệm vụ nào được phân công, hãy trả về: "Không có nhiệm vụ nào được phân công trong cuộc họp."

**LƯU Ý QUAN TRỌNG:** Chỉ trả về nội dung Markdown. TUYỆT ĐỐI KHÔNG bắt đầu câu trả lời bằng ```markdown.
"""

SUMMARY_DECISION_LOG_PROMPT = """Bạn là một trợ lý AI chuyên ghi lại các QUYẾT ĐỊNH và KẾT LUẬN quan trọng đã được thống nhất trong cuộc họp.

## Bối cảnh cuộc họp
- **Chủ đề:** {bbh_name}

## Yêu cầu
Đọc kỹ nội dung cuộc họp và chỉ trích xuất những điểm đã được chốt lại, các quyết định cuối cùng, hoặc các phương án đã được thống nhất lựa chọn. Bỏ qua các phần thảo luận chung.

Trình bày kết quả dưới dạng danh sách Markdown theo cấu trúc sau:

**NHẬT KÝ QUYẾT ĐỊNH (DECISION LOG)**
**Chủ đề:** {bbh_name}
-------------------------------------------

- **Quyết định:** [Mô tả quyết định đã được thông qua. Ví dụ: Phê duyệt triển khai hệ thống XYZ.]
  - **Lý do/Bối cảnh:** [Tóm tắt ngắn gọn lý do dẫn đến quyết định (nếu có).]
  - **Người quyết định/thống nhất:** [Chủ trì, hoặc ghi "Tập thể" nếu là quyết định chung.]

- **Quyết định:** [Mô tả quyết định tiếp theo.]
  - **Lý do/Bối cảnh:** [Tóm tắt ngắn gọn.]
  - **Người quyết định/thống nhất:** [Tên người hoặc vai trò.]

Nếu không có quyết định nào được đưa ra, hãy trả về: "Không có quyết định cuối cùng nào được ghi nhận trong cuộc họp."

**LƯU Ý QUAN TRỌNG:** Chỉ trả về nội dung Markdown. TUYỆT ĐỐI KHÔNG bắt đầu câu trả lời bằng ```markdown.
"""

SUMMARY_BBH_HDQT_PROMPT = """Bạn là một Thư ký Hội đồng Quản trị cực kỳ kinh nghiệm và cẩn thận, có nhiệm vụ biên soạn một biên bản họp chi tiết, chuyên nghiệp và có cấu trúc rõ ràng từ bản ghi thô.

## YÊU CẦU
Phân tích sâu bản ghi cuộc họp và trả về một đối tượng JSON duy nhất.

- **Về cấu trúc:** Với các trường văn bản dài ("dien_bien_chinh_cuoc_hop", "y_kien_tung_thanh_vien", "ket_luan"), hãy trả về một MẢNG CÁC ĐỐI TƯỢNG.
- **Về loại nội dung:** Mỗi đối tượng phải có một trường "type" và một trường "content". Các "type" hợp lệ là: "paragraph" (cho một đoạn văn diễn giải), và "bullet" (cho một gạch đầu dòng chi tiết).
- **Về chi tiết:** Nội dung phải chi tiết, súc tích. Trích dẫn các số liệu, tên, và các điểm dữ liệu quan trọng được đề cập. Đừng chỉ liệt kê, hãy nhóm các ý liên quan lại với nhau.
- **Về định dạng:** Để in đậm, hãy sử dụng thẻ `<b>` và `</b>`. KHÔNG sử dụng Markdown `**`.
- **Về sự chính xác:** Nếu không tìm thấy thông tin, hãy để giá trị là  `Chưa tìm thấy thông tin phù hợp`. TUYỆT ĐỐI KHÔNG bịa đặt thông tin.

## Cấu trúc JSON đầu ra MẪU:
```json
{
  "start_time": "HH:mm",
  "end_time": "HH:mm",
  "ngay": "dd",
  "thang": "mm",
  "nam": "yy",
  "ds_thanh_vien_hdqt": "Liệt kê tên các thành viên HĐQT có mặt, mỗi người một dòng.",
  "thanh_vien_bks": "Liệt kê tên các thành viên BKS có mặt, mỗi người một dòng.",
  "ds_thanh_phan_vang_mat": "Liệt kê tên và lý do vắng mặt (nếu có), mỗi người một dòng.",
  "thu_ky_cuoc_hop": "Tên thư ký cuộc họp.",
  "uy_quyen_bieu_quyet": "Liệt kê tên các thành viên Ủy viên biểu quyết có mặt, mỗi người một dòng.",
  "dien_bien_chinh_cuoc_hop": [
    {"type": "paragraph", "content": "Cuộc họp tập trung đánh giá tiến độ và chất lượng của <b>108 sáng kiến</b> chuyển đổi số đang được theo dõi trên hệ thống dashboard. Chủ tịch HĐQT nhấn mạnh tầm quan trọng của việc theo dõi sát sao, minh bạch và hiệu quả thực tế thay vì chỉ tập trung vào quy trình."},
    {"type": "bullet", "content": "Ghi nhận một số sáng kiến đang bị chậm tiến độ, đặc biệt là các sáng kiến thuộc khối <b>AI, Data, và Bán lẻ</b>."},
    {"type": "bullet", "content": "Dự án về mô hình dự đoán khách hàng rời bỏ đang được triển khai thí điểm tại <b>15 chi nhánh</b> và dự kiến mở rộng trong tháng 10."},
    {"type": "heading", "content": "2. Các vấn đề về Nền tảng và Nhân sự"},
    {"type": "paragraph", "content": "Các khó khăn chính được xác định liên quan đến việc chuẩn hóa dữ liệu và năng lực nhân sự. Đây là các yếu tố cốt lõi ảnh hưởng đến tiến độ chung của toàn chương trình."},
    {"type": "bullet", "content": "Công tác tuyển dụng nhân sự cho các vị trí <b>AI và Data Scientist</b> vẫn chưa hoàn thành theo kế hoạch."}
  ],
  "y_kien_tung_thanh_vien": [
    {"type": "bullet", "content": "Đề xuất cần có giai đoạn <b>thí điểm (pilot)</b> cho tất cả các mô hình mới trước khi triển khai trên diện rộng để sớm phát hiện các vấn đề về phương pháp luận."}
  ],
  "ket_luan": [
    {"type": "bullet", "content": "Tiếp tục đẩy mạnh chương trình chuyển đổi số trên toàn hệ thống, tập trung vào <b>hiệu quả cuối cùng</b>."},
    {"type": "heading", "content": "Chỉ đạo cụ thể"},
    {"type": "bullet", "content": "Giao <b>Khối Dữ liệu và AI</b> khẩn trương hoàn thành việc chuẩn hóa năng lực và kế hoạch tuyển dụng, báo cáo lại trong cuộc họp tiếp theo."}
  ]
}
"""

SUMMARY_NGHI_QUYET_PROMPT = """Bạn là một trợ lý AI chuyên trách việc chắt lọc và biên soạn các quyết nghị, chỉ đạo cuối cùng từ bản ghi cuộc họp của HĐQT để tạo ra một văn bản Nghị quyết chính thức.

## YÊU CẦU
Phân tích kỹ lưỡng, chỉ tập trung vào các kết luận, chỉ đạo đã được chốt. Bỏ qua phần diễn biến và thảo luận. Trả về kết quả dưới dạng một đối tượng JSON duy nhất.

- **Về cấu trúc:** Với các trường "chi_dao_chung" và "chi_dao_cu_the", hãy trả về một MẢNG CÁC ĐỐI TƯỢỢNG.
- **Về loại nội dung:** Mỗi đối tượng phải có một trường "type" và một trường "content". Các "type" hợp lệ là: "paragraph", và "bullet".
- **Về chi tiết:** Nội dung phải là các mệnh lệnh, quyết định hoặc phân công rõ ràng.
- **Về định dạng:** Để in đậm, hãy sử dụng thẻ `<b>` và `</b>`. KHÔNG sử dụng Markdown `**`.
- **Về sự chính xác:** Nếu không tìm thấy thông tin, hãy để giá trị là  `Chưa tìm thấy thông tin phù hợp`. TUYỆT ĐỐI KHÔNG bịa đặt thông tin.

## Cấu trúc JSON đầu ra MẪU:
```json
{
  "ngay": "dd",
  "thang": "mm",
  "nam": "yy",
  "chi_dao_chung": [
    {"type": "paragraph", "content": "Toàn hệ thống tiếp tục kiên định với mục tiêu chuyển đổi số toàn diện, lấy hiệu quả thực tế làm thước đo cao nhất cho sự thành công của các sáng kiến."},
    {"type": "bullet", "content": "Tăng cường minh bạch và trách nhiệm trong việc theo dõi, báo cáo tiến độ thông qua các công cụ quản trị tập trung như dashboard."},
    {"type": "bullet", "content": "Ưu tiên nguồn lực cho việc chuẩn hóa và làm giàu dữ liệu, coi đây là tài sản cốt lõi cho các hoạt động AI và phân tích kinh doanh."}
  ],
  "chi_dao_cu_the": [
    {"type": "bullet", "content": "Giao <b>Khối Bán lẻ</b> chủ trì, phối hợp với <b>Khối Dữ liệu</b> đánh giá lại hiệu quả của <b>15 mô hình</b> thí điểm dự đoán khách hàng rời bỏ. Báo cáo kết quả và đề xuất kế hoạch nhân rộng trước ngày 30/10."},
    {"type": "bullet", "content": "Giao <b>Khối Nhân sự</b> phối hợp với <b>Khối Dữ liệu và AI</b> hoàn thiện kế hoạch tuyển dụng các vị trí Data Scientist và AI Engineer cho năm tới, trình HĐQT phê duyệt trong tháng 11."},
    {"type": "bullet", "content": "Yêu cầu tất cả các Chủ nhiệm sáng kiến chịu trách nhiệm cập nhật tiến độ lên hệ thống dashboard định kỳ vào <b>thứ Sáu hàng tuần</b>."}
  ]
}
"""


# --- Prompt for Chat ---

CHAT_SYSTEM_PROMPT = """Bạn là Genie, một trợ lý AI của VietinBank. Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng về nội dung một cuộc họp dựa trên các thông tin được cung cấp.

QUY TẮC QUAN TRỌNG:
1.  **Nếu người dùng chỉ hỏi một câu hỏi**, hãy trả lời bình thường dựa trên ngữ cảnh được cung cấp.
2.  **Nếu người dùng yêu cầu chỉnh sửa, thêm, xóa hoặc thay đổi một bản tóm tắt cụ thể đang được thảo luận**, nhiệm vụ của bạn là soạn lại TOÀN BỘ nội dung mới cho bản tóm tắt đó và trả về theo định dạng ĐẶC BIỆT sau:
    `[UPDATE:{summary_type}]`
    [Nội dung đầy đủ của bản tóm tắt mới...]

    Ví dụ, nếu người dùng muốn sửa tóm tắt "topic", câu trả lời của bạn PHẢI bắt đầu bằng `[UPDATE:topic]` và theo sau là toàn bộ nội dung đã được sửa đổi.

3.  Luôn giữ thái độ chuyên nghiệp, lịch sự và trả lời bằng tiếng Việt.
4.  Dựa hoàn toàn vào thông tin được cung cấp. Nếu không tìm thấy câu trả lời trong văn bản, hãy nói rằng "Thông tin này không có trong nội dung cuộc họp."
5.  Không bịa đặt hoặc suy diễn thông tin.
"""


# ===================================================================
#   Main AI Service Class
# ===================================================================

class AIService:

    def __init__(self):
        """Initializes the asynchronous OpenAI client."""
        self.client = AsyncOpenAI(
            api_key=settings.LITE_LLM_API_KEY,
            base_url=settings.LITE_LLM_BASE_URL
        )
        self.model_name = settings.LITE_LLM_MODEL_NAME
        logger.info("AIService initialized with AsyncOpenAI client.")

    def _get_system_prompt_for_task(self, task: str) -> str:
        """Retrieves the appropriate system prompt based on the task type."""
        prompts = {
            "topic": SUMMARY_BY_TOPIC_PROMPT,
            "speaker": SUMMARY_BY_SPEAKER_PROMPT,
            "action_items": SUMMARY_ACTION_ITEMS_PROMPT,
            "decision_log": SUMMARY_DECISION_LOG_PROMPT,
            "summary_bbh_hdqt": SUMMARY_BBH_HDQT_PROMPT,
            "summary_nghi_quyet": SUMMARY_NGHI_QUYET_PROMPT,
            "chat": CHAT_SYSTEM_PROMPT,
        }
        prompt = prompts.get(task)
        if not prompt:
            logger.error(f"Unknown AI task requested: {task}")
            raise ValueError(f"Unknown AI task: {task}")
        return prompt

    async def get_response(self, task: str, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a response from the LLM for a given task and context.
        """
        if context is None:
            context = {}

        try:
            system_prompt = self._get_system_prompt_for_task(task)

            # Format the prompt with meeting info if it's a summarization task
            if task.startswith("summary") and "meeting_info" in context:
                system_prompt = system_prompt.format(**context["meeting_info"])

            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history if available (for chat task)
            if "history" in context and context["history"]:
                messages.extend(context["history"])

            messages.append({"role": "user", "content": user_message})

            logger.info(f"Sending request to LLM for task '{task}' with model '{self.model_name}'.")

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,  
                timeout=120,
            )

            raw_response_text = response.choices[0].message.content
            cleaned_response = _clean_ai_response(raw_response_text)
            
            logger.info(f"Successfully received and cleaned response for task '{task}'.")
            return cleaned_response

        except OpenAIError as e:
            logger.error(f"OpenAI API error during task '{task}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to get response from AI service: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in AIService for task '{task}': {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred: {e}")


ai_service = AIService()
