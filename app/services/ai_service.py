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
    # Pattern to find markdown code fences and extract the content within.
    # Handles fences with or without language identifiers (e.g., ```json, ```).
    fence_pattern = r'^\s*```(?:\w+)?\s*\n(.*?)\n\s*```\s*$'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
        logger.debug("Removed Markdown fence wrapper from AI response.")
        return cleaned_text

    # Fallback for cases where only a keyword is present without fences
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

# --- Prompt for Chat ---

CHAT_SYSTEM_PROMPT = """Bạn là Genie, một trợ lý AI của VietinBank. Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng về nội dung một cuộc họp dựa trên các thông tin được cung cấp, bao gồm: bản ghi hội thoại đầy đủ và bản tóm tắt.

Quy tắc ứng xử:
- Luôn giữ thái độ chuyên nghiệp, lịch sự.
- Trả lời bằng tiếng Việt.
- Dựa hoàn toàn vào thông tin được cung cấp. Nếu không tìm thấy câu trả lời trong văn bản, hãy nói rằng "Thông tin này không có trong nội dung cuộc họp."
- Không bịa đặt hoặc suy diễn thông tin.
"""

# ===================================================================
#   Main AI Service Class
# ===================================================================

class AIService:
    """
    A singleton service to handle all interactions with the LLM.
    It is designed to be asynchronous and configured via the central settings.
    """
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
            # Map summary types from the DB/Schema to the correct prompt
            "topic": SUMMARY_BY_TOPIC_PROMPT,
            "speaker": SUMMARY_BY_SPEAKER_PROMPT,
            "action_items": SUMMARY_ACTION_ITEMS_PROMPT,
            "decision_log": SUMMARY_DECISION_LOG_PROMPT,
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

        Args:
            task (str): The type of task to perform (e.g., 'topic', 'chat').
            user_message (str): The primary input text (e.g., full transcript, user chat query).
            context (dict, optional): A dictionary containing additional context, such as:
                - 'meeting_info': Dict with bbh_name, meeting_type, meeting_host.
                - 'history': A list of previous chat turns for conversation history.

        Returns:
            The cleaned, text-only response from the AI model.
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
                temperature=0.2,  # Low temperature for factual, consistent outputs
                timeout=120, # Set a timeout for the API call
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

# Create a single, shared instance of the service for the application to use.
ai_service = AIService()