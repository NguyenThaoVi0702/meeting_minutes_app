import logging
import re
from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI, OpenAIError
from app.core.config import settings
from app.core.ai_prompts import (
  SUMMARY_BY_TOPIC_PROMPT, SUMMARY_BY_SPEAKER_PROMPT, SUMMARY_ACTION_ITEMS_PROMPT,
  SUMMARY_DECISION_LOG_PROMPT, SUMMARY_BBH_HDQT_PROMPT, SUMMARY_NGHI_QUYET_PROMPT,
  CHAT_SYSTEM_PROMPT, INTENT_ANALYSIS_PROMPT
)
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
            "intent_analysis": INTENT_ANALYSIS_PROMPT,
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
            if task in ["topic", "speaker"] and "meeting_info" in context:
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
