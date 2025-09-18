import json
import logging
from io import BytesIO
from pathlib import Path

from docxtpl import DocxTemplate
from app.utils import add_markdown_to_doc

logger = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"

def generate_templated_document(template_type: str, llm_json_output: str) -> BytesIO:
    """
    Generates a DOCX file from a template and JSON data from an LLM.

    Args:
        template_type (str): The type of document, e.g., 'bbh_hdqt' or 'nghi_quyet'.
        llm_json_output (str): The JSON string response from the AI service.

    Returns:
        BytesIO: An in-memory buffer containing the generated DOCX file.
    """
    template_map = {
        "bbh_hdqt": "bbh_hdqt_template.docx",
        "nghi_quyet": "nghi_quyet_template.docx",
    }
    template_filename = template_map.get(template_type)
    if not template_filename:
        raise ValueError(f"Invalid template type: {template_type}")

    template_path = TEMPLATES_DIR / template_filename
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found at: {template_path}")

    try:
        doc = DocxTemplate(template_path)
        context = json.loads(llm_json_output)

        # Process fields that contain Markdown into RichText objects
        for key, value in context.items():
            if isinstance(value, str) and value.startswith("[Markdown]"):
                markdown_content = value.replace("[Markdown]", "").strip()
                context[key] = add_markdown_to_doc(
                    doc,
                    markdown_content,
                    default_style='Normal', 
                    bullet_style='bullet'   
                )

        doc.render(context)
        
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    except json.JSONDecodeError:
        logger.error(f"Failed to decode LLM JSON output: {llm_json_output}")
        raise ValueError("AI service returned invalid JSON. Cannot generate document.")
    except Exception as e:
        logger.error(f"Failed to generate document for template '{template_type}': {e}", exc_info=True)
        raise RuntimeError("An unexpected error occurred during document generation.")
