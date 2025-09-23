import json
import logging
from io import BytesIO
from pathlib import Path
from docxtpl import DocxTemplate
from app.utils import create_subdoc_from_structured_data

logger = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"

def generate_templated_document(template_type: str, llm_json_output: str) -> BytesIO:
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
        context_data = json.loads(llm_json_output)
        
        render_context = {}
        for key, value in context_data.items():
            if isinstance(value, list):
                render_context[key] = create_subdoc_from_structured_data(doc, value)
            else:
                render_context[key] = value

        logger.info(f"Final render context for '{template_type}': {render_context}")
        
        doc.render(render_context)
        
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    except json.JSONDecodeError:
        logger.error(f"Failed to decode LLM JSON: {llm_json_output}")
        raise ValueError("AI service returned invalid JSON.")
    except Exception as e:
        logger.error(f"Failed to generate document for template '{template_type}': {e}", exc_info=True)
        raise RuntimeError("An unexpected error occurred during document generation.")
