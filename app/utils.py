import logging
from io import BytesIO

try:
    import pypandoc
    PANDOC_AVAILABLE = True
except ImportError:
    PANDOC_AVAILABLE = False
    
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

logger = logging.getLogger(__name__)
from docxtpl import DocxTemplate, RichText

# ===================================================================
#   Document Conversion Utility
# ===================================================================


def add_markdown_to_doc(doc, markdown_text: str, default_style='Normal', bullet_style='List Bullet') -> RichText:
    """
    Converts a Markdown string into a RichText object for docxtpl,
    applying specific named styles from the Word document.

    Args:
        doc: The DocxTemplate object (needed for style context).
        markdown_text: The string containing Markdown.
        default_style (str): The name of the style for regular paragraphs.
        bullet_style (str): The name of the style for list items.

    Returns:
        RichText: The formatted object ready for rendering.
    """
    rt = RichText()
    lines = markdown_text.strip().split('\n')
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue

        # Check if the line is a list item
        is_bullet = False
        if line_strip.startswith(('* ', '- ')):
            is_bullet = True
            # Remove the markdown for the bullet
            content = line_strip[2:]
        else:
            content = line_strip

        # Split by bold tags to handle bolding within the line
        parts = re.split(r'(\*\*.*?\*\*)', content)
        
        # Add a temporary flag to apply style after adding content
        paragraph_added = False
        
        for part in parts:
            if not part: continue
            paragraph_added = True
            if part.startswith('**') and part.endswith('**'):
                rt.add(part[2:-2], bold=True)
            else:
                rt.add(part)
        
        # Add the paragraph break and apply the correct style
        if paragraph_added:
            rt.add('\n')
            if is_bullet:
                rt.paragraphs[-1].style = bullet_style
            else:
                rt.paragraphs[-1].style = default_style
            
    return rt


def create_meeting_minutes_doc_buffer(markdown_content: str) -> BytesIO:
    """
    Converts a Markdown string into an in-memory DOCX file buffer.
    """
    if PANDOC_AVAILABLE:
        try:
            logger.info("Attempting DOCX conversion with Pandoc...")
            docx_bytes = pypandoc.convert_text(
                source=markdown_content,
                to='docx',
                format='md',
                outputfile=None  
            )
            buffer = BytesIO(docx_bytes)
            logger.info("Pandoc conversion successful!")
            buffer.seek(0) 
            return buffer
        except Exception as e:
            logger.error(f"Pandoc conversion failed: {e}. Falling back to basic DOCX generation.", exc_info=True)
            

    # --- Fallback Method ---
    logger.info("Using basic docx library for conversion.")
    try:
        document = Document()
        
        style = document.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)

        for line in markdown_content.split('\n'):
            
            if line.startswith('# '):
                
                document.add_heading(line.lstrip('# ').strip(), level=1)
            elif line.startswith('## '):
                document.add_heading(line.lstrip('## ').strip(), level=2)
            elif line.startswith('### '):
                document.add_heading(line.lstrip('### ').strip(), level=3)
            elif line.strip().startswith(('* ', '- ')):
                document.add_paragraph(line.lstrip('*- ').strip(), style='List Bullet')
            else:
                document.add_paragraph(line)

        buffer = BytesIO()
        document.save(buffer)
        buffer.seek(0) 
        logger.info("Basic DOCX generation successful.")
        return buffer
    except Exception as fallback_e:
        logger.critical(f"FATAL: Basic DOCX conversion also failed: {fallback_e}", exc_info=True)
        raise RuntimeError("Failed to generate DOCX document using all available methods.")
