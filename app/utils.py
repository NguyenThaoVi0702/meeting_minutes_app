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

logger = logging.getLogger(__name__)

# ===================================================================
#   Document Conversion Utility
# ===================================================================

def create_meeting_minutes_doc_buffer(markdown_content: str) -> BytesIO:
    """
    Converts a Markdown string into an in-memory DOCX file buffer.

    This function prioritizes using Pandoc for a high-fidelity conversion
    that correctly interprets Markdown syntax (headings, bold, lists).

    If Pandoc is not installed or fails for any reason, it falls back to a
    basic conversion method that writes the content into a standard
    Word document.

    Args:
        markdown_content (str): The string containing the meeting summary in Markdown format.

    Returns:
        BytesIO: An in-memory binary buffer containing the generated .docx file.
    
    Raises:
        RuntimeError: If both Pandoc and the fallback method fail to generate the document.
    """
    if PANDOC_AVAILABLE:
        try:
            logger.info("Attempting DOCX conversion with Pandoc...")
            # Convert text to DOCX bytes in memory
            docx_bytes = pypandoc.convert_text(
                source=markdown_content,
                to='docx',
                format='md',
                outputfile=None  # Ensures output is returned as bytes
            )
            buffer = BytesIO(docx_bytes)
            logger.info("Pandoc conversion successful!")
            buffer.seek(0) # Rewind the buffer to the beginning for reading
            return buffer
        except Exception as e:
            logger.error(f"Pandoc conversion failed: {e}. Falling back to basic DOCX generation.", exc_info=True)
            # Proceed to the fallback method below

    # --- Fallback Method ---
    logger.info("Using basic docx library for conversion.")
    try:
        document = Document()
        
        # You can add custom styling for the fallback document here
        # For example, set a default font
        style = document.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)

        # Simple logic to handle basic Markdown: split by lines and add paragraphs
        # This won't render complex markdown but is a safe fallback.
        for line in markdown_content.split('\n'):
            # A very basic attempt to handle headings
            if line.startswith('# '):
                # Using a built-in style for headings
                document.add_heading(line.lstrip('# ').strip(), level=1)
            elif line.startswith('## '):
                document.add_heading(line.lstrip('## ').strip(), level=2)
            elif line.startswith('### '):
                document.add_heading(line.lstrip('### ').strip(), level=3)
            # A basic attempt to handle lists
            elif line.strip().startswith(('* ', '- ')):
                 # Using a built-in style for lists
                document.add_paragraph(line.lstrip('*- ').strip(), style='List Bullet')
            else:
                document.add_paragraph(line)

        buffer = BytesIO()
        document.save(buffer)
        buffer.seek(0) # Rewind buffer
        logger.info("Basic DOCX generation successful.")
        return buffer
    except Exception as fallback_e:
        logger.critical(f"FATAL: Basic DOCX conversion also failed: {fallback_e}", exc_info=True)
        # If both methods fail, we raise an exception to be caught by the API endpoint
        raise RuntimeError("Failed to generate DOCX document using all available methods.")

# You can add other cross-application utility functions here in the future.
# For example, a function to format timestamps, sanitize filenames, etc.