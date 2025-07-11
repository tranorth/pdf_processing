# tools/schemas.py

from typing import Type
from pydantic.v1 import BaseModel, Field

class PDFProcessorSchema(BaseModel):
    """Input schema for the PDFProcessorTool."""
    pdf_path: str = Field(description="The full file path to the PDF document to be processed.")