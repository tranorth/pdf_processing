# tools/schemas.py

from typing import Type, List
from pydantic.v1 import BaseModel, Field

class PDFPathSchema(BaseModel):
    """Input for the tool that needs a PDF file path."""
    pdf_path: str = Field(description="The full file path to the PDF document.")

class TableListSchema(BaseModel):
    """Input for the tool that operates on a list of tables."""
    tables: List[str] = Field(description="A list of tables, as raw strings, to select from.")

class RawTableSchema(BaseModel):
    """Input for the tool that processes the single selected table."""
    raw_table_data: str = Field(description="A single, selected table in its raw string format.")