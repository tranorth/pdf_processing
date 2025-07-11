# tools/schemas.py

from typing import Type, List
from pydantic.v1 import BaseModel, Field

class PDFPathSchema(BaseModel):
    """Input for the tool that needs a PDF file path."""
    pdf_path: str = Field(description="The full file path to the PDF document.")

class TableListSchema(BaseModel):
    """Input for the tool that operates on a list of tables."""
    tables_as_csv: List[str] = Field(description="A list of tables, each represented as a raw CSV string, to be analyzed.")

class RawCSVSchema(BaseModel):
    """Input for the tool that processes a single raw CSV string."""
    raw_csv_data: str = Field(description="A single table, represented as a raw CSV string, to be formatted.")

class FinalCSVSchema(BaseModel):
    """Input for the final cleanup tool."""
    generated_csv: str = Field(description="The generated CSV string from the previous step, which may include extra text or markdown.")