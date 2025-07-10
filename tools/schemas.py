# tools/schemas.py

from typing import Type, List
from pydantic.v1 import BaseModel, Field

class PDFPathSchema(BaseModel):
    pdf_path: str = Field(description="The full file path to the PDF document.")

class WebSearchSchema(BaseModel):
    query: str = Field(description="A targeted search query for Google.")

class RawCSVSchema(BaseModel):
    """Input for the tool that processes a raw CSV string."""
    raw_csv_data: str = Field(description="A single table represented as a raw CSV string to be formatted.")