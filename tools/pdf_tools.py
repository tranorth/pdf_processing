# tools/pdf_tools.py

import os
from typing import Type, List

import pandas as pd
import pdfplumber
from langchain.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic.v1 import BaseModel, Field

from .schemas import PDFPathSchema, WebSearchSchema, RawCSVSchema

class PDFTableExtractorTool(StructuredTool):
    """Tool 1: Extracts every table from a PDF, fixing blank headers."""
    name: str = "extract_all_tables_from_pdf"
    description: str = "Reads a PDF file and extracts every table found into a list of clean CSV strings."
    args_schema: Type[BaseModel] = PDFPathSchema

    def _run(self, pdf_path: str, **kwargs: any) -> List[str]:
        if isinstance(pdf_path, dict): path = pdf_path.get('pdf_path')
        else: path = pdf_path
        
        print(f"🔎 Extracting and cleaning all tables from: {path}")
        if not path or not os.path.exists(path):
            return ["Error: PDF path not found or is invalid."]

        all_tables_as_csv = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    for table_data in page.extract_tables():
                        if table_data and len(table_data) > 1:
                            headers = table_data[0]
                            df = pd.DataFrame(table_data[1:], columns=headers)
                            if df.columns[0] is None or 'Unnamed' in str(df.columns[0]):
                                df.rename(columns={df.columns[0]: 'primary_submarket'}, inplace=True)
                            all_tables_as_csv.append(df.to_csv(index=False))
            return all_tables_as_csv
        except Exception as e:
            print(f"Error processing PDF with pdfplumber: {e}")
            return [f"An error occurred during table extraction: {e}"]


class WebSearchTool(StructuredTool):
    """Tool 2: A general web search tool powered by DuckDuckGo."""
    name: str = "web_search"
    description: str = "Use this to search the public web for information. Useful for verifying geographic locations like submarkets or counties."
    args_schema: Type[BaseModel] = WebSearchSchema
    
    # --- THIS IS THE CORRECTED LOGIC ---
    def _run(self, query: str) -> str:
        """Executes the web search."""
        print(f"🔎 Searching the web via DuckDuckGo for: '{query}'")
        # Instantiate the search tool here and call its run method.
        search_executor = DuckDuckGoSearchRun()
        return search_executor.run(query)
    # --- END OF FIX ---


class CSVFinalizerTool(StructuredTool):
    """The final tool that formats the confirmed correct table."""
    name: str = "format_data_to_final_csv"
    description: str = "Takes a single, confirmed correct table as a CSV string and formats it into the final, clean CSV according to detailed rules."
    args_schema: Type[BaseModel] = RawCSVSchema
    llm: ChatVertexAI

    def _run(self, raw_csv_data: str) -> str:
        print("⚙️ Formatting final CSV...")
        formatting_prompt = f"""
        You are a meticulous data extraction expert. Convert the provided raw CSV data into a pristine CSV format.

        ### 1. FINAL OUTPUT REQUIREMENTS
        - The entire response MUST be ONLY the final CSV data, including a header row.
        - The CSV MUST have these exact columns: `primary_submarket,secondary_submarket,property_type,total_inventory_q,vacancy_q,net_absorption_q,under_construction_q,rent_q,delivered_q,leasing_activity_q`

        ### 2. MAPPING & CLEANING RULES
        - The source data has a header. The first column is the `primary_submarket`.
        - Map the other source columns by position and context to the target schema.
        - Convert vacancy percentages to decimals (e.g., 5.4% -> 0.054).
        - Numbers in parentheses `(123)` must be negative `-123`.
        - Remove all quotes, dollar signs, and commas from numbers.

        Raw CSV Data to Process:
        ```csv
        {raw_csv_data}
        ```
        """
        response = self.llm.invoke(formatting_prompt).content
        return response.strip()