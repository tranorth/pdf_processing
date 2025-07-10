# tools/pdf_tools.py

import os
from typing import Type, List

import pandas as pd
import pdfplumber
from langchain.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI
from pydantic.v1 import BaseModel

from .schemas import PDFPathSchema, TableListSchema, RawTableSchema

class PDFTableExtractorTool(StructuredTool):
    """Tool 1: Extracts every table from a PDF."""
    name: str = "extract_all_tables_from_pdf"
    description: str = "Reads a PDF file and extracts every table into a list of raw strings."
    args_schema: Type[BaseModel] = PDFPathSchema

    def _run(self, pdf_path: any, **kwargs: any) -> List[str]:
        if isinstance(pdf_path, dict): path = pdf_path.get('pdf_path')
        else: path = pdf_path
        print(f"🔎 Extracting all tables from: {path}")
        if not path or not os.path.exists(path): return ["Error: PDF file not found."]
        
        all_tables = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        if table:
                            all_tables.append(pd.DataFrame(table).to_csv(index=False, header=False))
            return all_tables
        except Exception as e:
            return [f"Error processing PDF: {e}"]


class TableSelectorTool(StructuredTool):
    """Tool 2: Intelligently selects the correct table from a list."""
    name: str = "select_submarket_statistics_table"
    description: str = "Analyzes a list of tables to select the one containing market statistics by submarket."
    args_schema: Type[BaseModel] = TableListSchema
    llm: ChatVertexAI

    def _run(self, tables: List[str], **kwargs: any) -> str:
        print("🤖 Deciphering which table is correct...")
        context = ""
        for i, table_str in enumerate(tables):
            context += f"--- TABLE {i+1} ---\n{table_str[:500]}\n\n"

        prompt = f"""
        From the following list of tables, identify the single best one that represents 'Market Statistics by Submarket'.
        The correct table has geographic locations like 'Hays County' or 'Southeast' in the first column.
        Respond with ONLY the number of the correct table (e.g., "4").
        {context}
        """
        response = self.llm.invoke(prompt)
        try:
            table_index = int(''.join(filter(str.isdigit, response.content))) - 1
            if 0 <= table_index < len(tables):
                print(f"✅ Selected Table #{table_index + 1}.")
                return tables[table_index]
            return "Error: LLM returned an invalid table number."
        except (ValueError, IndexError):
            return "Error: Could not determine the correct table."


class CSVProcessorTool(StructuredTool):
    """Tool 3: Formats and Validates the selected table, returning the final CSV."""
    name: str = "process_and_validate_table"
    description: str = "Takes the single correct raw table string, formats it, validates it, and returns the final clean CSV."
    args_schema: Type[BaseModel] = RawTableSchema
    llm: ChatVertexAI

    def _run(self, raw_table_data: str, **kwargs: any) -> str:
        print("⚙️ Formatting selected table...")
        formatting_prompt = f"""
        You are a meticulous data extraction expert. Your sole purpose is to convert raw text from a real estate market report table into a pristine, structured CSV format. You must perform a final check on your own work to ensure all rules were followed.

        ### 1. FINAL OUTPUT REQUIREMENTS
        - The entire response MUST be ONLY the CSV data, including a header row.
        - Do NOT include any explanations, comments, or markdown formatting like ```csv.
        - The CSV MUST have these exact columns in this exact order:
        `primary_submarket,secondary_submarket,property_type,total_inventory_q,vacancy_q,net_absorption_q,under_construction_q,rent_q,delivered_q,leasing_activity_q`

        ### 2. COLUMN MAPPING & PRIORITIZATION
        Map the source data to the required CSV columns. Prioritize industrial/warehouse data.

        - `primary_submarket`: Map from 'Submarket', 'County', or a primary geographic area.
        - `secondary_submarket`: Map from 'City' or a more granular geographic area.
        - `property_type`: Map from 'Type' or 'Product Type'. Look for values like 'Warehouse', 'Manufacturing', 'Flex', etc.
        - `total_inventory_q`: Map from 'Net Rentable Area', 'Inventory', 'Total Inventory', etc.
        - `vacancy_q`: Map from 'Vacancy Rate', 'Vacancy %', etc. **Value MUST be converted to a decimal (e.g., 5.4% -> 0.054).**
        - `net_absorption_q`: Map from 'Net Absorption'. **Prioritize a quarterly value over a Year-to-Date (YTD) value.**
        - `under_construction_q`: Map from 'Under Construction', 'Under Const.', etc.
        - `rent_q`: Map from 'Asking Rent'. Prioritize Industrial/Warehouse specific rent.
        - `delivered_q`: Map from 'Deliveries'. **Prioritize a quarterly value over a YTD value.**
        - `leasing_activity_q`: Map from 'SQ FT Leased', 'Gross Absorption', 'Leasing Activity'. **'SQ FT Leased' is the highest priority.**

        ### 3. DATA CLEANING & FORMATTING RULES
        - **No Quotes:** The final CSV must contain NO quotation marks (`"`) at all.
        - **Numbers:** Must be digits only. Remove all commas or currency symbols.
        - **Negative Values:** Numbers in parentheses `(123)` or with a trailing minus `123-` MUST be converted to standard negative numbers `-123`.
        - **Missing Data:** Cells with `N/A`, `-`, or that are empty should be a completely empty field in the CSV.
        - **Row Integrity:** Extract ALL rows from the source table, including totals and subtotals.

        Raw Table Text to Process:
        {raw_table_data}
        """
        generated_csv = self.llm.invoke(formatting_prompt).content.strip()
        if not generated_csv:
            return "Error: Failed to generate CSV from the data."

        print("✅ Validating final CSV...")
        validation_prompt = f"""
        You are a data validation expert. Review the "Generated CSV". Does it appear to be a correctly formatted and complete submarket statistics table?
        Respond with a brief, one-sentence confirmation or state any obvious errors.
        Generated CSV:
        {generated_csv}
        """
        report = self.llm.invoke(validation_prompt).content
        print(f"--- Validation Report ---\n{report}")
        
        return generated_csv # Return just the clean CSV