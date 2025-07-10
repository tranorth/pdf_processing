#pdf_tools.py

import os
from typing import Type

import pandas as pd
import pdfplumber
from langchain.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI
from pydantic.v1 import BaseModel

from .schemas import DataExtractorSchema, TableFinderSchema

class TableFinderTool(StructuredTool):
    """A tool to intelligently find and extract relevant tables from a PDF."""
    name: str = "find_pdf_submarket_tables"
    description: str = "Scans a PDF file to find and return the raw text of tables containing detailed industrial submarket statistics."
    args_schema: Type[BaseModel] = TableFinderSchema
    llm: ChatVertexAI

    # def __init__(self, llm: ChatVertexAI, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.llm = llm

    def _run(self, pdf_path: any, **kwargs: any) -> str:
        if isinstance(pdf_path, dict):
            path = pdf_path.get('pdf_path')
        else:
            path = pdf_path

        print(f"🔎 Intelligently finding tables in: {path}")
        if not path or not os.path.exists(path):
            return f"Error: PDF file not found at {path}"

        relevant_tables = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    for table_data in page.extract_tables():
                        if not table_data or not table_data[0]: continue
                        
                        headers = ", ".join(filter(None, table_data[0]))
                        prompt = f"""
                        The following are column headers from a table: "{headers}"
                        Does this appear to be a detailed real estate market statistics table, organized by submarket? It should contain columns related to inventory, vacancy, absorption, and rent.
                        Answer with only the word YES or NO.
                        """
                        response = self.llm.invoke(prompt)

                        if "YES" in response.content.upper():
                            headers = table_data[0]
                            if headers and (headers[0] is None or headers[0].strip() == ""):
                                headers[0] = "primary_submarket"
                            df = pd.DataFrame(table_data[1:], columns=headers)
                            df_as_csv = df.to_csv(index=False)
                            relevant_tables.append(f"--- Table from Page {page_num + 1} ---\n{df_as_csv}\n")

            return "\n".join(relevant_tables) if relevant_tables else "No relevant submarket statistics tables were found."
        except Exception as e:
            return f"Error processing PDF with pdfplumber: {e}"


class FormatAndValidateTool(StructuredTool):
    """
    Takes raw table text, converts it to a clean CSV using detailed rules, 
    validates the result, and returns a final report including the CSV.
    """
    name: str = "format_and_validate_table"
    description: str = "Takes raw table text, formats it into a clean CSV, validates it, and returns the final report."
    args_schema: Type[BaseModel] = DataExtractorSchema
    llm: ChatVertexAI

    # def __init__(self, llm: ChatVertexAI, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.llm = llm

    def _run(self, raw_table_data: str, **kwargs: any) -> str:
        # --- Step 1: Format the data using your detailed prompt ---
        print("🤖 Extracting and formatting data with detailed rules...")
        
        # This is your detailed prompt, cleaned and consolidated.
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
        generated_csv = self.llm.invoke(formatting_prompt).content
        
        if not generated_csv.strip():
            return "Error: Failed to generate CSV. The data might have been unclear."

        # --- Step 2: Validate the result ---
        print("✅ Validating final CSV...")
        validation_prompt = f"""
        You are a data validation expert. Compare the "Raw Data" with the "Generated CSV" and briefly report on the outcome.
        1. If the CSV is accurate, respond with: "Validation successful. CSV is accurate."
        2. If data was changed during formatting (e.g., removing commas), briefly state what was done.
        
        Raw Data:
        {raw_table_data}
        Generated CSV:
        {generated_csv}
        Your Validation Report:
        """
        report = self.llm.invoke(validation_prompt).content

        # --- Step 3: Combine and return the final output ---
        print(f"--- Validation Report ---\n{report}") # Print the report to the console
        return generated_csv # Return just the CSV string