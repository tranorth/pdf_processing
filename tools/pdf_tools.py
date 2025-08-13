# tools/pdf_tools.py

import os
from typing import Type, List

import pandas as pd
import pdfplumber
from langchain.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI
from pydantic.v1 import BaseModel

from .schemas import PDFPathSchema, TableListSchema, RawCSVSchema, FinalCSVSchema

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
            return []

        all_tables_as_csv = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    for table_data in page.extract_tables():
                        if table_data and len(table_data) > 1:
                            headers = table_data[0]
                            df = pd.DataFrame(table_data[1:], columns=headers)

                            if df.columns[0] is None or 'Unnamed' in str(df.columns[0]):
                                df.rename(columns={df.columns[0]: 'primary_submarket'}, inplace=True)
                            
                            csv_string = df.to_csv(index=False)
                            all_tables_as_csv.append(csv_string)
                            
                            print(f"\n--- Extracted Table from Page {i+1} ---")
                            print(csv_string[:400] + "...")
            return all_tables_as_csv
        except Exception as e:
            print(f"Error during table extraction: {e}")
            return []


class TableSelectorTool(StructuredTool):
    """Tool 2: Intelligently selects the correct table from a list."""
    name: str = "select_submarket_statistics_table"
    description: str = "Analyzes a list of tables (as CSV strings) and selects the one containing market statistics by submarket."
    args_schema: Type[BaseModel] = TableListSchema
    llm: ChatVertexAI

    def _run(self, tables_as_csv: List[str], **kwargs: any) -> str:
        print("\n🤖 Deciphering which table is correct...")
        context_for_llm = ""
        for i, table_csv in enumerate(tables_as_csv):
            context_for_llm += f"--- TABLE {i+1} ---\n{table_csv[:350]}\n\n"

        prompt = f"""
        From the following list of tables, identify the single best one that represents 'Market Statistics by Submarket'.
        The correct table has geographic locations like counties or directional areas (e.g., 'Southeast') in the first column.
        Do NOT choose tables about 'Notable Projects', 'by Size', or 'by Class'.

        {context_for_llm}

        Which table number is the correct one? Respond with ONLY the number (e.g., "4").
        """
        response = self.llm.invoke(prompt)
        try:
            table_index = int(''.join(filter(str.isdigit, response.content))) - 1
            if 0 <= table_index < len(tables_as_csv):
                print(f"✅ AI selected Table #{table_index + 1}.")
                return tables_as_csv[table_index]
            return "Error: LLM returned an invalid table number."
        except (ValueError, IndexError):
            return "Error: Could not determine the correct table from the LLM's response."


class CSVGeneratorTool(StructuredTool):
    """Tool 3: Formats the selected table into a CSV string."""
    name: str = "generate_formatted_csv"
    description: str = "Takes a single, confirmed correct table as a CSV string and applies detailed formatting rules."
    args_schema: Type[BaseModel] = RawCSVSchema
    llm: ChatVertexAI

    def _run(self, raw_csv_data: str, **kwargs: any) -> str:
        print("\n⚙️ Formatting CSV with detailed rules...")
        formatting_prompt = f"""
        You are a meticulous data extraction expert. Convert the provided raw CSV data into a pristine CSV format.
        Your output should be the formatted CSV data, wrapped in a ```csv markdown block.

        ### FINAL OUTPUT COLUMNS
        The final CSV MUST have these exact columns: `primary_submarket,secondary_submarket,property_type,total_inventory_q,vacancy_q,net_absorption_q,under_construction_q,rent_q,delivered_q,leasing_activity_q`
        
        make sure that youre double and triple checking your answers / output. 



        if there are multiple tables, we only need the ones for industrial / warehouse



        Analyze these files and then tell me which page the submarket data is on, then i will need you to extract the data from each file and put them into their respective csv format with columns as close to this as possible if applicable: primary_submarket =submarket, county, etc



        secondary_submarket = city, etc



        property_type = warehouse, manufacturing, flex, bulk, class a, b etc


        total_inventory_q = net rentable Area, bldg sqft, inventory etc



        vacancy_q = vacancy rate etc, (keep as a decimal)



        net_absorption_q = ytd net absorbtion, year quarter deliveries(the quarter version is priority over ytd), etc



        under_construction_q = under construction, under construct etc



        Rent_q = median asking rent, average asking rent etc (Industrial/ Warehouse/ Whs. is priority)



        delivered_q = ytd deliveries, quarter deliveries(the quarter version is priority over ytd), etc.



        leasing_activity_q = sq ft leased(sqft leased is priority), gross absorption(this is secondary priority if sqft leased column is not available not value so if there is a sqft leased column use that value even if nothing is there), gross activity column should be a last result, etc





        if there are more than one options for a csv column then use best judgement to ascertain which one would best fit the csv, you can analyze other files to check if something should be somewhere and make sure to include the market total rows and submarket total rows if applicable, make sure to include the market total row at the bottom



        If possible we only need things to do with industrial warehouse and/or logistics



        these are the columns i need even if the files dont have data values for each column and keep it to this: primary_submarket,secondary_submarket,property_type,total_inventory_q,vacancy_q,net_absorption_q,under_construction_q,rent_q,delivered_q,leasing_activity_q



        make sure to do your due diligence of the data making sure the values are under the correct column and accuracy of the data, reanalyze the data of you have to and give yourself a score from 1 to 100



        all n/a should be 0 or blank(blank is priority)

        all - should be 0 or blank(blank is priority)

        all numbers in parenthesis should be negative

        no numbers should have "" quotes around them

        no names should have "" quotes around them

        there should be any "" in the data

        get rows even if they dont have data in them so we can have table completion

        dont make up numbers for the leasing column if they aren't there then just leave blank, same for deliveries as well

        make sure to include both primary and secondary submarkets if and when needed but dont include the market as primary submarket unless its necessary to differentiate between subsections



        Industrial Data is the priority
        
        Raw CSV Data to Process:
        {raw_csv_data}
        """
        response = self.llm.invoke(formatting_prompt).content
        return response.strip()


class FinalOutputTool(StructuredTool):
    """Tool 4: Cleans the final output to be a valid, raw CSV string ONLY."""
    name: str = "create_final_csv_output"
    description: str = "Takes the potentially messy output from the generation step and cleans it to be a valid, raw CSV string with no extra text."
    args_schema: Type[BaseModel] = FinalCSVSchema

    def _run(self, generated_csv: str, **kwargs: any) -> str:
        print("\n🧼 Cleaning final output...")
        try:
            # Find the start of the CSV header and take everything after it
            start_index = generated_csv.index("primary_submarket")
            clean_csv = generated_csv[start_index:]
            # Remove any trailing markdown
            clean_csv = clean_csv.replace("```", "").strip()
            
            print("--- Final Clean CSV ---")
            print(clean_csv)
            
            return clean_csv
        except ValueError:
            return "Error: Could not find the CSV header in the final output."