# tools/pdf_tools.py

import os
from typing import Type

import pdfplumber
from langchain.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from pydantic.v1 import BaseModel

from .schemas import PDFProcessorSchema

class PDFProcessorTool(StructuredTool):
    """A single tool that extracts text and tables from a PDF, then calls the LLM for processing."""
    name: str = "process_pdf_for_market_data"
    description: str = "Takes a path to a PDF, extracts all text and tables, and formats the submarket data into a clean CSV."
    args_schema: Type[BaseModel] = PDFProcessorSchema
    llm: ChatVertexAI

    def _run(self, pdf_path: str, **kwargs: any) -> str:
        if isinstance(pdf_path, dict):
            path = pdf_path.get('pdf_path')
        else:
            path = pdf_path

        print(f"⚙️ Extracting text and tables from {path}...")
        if not path or not os.path.exists(path):
            return "Error: PDF file not found."

        try:
            full_pdf_text = ""
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract plain text from the page
                    page_text = page.extract_text() or ""
                    
                    # Extract tables and format them as Markdown
                    tables = page.extract_tables()
                    tables_as_markdown = ""
                    for table in tables:
                        # Filter out None values which can break the join operation
                        cleaned_table = [list(map(lambda cell: str(cell) if cell is not None else "", row)) for row in table]
                        
                        # Create Markdown table
                        if cleaned_table:
                            headers = " | ".join(cleaned_table[0])
                            separator = " | ".join(["---"] * len(cleaned_table[0]))
                            rows = "\n".join([" | ".join(row) for row in cleaned_table[1:]])
                            tables_as_markdown += f"\n\n--- Table on Page {i+1} ---\n| {headers} |\n| {separator} |\n{rows}\n--- End Table ---\n\n"
                    
                    # Combine the page text and its tables
                    full_pdf_text += f"--- Page {i+1} Text ---\n{page_text}\n{tables_as_markdown}"

            # Your verbatim prompt, now with the full text context
            user_prompt_text = """
            make sure that youre double and triple checking your answers / output. 

            if there are multiple tables, we only need the ones for industrial / warehouse
            
            Analyze and then figure out where the submarket data in the tables are on, then i will need you to extract the data and put them into their respective csv format with columns as close to this as possible if applicable: primary_submarket =submarket, county, etc
            
            secondary_submarket = city,  etc 
            
            property_type = warehouse, manufacturing, flex, bulk, class a, b etc,  
            
            total_inventory_q = net rentable Area, bldg sqft, inventory etc 
            
            vacancy_q = vacancy rate etc, (keep as a decimal) 
            
            net_absorption_q = ytd net absorbtion, year quarter deliveries(the quarter version is priority over ytd), etc 
            
            under_construction_q = under construction, under construct etc 
            
            Rent_q = median asking rent, average asking rent etc (Industrial/ Warehouse/ Whs. is priority)
            
            delivered_q = ytd deliveries, quarter deliveries(the quarter version is priority over ytd), etc.
            
            leasing_activity_q = sq ft leased(sqft leased is priority), gross absorption(this is secondary priority if sqft leased column is not available not value so if there is a sqft leased column use that value even if nothing is there), gross activity column should be a last result, etc
            
            
            if there are more than one options for a csv column then use best judgement to ascertain which one would best fit the csv, you can analyze other files to check if something should be somewhere and make sure to include the market total rows and submarket total rows if applicable, make sure to include the market total row at the bottom
            
            also you can decide which table is the submarket data by searching the first column and determining if the valuse coincide with geographic locations with in the overall market, be careful to get the first column data because many times it doesnt have a value in the header row but still has values in the subsequent rows.

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
            """
            
            message = HumanMessage(content=user_prompt_text)
            
            print("✅ Content extracted, invoking the Gemini model for analysis...")
            response = self.llm.invoke([message])
            
            # Clean the markdown from the response
            clean_response = response.content.replace("```csv", "").replace("```", "").strip()
            return clean_response
            
        except Exception as e:
            return f"An unexpected error occurred in the PDF processing tool: {e}"