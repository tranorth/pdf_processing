# main.py

import os
import io
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from agent import create_pdf_agent

def process_pdf_with_agent(pdf_path: str, agent_executor):
    """
    Invokes the agent for a single PDF file and returns a DataFrame on success.
    """
    filename = os.path.basename(pdf_path)
    print(f"\n--- 🚀 Starting processing for: {filename} ---")
    try:
        # The agent's job is to return a clean CSV string in the 'output' key
        result = agent_executor.invoke({
            "input": f"Fully process the PDF file located at the following path: {pdf_path}. The main geographic area is likely '{filename.split('_')[0]}'."
        })
        
        csv_output_string = result.get('output', '')

        # More robust check to ensure the output is a valid, non-empty CSV string
        if csv_output_string and isinstance(csv_output_string, str) and csv_output_string.strip().startswith('primary_submarket'):
            # Use io.StringIO to read the CSV string directly into a DataFrame
            df = pd.read_csv(io.StringIO(csv_output_string))
            print(f"✅ Successfully created DataFrame for: {filename}")
            return df
        else:
            print(f"❌ Agent did not return valid CSV data for {filename}.")
            print(f"   Received: {str(csv_output_string)[:200]}...") # Print a snippet of what was received
            return None
            
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {filename}: {e}")
        # Consider logging the full traceback here if needed
        # import traceback
        # traceback.print_exc()
        return None

if __name__ == "__main__":
    report_dir = r"pdf_processing/2025 Q2"
    if not os.path.isdir(report_dir):
        print(f"Directory not found: {report_dir}. Please create it and add your PDF files.")
    else:
        pdf_files = list(Path(report_dir).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in the '{report_dir}' directory.")
        else:
            pdf_agent_executor = create_pdf_agent()
            
            # List to hold the resulting DataFrames from each successful run
            all_dataframes = []
            
            # Using a thread pool to process files concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Create a future for each PDF processing task
                future_to_pdf = {executor.submit(process_pdf_with_agent, str(file), pdf_agent_executor): file for file in pdf_files}
                
                for future in as_completed(future_to_pdf):
                    # As each task completes, get the result
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        # Add the successful DataFrame to our list
                        all_dataframes.append(result_df)
            
            # After all files are processed, combine the results
            if all_dataframes:
                print("\n\n--- 📊 Combining all results into a single DataFrame ---")
                # Use pd.concat to merge all DataFrames in the list
                master_df = pd.concat(all_dataframes, ignore_index=True)
                
                print(f"Combined DataFrame has {master_df.shape[0]} rows and {master_df.shape[1]} columns.")
                print(master_df.head())
                
                # Save the final combined DataFrame to a single CSV file
                output_path = "master_output.csv"
                master_df.to_csv(output_path, index=False)
                print(f"\n✅ Saved combined data to {output_path}")
            else:
                print("\n\n--- ⚠️ No data was successfully processed from any PDF files. ---")