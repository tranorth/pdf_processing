#main.py

import os
import io
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import the function that builds your agent
from agent import create_pdf_agent

def process_pdf_with_agent(pdf_path: str, agent_executor):
    """
    Invokes the agent for a single PDF file and returns a DataFrame on success.
    """
    print(f"\n--- Starting processing for: {os.path.basename(pdf_path)} ---")
    try:
        # The agent's job is to return a clean CSV string
        result = agent_executor.invoke({
            "input": f"Fully process the PDF file located at the following path: {pdf_path}"
        })
        
        # The agent's final output should be the CSV data string
        csv_output_string = result.get('output', '')

        # Convert the CSV string into a pandas DataFrame
        if csv_output_string and isinstance(csv_output_string, str):
            df = pd.read_csv(io.StringIO(csv_output_string))
            print(f"✅ Successfully created DataFrame for: {os.path.basename(pdf_path)}")
            # Return the DataFrame
            return df
        else:
            print(f"❌ Agent did not return valid CSV data for {os.path.basename(pdf_path)}.")
            # Return None to indicate failure
            return None
            
    except Exception as e:
        print(f"❌ An error occurred processing {os.path.basename(pdf_path)}: {e}")
        # Return None on error
        return None

if __name__ == "__main__":
    report_dir = "data"
    if not os.path.isdir(report_dir):
        print(f"Directory not found: {report_dir}. Please create it and add your PDFs.")
    else:
        pdf_files = list(Path(report_dir).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {report_dir}.")
        else:
            pdf_agent_executor = create_pdf_agent()
            
            # Create a list to hold the resulting DataFrames
            all_dataframes = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_pdf_with_agent, str(file), pdf_agent_executor) for file in pdf_files]
                
                for future in futures:
                    # Capture the returned DataFrame from each function call
                    result_df = future.result()
                    if result_df is not None:
                        # Add the successful DataFrame to our list
                        all_dataframes.append(result_df)
            
            # After all files are processed, combine the results
            if all_dataframes:
                print("\n\n--- Combining all results into a single DataFrame ---")
                # Use pd.concat to merge all DataFrames in the list
                master_df = pd.concat(all_dataframes, ignore_index=True)
                
                print(master_df)
                
                # Save the final combined DataFrame to a single CSV file
                master_df.to_csv("master_output.csv", index=False)
                print("\nSaved combined data to master_output.csv")
            else:
                print("\nNo data was successfully processed.")