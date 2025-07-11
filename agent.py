# agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI

from tools.pdf_tools import PDFTableExtractorTool, TableSelectorTool, CSVGeneratorTool, FinalOutputTool

def create_pdf_agent():
    """Builds and returns the PDF processing agent."""
    llm = ChatVertexAI(model='gemini-2.5-pro', temperature=0)

    tools = [
        PDFTableExtractorTool(),
        TableSelectorTool(llm=llm),
        CSVGeneratorTool(llm=llm),
        FinalOutputTool(),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a powerful PDF processing assistant. You have four tools to help you."
            "You must follow these steps in order:"
            "1. Use `extract_all_tables_from_pdf` to get a list of all tables."
            "2. Use `select_submarket_statistics_table` with that list to choose the correct table."
            "3. Use `generate_formatted_csv` with the selected table to create the formatted CSV."
            "4. Use `create_final_csv_output` on the result of the previous step to clean it up and produce the final answer. This MUST be your final step."
        )),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
    return executor