# agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI

from tools.pdf_tools import PDFTableExtractorTool, TableSelectorTool, CSVFinalizerTool

def create_pdf_agent():
    """Builds and returns the PDF processing agent."""
    llm = ChatVertexAI(model='gemini-2.5-pro', temperature=0)

    tools = [
        PDFTableExtractorTool(),
        TableSelectorTool(llm=llm),
        CSVFinalizerTool(llm=llm),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a powerful PDF processing assistant. You have three tools to help you."
            "You must follow these steps in order:"
            "1. Use the `extract_all_tables_from_pdf` tool to get a list of all tables from the PDF."
            "2. Use the `select_submarket_statistics_table` tool with the list from step 1 to choose the single correct table."
            "3. Use the `format_data_to_final_csv` tool with the selected table from step 2 to get the final result."
        )),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )
    return executor